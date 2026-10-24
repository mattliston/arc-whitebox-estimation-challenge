"""Actual inherited-kernel certificates against independent dense algebra."""
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import types
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_info

HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
build = json.loads((HERE/'build.json').read_text())
assert sha(HERE/'candidate.py') == build['candidate_sha256']
assert sha(HERE/'krylov_fnp.py') == build['fragment_sha256']
assert all(t['num_threads'] == 1 for t in threadpool_info())
stub = types.ModuleType('whestbench')
stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
rng = np.random.default_rng(2026092005)
f32 = lambda x: np.asarray(x, dtype=np.float32)
fixtures = []
for name,n,m,old,low_rank in [('joining_only',448,1152,False,False),
                              ('joining_and_old',448,576,True,False),
                              ('rank_deficient',448,384,False,True),
                              ('zero',384,384,False,False)]:
    if name == 'zero':
        J = np.zeros((n,m), np.float32)
    elif low_rank:
        J = f32(rng.normal(size=(n,32)) @ rng.normal(size=(32,m)) / np.sqrt(32*m))
    else:
        J = f32(rng.normal(size=(n,m)) / np.sqrt(m))
    weight = f32(rng.uniform(.25,1.25,size=m))
    omega = f32(rng.normal(size=(n,192)))
    if old:
        Qp = f32(np.linalg.qr(rng.normal(size=(n,384)), mode='reduced')[0])
        h = rng.normal(size=(384,384))
        core = f32(h@h.T/384 + .5*np.eye(384))
    else:
        Qp = core = None
    jd, wd = J.astype(np.float64), weight.astype(np.float64)
    G = (jd*wd[None,:])@jd.T
    if old:
        qd = Qp.astype(np.float64)
        G += qd @ core.astype(np.float64) @ qd.T
    first = G @ omega.astype(np.float64)
    second = G @ first
    fixtures.append((name,(J,weight,Qp,core,omega),(first,second)))


def scaled_error(actual,expected):
    denominator = float(np.linalg.norm(expected))
    numerator = float(np.linalg.norm(actual-expected))
    return numerator/denominator if denominator else numerator


def run(index):
    spec = importlib.util.spec_from_file_location('krylov_certificate', HERE/'candidate.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    outputs,rows,rejections = [],[],[]
    with flops.BudgetContext(flop_budget=2**39, quiet=True) as ctx:
        spec.loader.exec_module(mod)
        for name,inputs,expected in fixtures:
            jj,ww,qp,cc,oo = [None if x is None else fnp.asarray(x.copy()) for x in inputs]
            arrays = [x for x in (jj,ww,qp,cc,oo) if x is not None]
            before = [np.asarray(x).tobytes() for x in arrays]
            applied = np.asarray(mod._krylov384_apply(jj,ww,qp,cc,oo)).copy()
            q = np.asarray(mod._krylov384_basis(jj,ww,qp,cc,oo)).copy()
            assert q.dtype == np.float32 and q.shape == (inputs[0].shape[0],384)
            assert np.isfinite(q).all() and np.isfinite(applied).all()
            assert before == [np.asarray(x).tobytes() for x in arrays]
            qd = q.astype(np.float64)
            orth = float(np.linalg.norm(qd.T@qd-np.eye(384))/np.sqrt(384))
            action = scaled_error(applied.astype(np.float64), expected[0])
            inclusion = [scaled_error(qd@(qd.T@y), y) for y in expected]
            assert action <= 2e-4, (name,'action',action)
            assert max(inclusion) <= 2e-4, (name,'inclusion',inclusion)
            assert orth <= 5e-6, (name,'orthogonality',orth)
            if name == 'zero':
                assert np.count_nonzero(applied) == 0
            rows.append(dict(case=name, n=q.shape[0], join_columns=inputs[0].shape[1],
                             old_core=qp is not None, operator_relative_error=action,
                             normalized_orthogonality=orth,
                             krylov_block_inclusion_errors=inclusion,
                             input_bytes_unchanged=True))
            outputs.extend([applied,q])
        jj,ww,qp,cc,oo = [None if x is None else fnp.asarray(x.copy())
                           for x in fixtures[1][1]]
        bad = [('weight_length',(jj,ww[:-1],qp,cc,oo)),
               ('omega_width',(jj,ww,qp,cc,oo[:,:191])),
               ('weight_dtype',(jj,ww.astype(fnp.float64),qp,cc,oo)),
               ('old_core_ownership',(jj,ww,qp,None,oo)),
               ('old_core_shape',(jj,ww,qp,cc[:383],oo)),
               ('too_few_rows',(jj[:383],ww,None,None,oo[:383]))]
        for name,args in bad:
            try:
                mod._krylov384_basis(*args)
            except RuntimeError:
                rejections.append(name)
            else:
                raise AssertionError(name)
    bill = ctx.summary_dict()['flops_used']
    del sys.modules[spec.name]
    del mod
    gc.collect()
    return outputs,rows,rejections,bill


arrays,rows,rejections,bill = run(0)
again,rows2,rejections2,bill2 = run(1)
assert rows == rows2 and rejections == rejections2 and bill == bill2
assert all(x.tobytes()==y.tobytes() for x,y in zip(arrays,again))
out = HERE/'unit_v1'
out.mkdir(exist_ok=False)
record = dict(passed=True, cases=rows, actual_inherited_kernel_used=True,
              repeated_arrays_byte_exact=True, repeated_fresh_module_bills_exact=True,
              complete_bill_per_batch=bill, rejections=rejections,
              source_sha256=sha(__file__), candidate_sha256=build['candidate_sha256'],
              target_arrays_opened=False,
              max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
with (out/'result.json').open('x') as stream:
    json.dump(record,stream,indent=2)
print(json.dumps(record), flush=True)
