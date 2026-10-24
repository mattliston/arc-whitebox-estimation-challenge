"""Root-only fixed production-width accuracy/repeat primitive; never a model run.

CLI: NATIVE -B probe.py EXCLUSIVE_RESULT_JSON
Root must guard45s/7.5GB/zero-added-swap, with all BLAS thread variables1.
"""
from pathlib import Path
import hashlib
import json
import os
import resource
import sys
import time
import types
import numpy as np
import flopscope as flops

D = Path(__file__).resolve().parent
NATIVE = D.parent.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
CANDIDATE_SHA = '88683ee1ae144556393ba83e013ba975c63a2b91469fce5c6aee9f8f0c4db5d6'
PARENT_SHA = 'e82f7bf1fbe736d37816492e42c456198df07006152436a299b36eee757e2341'
BUILD_SHA = '646bbd8265b6febda17cb204d99cb3407b8e83211274fd902fea17aebee5441b'
THREADS = ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')


def pin(path):
    path = Path(path).resolve(); raw=path.read_bytes()
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def checked(record):
    assert pin(record['path']) == record,record['path']


def summary(ctx):
    value=ctx.summary_dict()
    return dict(flops=int(value['flops_used']),
        operations=sum(int(v.get('calls',v.get('count',0))) for v in value['operations'].values()))


def rss():
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform=='darwin' else value*1024)


def main():
    assert len(sys.argv)==2
    target=Path(sys.argv[1]).resolve()
    assert not target.exists() and target.parent.is_dir()
    assert Path(sys.executable).absolute()==NATIVE and np.__version__=='2.4.6'
    assert all(os.environ.get(k)=='1' for k in THREADS)
    build_pin=pin(D/'build.json')
    assert build_pin['sha256']==BUILD_SHA
    build=json.loads((D/'build.json').read_text())
    for key in ('parent','candidate','builder','protocol','fragment','pricing',
                'side_dependency','parent_build'):
        checked(build[key])
    assert build['candidate']['sha256']==CANDIDATE_SHA
    assert build['parent']['sha256']==PARENT_SHA
    assert build['edited_prefix_plus_append'] is True and build['source_inverse_exact'] is True
    source=Path(build['candidate']['path']); parent=Path(build['parent']['path'])
    edited=parent.read_text()
    for edit in build['edits']:
        assert edited.count(edit['before'])==1
        edited=edited.replace(edit['before'],edit['after'],1)
    assert source.read_text()==edited+'\n'+Path(build['fragment']['path']).read_text()
    inverse=edited
    for edit in reversed(build['edits']):
        assert inverse.count(edit['after'])==1
        inverse=inverse.replace(edit['after'],edit['before'],1)
    assert inverse==parent.read_text()
    pricing=json.loads(Path(build['pricing']['path']).read_text())
    assert pricing['child_warm_per_helper']==717329488
    assert pricing['parent_warm_per_helper']==709895628
    stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{})
    sys.modules['whestbench']=stub
    mod=types.ModuleType('gaussian3_symdiag128_candidate');mod.__file__=str(source)
    sys.modules[mod.__name__]=mod
    exec(compile(source.read_text(),str(source),'exec'),mod.__dict__)
    t0=time.monotonic()
    mod.Estimator().setup(types.SimpleNamespace(width=1024,depth=16,flop_budget=2**41,
        api_version='2',scratch_dir=None,submission_dir=None,seed=0))
    setup_s=time.monotonic()-t0
    assert 0<=setup_s<=4.1
    assert [r[1] for r in mod._PB_LOG if r[0]=='bound']==[682]
    assert [r[1] for r in mod._PB_LOG if r[0]=='lvfuse']==[649]
    assert [r[1] for r in mod._PB_LOG if r[0]=='psdb_fused']==[9]
    assert len(mod._PSDB_SIDES)==9 and mod._PACKED_ELEMENTS==391602176
    assert sum(len(mod.__dict__[name]) for name in mod._PB_CACHES)==682
    assert all(side._borrowed_arena is mod._SCRATCH[0] for side in mod._PSDB_SIDES.values())
    setup_log=list(mod._PB_LOG)
    fnp=mod.fnp
    records=[]
    for kind,seed in (('integer_gram',202609285),('gaussian_covariance',202609286)):
        rng=np.random.default_rng(seed);stage_one=None
        if kind=='integer_gram':
            w=(2*rng.integers(0,2,size=(1024,1024))-1).astype(np.float32)
            a,b=w.T,w.copy()
        else:
            w=(rng.standard_normal((1024,1024))/32).astype(np.float32)
            h=(rng.standard_normal((1024,1024))/32).astype(np.float32)
            cov=h@h.T
            with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
                cw=mod._mm(fnp.asarray(cov),fnp.asarray(w))
            stage_one=summary(ctx)
            a,b=w.T,np.asarray(cw).copy()
            del h,cov,cw
        reference=a.astype(np.float64)@b.astype(np.float64)
        inputs=[hashlib.sha256(x.tobytes()).hexdigest() for x in (a,b)]
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
            aa,bb=fnp.asarray(a),fnp.asarray(b)
        conversions=summary(ctx)
        runs=[];first_hash=None
        methods=(('child_first',mod._psdb_sym_upper),('child_repeat',mod._psdb_sym_upper))
        for label,method in methods:
            mod._TR_LAST[0]=None
            t0,c0=time.monotonic(),time.process_time()
            with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
                out=method(aa,bb)
            wall,cpu=time.monotonic()-t0,time.process_time()-c0
            value=np.asarray(out).copy()
            assert value.shape==(1024,1024) and value.dtype==np.float32 and np.isfinite(value).all()
            output_hash=hashlib.sha256(value.tobytes()).hexdigest()
            if first_hash is None:first_hash=output_hash
            assert output_hash==first_hash,'child repeat byte parity failed: '+label
            relative=float(np.linalg.norm(value.astype(np.float64)-reference)/np.linalg.norm(reference))
            exact=bool(np.array_equal(value.astype(np.float64),reference))
            assert exact if kind=='integer_gram' else relative<=2e-4
            assert inputs==[hashlib.sha256(np.asarray(x).tobytes()).hexdigest() for x in (aa,bb)]
            bill=summary(ctx)
            assert bill['flops']==717329488
            assert len(mod._PSDB_SIDES)==9
            assert all(side._borrowed_arena is mod._SCRATCH[0] for side in mod._PSDB_SIDES.values())
            assert mod._TR_LAST[0] is None
            runs.append(dict(label=label,wall_s=wall,cpu_s=cpu,process_peak_rss_bytes=rss(),
                output_sha256=output_hash,repeat_bytes_exact=True,
                relative_frobenius_error=relative,integer_reference_exact=exact,**bill))
            del out,value
        records.append(dict(kind=kind,seed=seed,input_sha256=inputs,
            input_strides=[list(x.strides) for x in (a,b)],stage_one=stage_one,
            conversions=conversions,runs=runs))
        print(json.dumps(dict(event='case_complete',**records[-1])),flush=True)
        del a,b,aa,bb,reference,w
    assert 0<rss()<7500000000
    mod._TR_LAST[0]=None
    mod._pb_reset((None,None))
    assert mod._SCRATCH is None and not mod._PSDB_SIDES
    assert all(not mod.__dict__[name] for name in mod._PB_CACHES)
    for key in ('parent','candidate','builder','protocol','fragment','pricing',
                'side_dependency','parent_build'):
        checked(build[key])
    checked(build_pin)
    result=dict(passed=True,source=build['candidate'],parent=build['parent'],build=build_pin,
        protocol=build['protocol'],fragment=build['fragment'],pricing=build['pricing'],probe=pin(__file__),
        setup_s=setup_s,pb_log=setup_log,cases=records,process_peak_rss_bytes=rss(),
        all_repeat_bytes_exact=True,side_pool_reset_passed=True,
        parent_byte_parity_required=False,
        rounding_note='Changed diagonal partition can change rounded results; dense-reference gates replace cross-parent byte equality.',
        cold_cost_note='All nine batch sides are setup-bound; setup time/counts are checked and the extra bind tariff is pinned in pricing. No hypothetical prediction cold saving is claimed.',
        cpu_and_wall_diagnostic_only=True,external_guard_required=True,
        model_executed=False,network_targets_read=False,serving_qualified=False)
    with target.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print('GAUSSIAN3_SYMDIAG128_FIXTURES_COMPLETE',flush=True)


if __name__=='__main__':main()
