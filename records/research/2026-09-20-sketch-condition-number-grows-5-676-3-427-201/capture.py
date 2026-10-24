import argparse
import importlib.util
import resource
import sys
import time
import types
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_info
from common import HERE, read, write, pin, verify, seal_check, weights_path
from snapshot import clone_fields

ap = argparse.ArgumentParser(); ap.add_argument('label'); ap.add_argument('out')
args = ap.parse_args()
seal_check(); assert read(HERE / 'unit.json')['passed']
assert all(x['num_threads'] == 1 for x in threadpool_info())
control = next(x for x in read(HERE / 'inputs.json')['controls'] if x['label'] == args.label)
for key in ['source', 'output', 'receipt']:
    verify(control[key])
out = HERE / args.out; out.mkdir(exist_ok=False)
weights_file = weights_path(38)
weights = np.load(weights_file, allow_pickle=False)
assert weights.shape == (16, 1024, 1024) and weights.dtype == np.float32
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ingress:
    ww = [fnp.asarray(w) for w in weights]
assert ingress.summary_dict()['flops_used'] == 0
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
spec = importlib.util.spec_from_file_location('sketch_trace_estimator', control['source']['path'])
mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod
mlp = types.SimpleNamespace(width=1024, depth=16, weights=ww, seed=0, name='rank38')
held = []; copy_flops = 0
original_qr = fnp.linalg.qr
begin = time.monotonic()
try:
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        spec.loader.exec_module(mod)
        def traced_qr(y, mode='reduced'):
            global copy_flops
            frame = sys._getframe(1)
            assert frame.f_code is mod._shared_old384_push.__code__ and mode == 'reduced'
            loc = frame.f_locals
            layer = int(loc['layer'])
            assert layer == 5 + len(held) and loc['Y'] is y
            fields = {k: loc[k] for k in ['J', 'weight', 'Omega', 'previous']}
            fields['Y'] = y
            if loc['old_columns']:
                fields.update(Qp=loc['Qp'], core=loc['state']['core'])
            before = ctx.summary_dict()['flops_used']
            saved = clone_fields(fields)
            copy_flops += ctx.summary_dict()['flops_used'] - before
            q, r = original_qr(y, mode=mode)
            before = ctx.summary_dict()['flops_used']
            saved.update(clone_fields(dict(Q=q, R=r)))
            copy_flops += ctx.summary_dict()['flops_used'] - before
            held.append((layer, saved))
            del frame, loc, fields
            return q, r
        fnp.linalg.qr = traced_qr
        prediction = mod.Estimator().predict(mlp, 2**41)
finally:
    fnp.linalg.qr = original_qr
elapsed = time.monotonic() - begin
bill = ctx.summary_dict()
assert len(held) == 10 and mod._W5_PORT_SCOPE is None and mod._W5_PORT_LAST['numeric_cache_retired']
assert bill['flops_used'] - copy_flops == control['flops']
output = np.asarray(prediction).copy()
reference = np.load(control['output']['path'], allow_pickle=False)
assert output.dtype == np.float32 and output.shape == (16, 1024)
assert np.isfinite(output).all() and (output >= 0).all() and output.tobytes() == reference.tobytes()
np.save(out / 'prediction.npy', output)
rows = []
for layer, fields in held:
    arrays = {k: np.asarray(v).copy() for k, v in fields.items()}
    assert all(np.isfinite(x).all() for x in arrays.values())
    assert arrays['Q'].shape == arrays['Y'].shape == arrays['Omega'].shape
    assert arrays['Q'].shape[1] == 384 and arrays['R'].shape == (384, 384)
    assert np.unique(arrays['previous']).size == arrays['previous'].size
    dest = out / f'layer{layer:02d}.npz'
    np.savez(dest, **arrays)
    rows.append(dict(layer=layer, trace=pin(dest), shapes={k:list(v.shape) for k,v in arrays.items()}))
held.clear()
write(out / 'result.json', dict(label=args.label, rank=38, source=control['source'], weights=pin(weights_file),
    prediction=pin(out/'prediction.npy'), rows=rows, all16_original_bytes_exact=True,
    flops=bill['flops_used'], copy_flops=copy_flops, original_flops=control['flops'],
    bill_less_copies_exact=True, wall_s=elapsed, numerical_cache_retired=True,
    max_self_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, threadpools=threadpool_info(),
    numpy_source=pin(np.linalg._linalg.__file__), targets_opened=False))
print(dict(label=args.label, all16_exact=True, original_flops=control['flops'], copy_flops=copy_flops,
           wall_s=elapsed, layers=len(rows)), flush=True)
