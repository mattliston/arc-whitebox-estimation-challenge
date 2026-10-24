"""Target-free full-bank f64 equivalence of the centered moment rewrite."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from threadpoolctl import threadpool_limits
import compact_fullpair_final_reference_v1 as parent
from centered_pair_numerics_v1 import nonlinear
from centered_relu_marginals_v1 import marginals
from radial_binary_compact import select
from radial_features import features
from radial_gnn_numpy import gnn_step
from compact_scalar_final_v1 import final_scalar_features
from compact_final_head_v1 import final_input, final_delta

HERE = Path(__file__).resolve().parent
OUT = HERE / 'centered_parent_numerics_v1.json'
assert not OUT.exists() and not OUT.with_suffix('.npz').exists()
cert = json.loads((HERE / 'centered_pair_numerics_small_v1.json').read_text())
assert cert['passed'] and cert['source_sha256'] == hashlib.sha256((HERE / 'centered_pair_numerics_v1.py').read_bytes()).hexdigest()
P = dict(np.load(HERE / 'payload_compact_feed_fullpair_v1.npz'))
Q = dict(np.load(HERE / 'payload_fullpair_final_head_v1.npz'))


def run(W, centered):
    state = np.zeros((1024, 64), np.float32)
    selections, saved = [], {}

    def selection(score, floor):
        ix, mask = select(score, floor)
        selections.append((ix.copy(), mask.copy()))
        return ix, mask

    def hook(layer, m, C, K21, aux, w):
        nonlocal state
        delta, state = gnn_step(layer, features(layer, m, C, K21, aux, w), aux['sigma'], w, state, P)
        return delta

    def final_hook(layer, m, aux, w):
        h = final_input(final_scalar_features(layer, m, aux, w), w, state, P)
        delta = final_delta(h, aux['sigma'], Q)
        saved.update(final_host=m.copy(), final_sigma=aux['sigma'].copy(), final_h=h.copy(), final_delta=delta.copy())
        return delta

    source = parent.source
    env = dict(parent.env, select=selection)
    if centered:
        before = '        beta = 2*(aux["gaussian_mean"]-host*g)'
        assert source.count(before) == 1
        source = source.replace(before, '        beta = aux["birth_beta"]')
        env.update(nonlinear=nonlinear, nonlinear_scalar=marginals)
    exec(source, env)
    prediction, _ = env['predict'](W, hook=hook, final_hook=final_hook, gate_floor=.01, cov32=True, metric='covariance')
    saved['prediction'] = np.maximum(prediction, 0)
    return saved, selections


records, outputs = [], {}
start = time.time()
with threadpool_limits(limits=3):
    for rank in (35, 100):
        W = np.load(HERE / 'rows_current_transport_shadow_v2' / f'row_{rank:04d}' / 'weights.npy')
        baseline, old_masks = run(W, False)
        print(json.dumps(dict(rank=rank, stage='baseline_complete', wall_s=time.time()-start)), flush=True)
        changed, new_masks = run(W, True)
        assert len(old_masks) == len(new_masks) == 30
        members = [bool(np.array_equal(a[1], b[1])) for a, b in zip(old_masks, new_masks)]
        orders = [bool(np.array_equal(a[0], b[0])) for a, b in zip(old_masks, new_masks)]
        errors = {key: float(np.max(np.abs(changed[key] - baseline[key]))) for key in baseline}
        passed = errors['prediction'] <= 2e-6 and all(members)
        row = dict(rank=rank, passed=bool(passed), maximum_absolute_errors=errors,
                   selection_membership_byte_equal=members, selection_order_byte_equal=orders,
                   final_prediction_rms=float(np.sqrt(np.mean((changed['prediction'][-1]-baseline['prediction'][-1])**2))))
        records.append(row)
        outputs[f'baseline_{rank}'] = baseline['prediction']
        outputs[f'centered_{rank}'] = changed['prediction']
        print(json.dumps(row), flush=True)
result = dict(passed=all(r['passed'] for r in records), records=records, target_access=False,
              pair_helper_sha256=cert['source_sha256'], marginal_helper_sha256=cert['marginal_helper_sha256'],
              wall_s=time.time()-start, peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
              utc_finished=datetime.now(timezone.utc).isoformat(),
              qualification='Both full1024x16 dynamic NumPy f64-moment/f32-carrier bank paths, same frozen P and scalar-final Q. Centered pair/scalar/birth arithmetic only. Actual membership checked; no labels, MPS/f32-moment/gradient, raw/adjusted or candidate claim.')
np.savez(OUT.with_suffix('.npz'), **outputs)
result['output_sha256'] = hashlib.sha256(OUT.with_suffix('.npz').read_bytes()).hexdigest()
OUT.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k:v for k,v in result.items() if k != 'records'}, indent=2), flush=True)
