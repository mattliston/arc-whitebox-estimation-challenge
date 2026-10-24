"""One shared bounded14-coefficient sensitivity gate, then at most two runs."""
from pathlib import Path
import json
import time
import numpy as np
from scipy.optimize import lsq_linear
from threadpoolctl import threadpool_limits
from distinct_birth_amplitude_v1 import predict
from radial_features import features
from radial_gnn_numpy import gnn_step

here = Path(__file__).resolve().parent
out = here/'distinct_birth_sensitivity_verdict_v1.json'
assert not out.exists()
P = dict(np.load(here/'payload_compact_feed_fullpair_v1.npz'))
data = {}
for rank in (35, 100):
    rec = json.loads((here/f'distinct_birth_sensitivity_r{rank}_v1.json').read_text())
    assert rec['baseline_byte_exact'] and rec['target_access'] is False
    z = dict(np.load(here/f'distinct_birth_sensitivity_r{rank}_v1.npz'))
    truth = np.load(here.parent/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{rank:04d}.npz')['mc_post'][-1]
    raw_baseline = z['baseline'][-1]
    baseline = np.maximum(raw_baseline, 0)
    jac = z['jacobian']*(raw_baseline > 0)[:, None]
    data[rank] = dict(baseline=baseline, raw_baseline=raw_baseline, jac=jac, truth=truth,
                      parent_mse=float(np.mean((baseline-truth)**2)))
norm = np.sqrt(np.mean([d['parent_mse'] for d in data.values()]))
A = np.concatenate([d['jac'] for d in data.values()], axis=0)/norm
b = np.concatenate([d['truth']-d['baseline'] for d in data.values()])/norm
started = time.time()
with threadpool_limits(limits=3):
    solution = lsq_linear(A, b, bounds=(-.10, .10), tol=1e-12, lsmr_tol=1e-12, max_iter=200)
assert solution.success, solution.message
offsets = solution.x
amplitudes = np.ones(16)
amplitudes[1:15] += offsets
rows = []
for rank, d in data.items():
    estimate = np.maximum(d['raw_baseline']+d['jac']@offsets, 0)
    mse = float(np.mean((estimate-d['truth'])**2))
    rows.append(dict(rank=rank, parent_raw_final_mse=d['parent_mse'], predicted_raw_final_mse=mse,
                     predicted_ratio=mse/d['parent_mse'],
                     jacobian_column_rms=np.sqrt(np.mean(d['jac']**2, axis=0)).tolist()))
linear_pass = all(r['predicted_ratio'] <= .8 for r in rows)
print(json.dumps(dict(shared_amplitudes=amplitudes.tolist(), linear_gate_pass=linear_pass,
                      rows=[{k: v for k, v in r.items() if k != 'jacobian_column_rms'} for r in rows])), flush=True)
actual_rows = []
if linear_pass:
    with threadpool_limits(limits=3):
        for rank in (35, 100):
            d = data[rank]
            folder = here/'rows_current_transport_shadow_v2'/f'row_{rank:04d}'
            W = np.load(folder/'weights.npy')
            state = np.zeros((1024, 64), np.float32)
            def hook(l, m, C, K21, aux, w):
                global state
                x = features(l, m, C, K21, aux, w)
                delta, state = gnn_step(l, x, aux['sigma'], w, state, P)
                return delta
            mean, _ = predict(W, hook=hook, birth_amplitudes=amplitudes, gate_floor=.01,
                              cov32=True, metric='covariance', collect_states=False)
            actual = np.maximum(mean[-1], 0)
            change = actual-d['baseline']
            expected = d['jac']@offsets
            response_error = float(np.linalg.norm(change-expected)/max(np.linalg.norm(expected), 1e-30))
            value = float(np.mean((actual-d['truth'])**2))
            pred = here/f'distinct_birth_shared_actual_r{rank}_v1.npz'
            assert not pred.exists()
            np.savez(pred, mean=mean, amplitudes=amplitudes)
            row = dict(rank=rank, raw_final_mse=value, parent_raw_final_mse=d['parent_mse'],
                       ratio=value/d['parent_mse'], response_prediction_relative_error=response_error)
            actual_rows.append(row)
            print(json.dumps(row), flush=True)
actual_pass = linear_pass and all(r['ratio'] <= .85 and r['response_prediction_relative_error'] <= .1 for r in actual_rows)
singular = np.linalg.svd(A, compute_uv=False)
receipt = dict(shared_amplitudes=amplitudes.tolist(), optimizer_status=solution.message,
               linear_gate_pass=linear_pass, predicted_rows=rows, actual_rows=actual_rows,
               actual_gate_pass=actual_pass, normalized_jacobian_singular_values=singular.tolist(),
               wall_s=time.time()-started,
               qualification='Two public development-net sensitivity fit with one shared vector bounded to+-10%. Not held-out, adjusted, or official performance; only passing both gates justifies a24-net fit.')
out.write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(dict(actual_gate_pass=actual_pass)), flush=True)
