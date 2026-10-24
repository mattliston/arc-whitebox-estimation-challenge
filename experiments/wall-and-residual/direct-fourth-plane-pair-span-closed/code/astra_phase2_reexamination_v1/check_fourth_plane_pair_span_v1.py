"""One-cell independent-replica coefficient-span oracle, not a learner bank."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from threadpoolctl import threadpool_limits
from current_pair_oracle_math_v1 import exact_change
from fourth_plane_pair_span_v1 import input_basis, PlaneOperator

HERE = Path(__file__).resolve().parent
OUT = HERE/'fourth_plane_pair_span_verdict_v1.json'
assert not OUT.exists() and not OUT.with_suffix('.npz').exists()
certificate = json.loads((HERE/'fourth_plane_pair_span_small_v1.json').read_text())
assert certificate['passed']
source_hash = hashlib.sha256((HERE/'fourth_plane_pair_span_v1.py').read_bytes()).hexdigest()
assert source_hash == certificate['source_sha256']
path = HERE/'fourth_trace_oracle_inputs_r35_v1.npz'
input_hash = hashlib.sha256(path.read_bytes()).hexdigest()
assert input_hash == json.loads(path.with_suffix('.json').read_text())['output_sha256']
f = dict(np.load(path))
H, powers, basis_info = input_basis(f)
# The input-only feature basis is now fixed before any oracle target access.
truth, provenance = {}, {}
for stream in ('A', 'B'):
    path = HERE/f'fourth_trace_oracle_r35_{stream}_v1.npz'
    meta = json.loads(path.with_suffix('.json').read_text())
    assert meta['passed'] and meta['frozen_inputs_sha256'] == input_hash
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == meta['output_sha256']
    truth[stream] = dict(np.load(path))
    provenance[stream] = digest
previous = dict(np.load(HERE/'fourth_trace_oracle_value_v1.npz'))
s = f['pre13_sigma']
common = (f['pre13_Cpre']/np.outer(s, s), s, f['pre13_alpha'], f['pre13_pdf'],
          f['pre13_gate'], f['pre13_gaussian_mean']/s, f['pre13_m']/s)
w14 = f['W14'].astype(np.float64)
start = time.time()
saved, solves, target_parity = {}, [], {}


def consume(q22, q31, k4):
    dm, dc = exact_change(common, dQ22=q22, dQ31=q31, dq4=k4)
    value = np.sum(w14*(dc@w14), axis=0)*f['response14']
    return dm, dc, value


with threadpool_limits(limits=3):
    operators = {key: PlaneOperator(H, powers, key == 'Q22') for key in ('Q22', 'Q31')}
    for stream in ('A', 'B'):
        t = truth[stream]
        targets = dict(Q22=(t['Q22']-f['pre13_k22'])/np.outer(s*s, s*s),
                       Q31=(t['Q31']-f['pre13_k31'])/(s[:, None]**3*s[None, :]),
                       k4=(t['k4']-f['pre13_k4'])/s**4)
        for key in ('Q22', 'Q31'):
            np.fill_diagonal(targets[key], 0)
        pred = {}
        for key, op in operators.items():
            coefficients, result = op.fit(targets[key])
            solves.append(dict(stream=stream, head=key, **result))
            print(json.dumps(dict(stream=stream, head=key, iterations=result['iterations'],
                                  residual=result['relative_normal_residual'], passed=result['passed'], wall_s=time.time()-start)), flush=True)
            if not result['passed']:
                failure = dict(passed=False, numerical_qualified=False, solves=solves, basis=basis_info,
                               qualification='Fixed solver gate failed before a value verdict; no iteration/regularizer sweep.')
                OUT.write_text(json.dumps(failure, indent=2)+'\n')
                raise SystemExit(1)
            saved[f'coefficients_{stream}_{key}'] = coefficients
            pred[key] = op.forward(coefficients)
        beta = (H.T@targets['k4'])/len(H)/1.001
        pred['k4'] = H@beta
        saved[f'coefficients_{stream}_k4'] = beta
        for arm, data in (('prediction', pred), ('target', targets)):
            response = consume(data['Q22'], data['Q31'], data['k4'])
            for kind, value in zip(('mean', 'covariance', 'next_mean'), response):
                key = f'{arm}_{stream}_{kind}'
                saved[key] = value
                if arm == 'target':
                    target_parity[key] = bool(np.array_equal(value, previous[key]))
            for key, value in data.items():
                saved[f'{arm}_{stream}_plane_{key}'] = value
    assert all(target_parity.values()), target_parity
    consumers = {}
    for consumer in ('covariance', 'next_mean', 'plane_Q22', 'plane_Q31', 'plane_k4'):
        ta, tb = (saved[f'target_{stream}_{consumer}'] for stream in ('A', 'B'))
        pa, pb = (saved[f'prediction_{stream}_{consumer}'] for stream in ('A', 'B'))
        signal = float(np.sum(ta*tb))
        residual = float(np.sum((ta-pa)*(tb-pb)))
        noise = float(np.sum((ta-tb)**2)/2)
        ordinary_ab = float(np.sum((pa-tb)**2)/np.sum(tb*tb))
        ordinary_ba = float(np.sum((pb-ta)**2)/np.sum(ta*ta))
        ratio = residual/signal if signal > 0 else None
        consumers[consumer] = dict(baseline_cross_energy=signal, residual_cross_energy=residual,
              signal_residual_ratio=ratio, target_noise_energy=noise, target_signal_to_noise=signal/noise if noise > 0 else None,
              ordinary_cross_predictionA_targetB=ordinary_ab, ordinary_cross_predictionB_targetA=ordinary_ba,
              passed=signal > 0 and ratio <= .20 and max(ordinary_ab, ordinary_ba) <= .30)
    result = dict(passed=all(consumers[key]['passed'] for key in ('covariance', 'next_mean')),
                  numerical_qualified=True, basis=basis_info, solves=solves, consumers=consumers,
                  target_byte_parity=target_parity, rank=35, layer=13, samples_per_stream=8000000,
                  input_sha256=input_hash, truth_sha256=provenance, source_sha256=source_hash,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  wall_s=time.time()-start, peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                  utc_finished=datetime.now(timezone.utc).isoformat(),
                  qualification='One-cell direct fourth-plane coefficient-span oracle. A/B coefficients fitted separately to independent reused8M streams. Not cross-network generalization, complete covariance-error reduction, final MSE, adjusted score, or a deployable model.')
    np.savez(OUT.with_suffix('.npz'), H=H, powers=powers, **saved)
    OUT.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)
