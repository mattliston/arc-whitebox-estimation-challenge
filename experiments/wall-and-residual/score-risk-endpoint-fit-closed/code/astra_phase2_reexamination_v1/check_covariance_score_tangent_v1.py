"""One fixed-frame tangent oracle on the already frozen total-error target."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from threadpoolctl import threadpool_limits
from covariance_score_subspace_v1 import tilt
from covariance_score_tangent_v1 import project_tangent
from gaussian_next_mean_risk_metric_v1 import geometry, risk

HERE = Path(__file__).resolve().parent
OUT = HERE / 'covariance_score_tangent_wide_v1.json'
assert not OUT.exists() and not OUT.with_suffix('.npz').exists()
cert = json.loads((HERE / 'covariance_score_tangent_small_v1.json').read_text())
helper_hash = hashlib.sha256((HERE / 'covariance_score_tangent_v1.py').read_bytes()).hexdigest()
assert cert['passed'] and helper_hash == cert['source_sha256']
path = HERE / 'covariance_score_rank32_oracle_v1.npz'
prior = json.loads(path.with_suffix('.json').read_text())
assert hashlib.sha256(path.read_bytes()).hexdigest() == prior['output_sha256']
f = dict(np.load(path))
assert prior['qualification_checks']['passed'] and all(prior['geometry_byte_checks'].values())
assert hashlib.sha256(f['input_basis'].tobytes()).hexdigest() == prior['input_basis_sha256']
next_path = HERE / 'public_p2_third_carry_inputs_v1.npz'
next_receipt = json.loads(next_path.with_suffix('.json').read_text())
assert hashlib.sha256(next_path.read_bytes()).hexdigest() == next_receipt['output_sha256']
nxt = dict(np.load(next_path))
W = np.load(HERE / 'public_p2_third_cell_mini0057_weights_v1.npy')
fr = json.loads((HERE / 'public_p2_third_cell_mini0057_v1.json').read_text())
assert hashlib.sha256(W.tobytes()).hexdigest() == fr['weights_array_sha256']
start = time.time()
with threadpool_limits(limits=3):
    E, C, mean, Q = (f[key] for key in ('target_covariance_error', 'bank_covariance', 'bank_mean', 'input_basis'))
    assert E.shape == (1024, 1024) and Q.shape == (1024, 32)
    u, beta, factor, v0 = geometry(mean, C, 2 / len(mean))
    Z = tilt(E, u, np.sqrt(1 - beta) - 1)
    pred_Z, V, shift = project_tangent(Z, Q)
    pred_C = tilt(pred_Z, u, 1 / np.sqrt(1 - beta) - 1)
    target_risk = risk(E, u, beta, factor)
    assert np.isclose(target_risk, prior['target_risk'], rtol=1e-12, atol=0)
    risk_ratio = risk(E - pred_C, u, beta, factor) / target_risk
    w = W[14].astype(np.float64)
    response = nxt['pdf14'] / (2 * nxt['sigma14'])
    target_value = np.sum(w * (E @ w), axis=0) * response
    assert np.array_equal(target_value, f['target_next_mean'])
    pred_value = np.sum(w * (pred_C @ w), axis=0) * response
    value_ratio = float(np.sum((target_value - pred_value) ** 2) / np.sum(target_value ** 2))
    residual = pred_Z - Z
    normal = (residual + .5 * np.trace(residual) * np.eye(len(mean))) @ Q
    normal_relative = float(np.linalg.norm(normal) / np.linalg.norm(Z @ Q))
    assert normal_relative <= 1e-10
    passed = risk_ratio <= .25 and value_ratio <= .25
    result = dict(passed=bool(passed), ordinal=57, network='gregory-wilson', role='train', post_layer=13,
                  next_mean_layer=14, fixed_input_directions=32, maximum_correction_rank=64,
                  analytic_risk_residual_ratio=risk_ratio, actual_W14_response_residual_ratio=value_ratio,
                  ordinary_covariance_residual_ratio=float(np.sum((E - pred_C) ** 2) / np.sum(E ** 2)),
                  subspace_stationarity_relative=normal_relative, trace_correction=shift,
                  old_core_only=prior['bases']['bank_geometry'], oracle_core_only=prior['bases']['oracle_spectral'],
                  target_covariance_rms=prior['target_covariance_rms'], target_next_mean_rms=prior['target_next_mean_rms'],
                  input_basis_sha256=prior['input_basis_sha256'], frozen_target_sha256=prior['output_sha256'],
                  helper_sha256=helper_hash, wall_s=time.time() - start,
                  peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                  utc_finished=datetime.now(timezone.utc).isoformat(),
                  qualification='Fixed same32 target-free bank directions with unrestricted oracle cross-block coefficients. Previously frozen100M training-cell target, no new target decoding/sample. Not a predictor, tensor-carrier retry, total network score, noise-corrected ceiling or state write.')
    np.savez(OUT.with_suffix('.npz'), predicted_covariance=pred_C, predicted_next_mean=pred_value,
             tilted_cross_factor=V, tilted_input_basis=Q)
    result['output_sha256'] = hashlib.sha256(OUT.with_suffix('.npz').read_bytes()).hexdigest()
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
