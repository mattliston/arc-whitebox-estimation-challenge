"""One preregistered score-risk oracle fit on the unchanged mini57 cell."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from threadpoolctl import threadpool_limits
from current_pair_oracle_math_v1 import exact_change
from gaussian_next_mean_risk_metric_v1 import geometry, risk
from k3_radial_numpy import gaussian_raw
from radial_binary_compact import select
from third_endpoint_span_v1 import input_basis, third_covariance_kernel
from third_endpoint_score_risk_v1 import ScoreRiskOperator

HERE = Path(__file__).resolve().parent
OUT = HERE / 'third_endpoint_score_risk_wide_v1.json'
assert not OUT.exists() and not OUT.with_suffix('.npz').exists()
cert = json.loads((HERE / 'third_endpoint_score_risk_small_v1.json').read_text())
helper_hash = hashlib.sha256((HERE / 'third_endpoint_score_risk_v1.py').read_bytes()).hexdigest()
assert cert['passed'] and cert['source_sha256'] == helper_hash
audit = json.loads((HERE / 'gaussian_next_mean_risk_metric_wide_v1.json').read_text())
assert audit['passed']
provenance = {}


def frozen(name, require_pass=True):
    path = HERE / name
    receipt = json.loads(path.with_suffix('.json').read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == receipt['output_sha256']
    if require_pass:
        assert receipt['passed']
    provenance[name] = digest
    return dict(np.load(path))


f = frozen('public_p2_third_cell_mini0057_v1.npz')
next_state = frozen('public_p2_third_carry_inputs_v1.npz')
old = frozen('public_p2_third_endpoint_span_v1.npz', require_pass=False)
W = np.load(HERE / 'public_p2_third_cell_mini0057_weights_v1.npy')
fr = json.loads((HERE / 'public_p2_third_cell_mini0057_v1.json').read_text())
assert hashlib.sha256(W.tobytes()).hexdigest() == fr['weights_array_sha256']
start = time.time()
with threadpool_limits(limits=3):
    delta = f['post13_head_delta'].astype(np.float64)
    mean = f['host'] + delta
    C = f['post_covariance'] - np.outer(f['host'], delta) - np.outer(delta, f['host']) - np.outer(delta, delta)
    _, alive = select(f['gate'], .01)
    cv = np.diag(C).copy(); C *= alive[:, None] * alive[None, :]; np.fill_diagonal(C, cv)
    w32, w64 = W[14].astype(np.float32), W[14].astype(np.float64)
    pre = (w32.T @ C.astype(np.float32) @ w32).astype(np.float64)
    mu = w64.T @ mean
    sn, an, pn, gn, mn, *_ = gaussian_raw(mu, np.diag(pre))
    checks = dict(Cpre=bool(np.array_equal(pre, next_state['Cpre14'])))
    for key, value in [('sigma', sn), ('alpha', an), ('pdf', pn), ('gate', gn), ('gaussian_mean', mn)]:
        checks[key] = bool(np.array_equal(value, next_state[key + '14']))
    assert all(checks.values()), checks
    u, beta, factor, v0 = geometry(mean, C, 2 / len(mean))
    s = f['sigma']; R = f['Cpre'] / np.outer(s, s)
    H, powers, basis = input_basis(f['state'], R)
    assert np.array_equal(H, old['H'])
    kernel = third_covariance_kernel(R, s, f['alpha'], f['pdf'], f['gate'], f['gaussian_mean'] / s)
    common = (R, s, f['alpha'], f['pdf'], f['gate'], f['gaussian_mean'] / s, f['host'] / s)
    op = ScoreRiskOperator(H, powers, kernel, u, beta, ridge=.001)
    target = old['target_covariance']
    control_ratio = risk(target - old['predicted_covariance'], u, beta, factor) / risk(target, u, beta, factor)
    assert control_ratio == audit['analytic_residual_ratio']
    print(json.dumps(dict(stage='qualified_fixed_inputs', byte_checks=checks, basis_rank=basis['numerical_rank'],
                          covariance_scale=op.covariance_scale, old_fit_risk_ratio=control_ratio)), flush=True)
    theta, solve = op.fit(target / op.covariance_scale, tolerance=1e-7, max_iterations=1000)
    plane = op.plane(theta)
    assert np.max(np.abs(np.diag(plane))) == 0
    pred_cov = op.forward(theta) * op.covariance_scale
    dm, direct = exact_change(common, dS=plane)
    kernel_error = float(np.max(np.abs(pred_cov - direct)))
    assert kernel_error <= 1e-12 and np.max(np.abs(dm)) == 0
    response = next_state['pdf14'] / (2 * next_state['sigma14'])
    target_value = np.sum(w64 * (target @ w64), axis=0) * response
    prediction_value = np.sum(w64 * (pred_cov @ w64), axis=0) * response
    assert np.array_equal(target_value, old['target_next_mean'])
    risk_ratio = risk(target - pred_cov, u, beta, factor) / risk(target, u, beta, factor)
    actual_ratio = float(np.sum((target_value - prediction_value) ** 2) / np.sum(target_value ** 2))
    weights2 = kernel * kernel + kernel.T * kernel.T
    weighted_ratio = float(np.sum(weights2 * (plane - old['target']) ** 2) / np.sum(weights2 * old['target'] ** 2))
    passed = solve['passed'] and risk_ratio <= .50 and actual_ratio <= .50
    result = dict(passed=bool(passed), ordinal=57, network='gregory-wilson', role='train', layer=13,
                  next_mean_layer=14, basis=basis, solve=solve, geometry_byte_parity=checks,
                  covariance_scale=op.covariance_scale, beta=beta, frozen_variance=v0,
                  analytic_risk_residual_ratio=risk_ratio, actual_W14_response_residual_ratio=actual_ratio,
                  ordinary_covariance_residual_ratio=float(np.sum((target - pred_cov) ** 2) / np.sum(target ** 2)),
                  old_weighted_plane_residual_ratio=weighted_ratio, old_fit_risk_residual_ratio=control_ratio,
                  marginal_mean_write_max=float(np.max(np.abs(dm))), covariance_kernel_max=kernel_error,
                  target_covariance_energy=float(np.sum(target ** 2)), target_next_mean_energy=float(np.sum(target_value ** 2)),
                  target_risk=risk(target, u, beta, factor), provenance=provenance, helper_sha256=helper_hash,
                  wall_s=time.time() - start, peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                  utc_finished=datetime.now(timezone.utc).isoformat(),
                  qualification='New certified Gaussian score-risk objective, same fixed endpoint basis and single training cell. Regularized in-sample oracle coefficients, not cross-net prediction or an unregularized ceiling. Actual W14 response is evaluation only. No new labels/download, carrier write, full-network raw/adjusted score or estimator change.')
    np.savez(OUT.with_suffix('.npz'), theta=theta, predicted_plane=plane, predicted_covariance=pred_cov,
             target_next_mean=target_value, predicted_next_mean=prediction_value)
    result['output_sha256'] = hashlib.sha256(OUT.with_suffix('.npz').read_bytes()).hexdigest()
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
