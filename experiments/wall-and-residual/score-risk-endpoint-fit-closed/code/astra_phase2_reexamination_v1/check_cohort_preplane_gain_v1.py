"""One fixed regularized oracle fit with causal cohort-gain features."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from threadpoolctl import threadpool_limits
from cohort_preplane_gain_v1 import CohortGainOperator, cohort_planes, plane_action
from current_pair_oracle_math_v1 import exact_change
from gaussian_next_mean_risk_metric_v1 import geometry, risk
from k3_radial_numpy import gaussian_raw
from radial_binary_compact import select
from radial_cubic import plane as parent_plane
from third_endpoint_span_v1 import third_covariance_kernel

HERE = Path(__file__).resolve().parent
OUT = HERE / 'cohort_preplane_gain_wide_v1.json'
assert not OUT.exists() and not OUT.with_suffix('.npz').exists()
cert = json.loads((HERE / 'cohort_preplane_gain_small_v1.json').read_text())
helper_hash = hashlib.sha256((HERE / 'cohort_preplane_gain_v1.py').read_bytes()).hexdigest()
assert cert['passed'] and helper_hash == cert['source_sha256']
data, provenance = {}, {}
for name, filename, require_pass in [
    ('cohort', 'cohort_preplane_gain_inputs_mini0057_v1.npz', True),
    ('base', 'public_p2_third_cell_mini0057_v1.npz', True),
    ('target', 'public_p2_third_endpoint_span_v1.npz', False),
    ('next', 'public_p2_third_carry_inputs_v1.npz', True),
]:
    path = HERE / filename
    receipt = json.loads(path.with_suffix('.json').read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == receipt['output_sha256']
    if require_pass:
        assert receipt['passed']
    provenance[name] = digest; data[name] = dict(np.load(path))
f, incoming, old, nxt = data['base'], data['cohort'], data['target'], data['next']
W = np.load(HERE / 'public_p2_third_cell_mini0057_weights_v1.npy')
fr = json.loads((HERE / 'public_p2_third_cell_mini0057_v1.json').read_text())
assert hashlib.sha256(W.tobytes()).hexdigest() == fr['weights_array_sha256']
start = time.time()
with threadpool_limits(limits=3):
    s = f['sigma']; R = f['Cpre'] / np.outer(s, s)
    U, singular, _ = np.linalg.svd(incoming['features'], full_matrices=False)
    rank = int(np.count_nonzero(singular > singular[0] * 1e-12))
    H = U[:, :rank] * np.sqrt(len(U))
    reconstruction = float(np.linalg.norm(H @ (H.T @ incoming['features']) / len(H) - incoming['features']) /
                           np.linalg.norm(incoming['features']))
    assert reconstruction <= 1e-10
    p, lam, counts, active = (incoming[k] for k in ('pushed_factors', 'signed_weights', 'cohort_counts', 'live_indices'))
    raw_planes = cohort_planes(p, lam, counts, active)
    base_cp = np.zeros((1024, 1024))
    base_cp[np.ix_(active, active)] = parent_plane(p[active], lam)
    np.fill_diagonal(base_cp, 0)
    physical_target = old['target'] * (s[:, None] ** 2 * s[None, :])
    decomposition_ratio = float(np.linalg.norm(raw_planes.sum(axis=0) - base_cp) / np.linalg.norm(physical_target))
    assert decomposition_ratio <= .01, decomposition_ratio
    planes = raw_planes / (s[None, :, None] ** 2 * s[None, None, :])
    del raw_planes
    kernel = third_covariance_kernel(R, s, f['alpha'], f['pdf'], f['gate'], f['gaussian_mean'] / s)
    common = (R, s, f['alpha'], f['pdf'], f['gate'], f['gaussian_mean'] / s, f['host'] / s)
    delta = f['post13_head_delta'].astype(np.float64)
    mean = f['host'] + delta
    C = f['post_covariance'] - np.outer(f['host'], delta) - np.outer(delta, f['host']) - np.outer(delta, delta)
    _, alive = select(f['gate'], .01)
    cv = np.diag(C).copy(); C *= alive[:, None] * alive[None, :]; np.fill_diagonal(C, cv)
    w32, w64 = W[14].astype(np.float32), W[14].astype(np.float64)
    pre = (w32.T @ C.astype(np.float32) @ w32).astype(np.float64)
    sn, an, pn, gn, mn, *_ = gaussian_raw(w64.T @ mean, np.diag(pre))
    checks = dict(Cpre=bool(np.array_equal(pre, nxt['Cpre14'])))
    for key, value in [('sigma', sn), ('alpha', an), ('pdf', pn), ('gate', gn), ('gaussian_mean', mn)]:
        checks[key] = bool(np.array_equal(value, nxt[key + '14']))
    assert all(checks.values()), checks
    u, beta, factor, v0 = geometry(mean, C, 2 / len(mean))
    op = CohortGainOperator(H, planes, kernel, u, beta, ridge=.001)
    target_cov = old['target_covariance']
    target_value = old['target_next_mean']
    target_risk = risk(target_cov, u, beta, factor)
    print(json.dumps(dict(stage='qualified_input', feature_rank=rank, cohorts=len(counts),
                          decomposition_roundoff_over_signal=decomposition_ratio,
                          covariance_scale=op.covariance_scale, geometry_byte_checks=checks)), flush=True)
    theta, solve = op.fit(target_cov / op.covariance_scale, tolerance=1e-7, max_iterations=1000)
    gains = op.gains(theta)
    response = nxt['pdf14'] / (2 * nxt['sigma14'])
    records, outputs = {}, {}
    for name, finite in [('linear', False), ('finite', True)]:
        dS = plane_action(planes, gains, finite=finite)
        assert np.max(np.abs(np.diag(dS))) == 0
        dm, dc = exact_change(common, dS=dS)
        assert np.max(np.abs(dm)) == 0
        if not finite:
            parity = float(np.max(np.abs(dc - op.covariance_scale * op.forward(theta))))
            assert parity <= 1e-12
        prediction_value = np.sum(w64 * (dc @ w64), axis=0) * response
        risk_ratio = risk(target_cov - dc, u, beta, factor) / target_risk
        value_ratio = float(np.sum((target_value - prediction_value) ** 2) / np.sum(target_value ** 2))
        records[name] = dict(analytic_risk_residual_ratio=risk_ratio, actual_W14_response_residual_ratio=value_ratio,
                             ordinary_covariance_residual_ratio=float(np.sum((target_cov - dc) ** 2) / np.sum(target_cov ** 2)),
                             passed=bool(risk_ratio <= .50 and value_ratio <= .50))
        outputs[name + '_covariance'] = dc
        outputs[name + '_next_mean'] = prediction_value
    gain_max = float(np.max(np.abs(gains)))
    passed = solve['passed'] and gain_max <= .05 and all(r['passed'] for r in records.values())
    result = dict(passed=bool(passed), network='gregory-wilson', ordinal=57, role='train', pre_layer=13,
                  next_mean_layer=14, feature_count=104, numerical_feature_rank=rank,
                  feature_span_reconstruction_relative=reconstruction, singular_values=singular.tolist(),
                  cohorts=len(counts), coefficient_count=int(np.prod(op.shape)), solve=solve,
                  gain_absolute_max=gain_max, gain_rms=float(np.sqrt(np.mean(gains * gains))),
                  gain_safety_gate_passed=gain_max <= .05, responses=records,
                  decomposition_roundoff_over_true_residual=decomposition_ratio, consumer_kernel_max=parity,
                  covariance_scale=op.covariance_scale, geometry_byte_checks=checks,
                  target_covariance_energy=float(np.sum(target_cov * target_cov)),
                  target_next_mean_energy=float(np.sum(target_value * target_value)), target_risk=target_risk,
                  provenance=provenance, helper_sha256=helper_hash, wall_s=time.time() - start,
                  peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                  utc_finished=datetime.now(timezone.utc).isoformat(),
                  qualification='One fixed regularized oracle of per-neuron birth-cohort gains with pre-plane features. Actual and finite isolated pair responses; no cross-net predictor, whole-state rollout, score, source download, sample or estimator write. Marginal third cumulants held fixed. Feature SVD is local input-only preconditioning, not a transferable coefficient convention.')
    np.savez(OUT.with_suffix('.npz'), theta=theta, gains=gains, H=H, **outputs)
    result['output_sha256'] = hashlib.sha256(OUT.with_suffix('.npz').read_bytes()).hexdigest()
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
