"""Fixed fresh-weight risk audit; no fitting, new labels, or estimator writes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
from threadpoolctl import threadpool_limits

from gaussian_next_mean_risk_metric_v1 import geometry, risk, variable_response
from k3_radial_numpy import gaussian_raw
from radial_binary_compact import select


HERE = Path(__file__).resolve().parent
OUT = HERE / 'gaussian_next_mean_risk_metric_wide_v1.json'
assert not OUT.exists() and not OUT.with_suffix('.npz').exists()
cert = json.loads((HERE / 'gaussian_next_mean_risk_metric_small_v1.json').read_text())
helper_hash = hashlib.sha256((HERE / 'gaussian_next_mean_risk_metric_v1.py').read_bytes()).hexdigest()
assert cert['passed'] and cert['source_sha256'] == helper_hash
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
span = frozen('public_p2_third_endpoint_span_v1.npz', require_pass=False)
W = np.load(HERE / 'public_p2_third_cell_mini0057_weights_v1.npy')
receipt = json.loads((HERE / 'public_p2_third_cell_mini0057_v1.json').read_text())
assert hashlib.sha256(W.tobytes()).hexdigest() == receipt['weights_array_sha256']
start = time.time()

with threadpool_limits(limits=3):
    n = len(f['host'])
    assert n == 1024 and W.shape == (16, n, n)
    delta = f['post13_head_delta'].astype(np.float64)
    m = f['host'] + delta
    C = f['post_covariance'] - np.outer(f['host'], delta) - np.outer(delta, f['host']) - np.outer(delta, delta)
    _, alive = select(f['gate'], .01)
    cv = np.diag(C).copy()
    C = C * (alive[:, None] * alive[None, :])
    np.fill_diagonal(C, cv)
    w32 = W[14].astype(np.float32)
    w64 = W[14].astype(np.float64)
    pre = (w32.T @ C.astype(np.float32) @ w32).astype(np.float64)
    mu = w64.T @ m
    sigma, alpha, pdf, gate, gm, *_ = gaussian_raw(mu, np.diag(pre))
    checks = {'next_covariance': bool(np.array_equal(pre, next_state['Cpre14']))}
    for key, value in [('sigma', sigma), ('alpha', alpha), ('pdf', pdf), ('gate', gate), ('gaussian_mean', gm)]:
        checks[key] = bool(np.array_equal(value, next_state[key + '14']))
    # The frozen boundary stores alpha, not raw mu. Require byte-exact alpha,
    # sigma, and Gaussian mean; alpha*sigma is only a round-trip diagnostic.
    mu_roundtrip = float(np.max(np.abs(mu - next_state['alpha14'] * next_state['sigma14'])))
    assert all(checks.values()), checks
    tau2 = 2 / n
    u, beta, factor, v0 = geometry(m, C, tau2)
    names = ['target', 'prediction', 'residual']
    target = span['target_covariance']
    prediction = span['predicted_covariance']
    fields = [target, prediction, target - prediction]
    analytic = np.array([risk(E, u, beta, factor) for E in fields])
    assert np.all(analytic > 0)
    print(json.dumps(dict(stage='geometry_parity_passed', checks=checks, beta=beta,
                          frozen_variance=v0, analytic_energies=analytic.tolist())), flush=True)
    batch_energies = []
    streams = []
    variance_extrema = [float('inf'), float('-inf')]
    for seed in (930084, 930085):
        rng = np.random.Generator(np.random.PCG64DXSM(seed))
        local = []
        for batch in range(8):
            directions = rng.standard_normal((n, 512)) * np.sqrt(tau2)
            mu_random = m @ directions
            r0 = np.exp(-mu_random * mu_random / (2 * v0)) / np.sqrt(2 * np.pi) / (2 * np.sqrt(v0))
            variance = np.sum(directions * (C @ directions), axis=0)
            variance_extrema[0] = min(variance_extrema[0], float(variance.min()))
            variance_extrema[1] = max(variance_extrema[1], float(variance.max()))
            r1 = variable_response(m, C, directions)
            q_target = np.sum(directions * (target @ directions), axis=0)
            q_prediction = np.sum(directions * (prediction @ directions), axis=0)
            q = np.stack([q_target, q_prediction, q_target - q_prediction])
            energy = np.stack([np.mean((q * r0) ** 2, axis=1), np.mean((q * r1) ** 2, axis=1)])
            local.append(energy)
        local = np.stack(local)
        batch_energies.extend(local)
        streams.append(dict(seed=seed, directions=4096, mean_energies=local.mean(axis=0).tolist(),
                            batch_standard_errors=(local.std(axis=0, ddof=1) / np.sqrt(8)).tolist()))
        print(json.dumps(dict(stage='stream_complete', **streams[-1])), flush=True)
    batches = np.stack(batch_energies)
    mc = batches.mean(axis=0)
    se = batches.std(axis=0, ddof=1) / np.sqrt(len(batches))
    records = {}
    for j, name in enumerate(names):
        z = float(abs(mc[0, j] - analytic[j]) / se[0, j])
        variable_ratio = float(mc[1, j] / analytic[j])
        records[name] = dict(analytic_energy=float(analytic[j]), frozen_mc_energy=float(mc[0, j]),
                             frozen_mc_batch_se=float(se[0, j]), analytic_mc_difference_in_se=z,
                             variable_mc_energy=float(mc[1, j]), variable_mc_batch_se=float(se[1, j]),
                             variable_mc_over_analytic=variable_ratio,
                             frozen_gate_passed=z <= 4, variable_gate_passed=abs(variable_ratio - 1) <= .25)
    analytic_ratio = float(analytic[2] / analytic[0])
    variable_ratio = float(mc[1, 2] / mc[1, 0])
    # Delta-method uncertainty retains within-batch target/residual covariance.
    ratio_influence = (batches[:, 1, 2] - variable_ratio * batches[:, 1, 0]) / mc[1, 0]
    variable_ratio_se = float(ratio_influence.std(ddof=1) / np.sqrt(len(batches)))
    ratio_relative_error = abs(analytic_ratio / variable_ratio - 1)
    actual_response = next_state['pdf14'] / (2 * next_state['sigma14'])
    actual_target = np.sum(w64 * (target @ w64), axis=0) * actual_response
    actual_prediction = np.sum(w64 * (prediction @ w64), axis=0) * actual_response
    assert np.array_equal(actual_target, span['target_next_mean'])
    assert np.array_equal(actual_prediction, span['predicted_next_mean'])
    passed = all(r['frozen_gate_passed'] and r['variable_gate_passed'] for r in records.values()) and ratio_relative_error <= .15
    result = dict(passed=bool(passed), network='gregory-wilson', ordinal=57, role='train', post_layer=13,
                  next_mean_layer=14, geometry_byte_parity=checks, mu_alpha_sigma_roundtrip_max=mu_roundtrip,
                  active_neurons=int(alive.sum()), mean_norm_squared=float(m @ m), covariance_trace=float(np.trace(C)),
                  beta=beta, frozen_variance=v0, fresh_variance_range=variance_extrema,
                  fields=records, analytic_residual_ratio=analytic_ratio, variable_mc_residual_ratio=variable_ratio,
                  variable_mc_residual_ratio_batch_se=variable_ratio_se,
                  residual_ratio_relative_disagreement=ratio_relative_error,
                  unweighted_covariance_residual_ratio=float(np.sum(fields[2] ** 2) / np.sum(target ** 2)),
                  actual_W14_residual_ratio=float(np.sum((actual_target - actual_prediction) ** 2) / np.sum(actual_target ** 2)),
                  streams=streams, provenance=provenance, helper_sha256=helper_hash,
                  wall_s=time.time() - start, peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                  utc_finished=datetime.now(timezone.utc).isoformat(),
                  qualification='Fixed covariance-error fields from one earlier in-sample third-plane fit; target-free bank geometry. Fresh Gaussian weight directions only. No fitted coefficients changed, no new targets read/downloaded, no carrier write, no full-network raw/adjusted score. Errors are the same unmasked isolated covariance responses as the previous span audit, not total bank errors.')
    np.savez(OUT.with_suffix('.npz'), batch_energies=batches, analytic_energies=analytic)
    result['output_sha256'] = hashlib.sha256(OUT.with_suffix('.npz').read_bytes()).hexdigest()
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
