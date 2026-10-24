"""Fixed two-basis rank32 audit of total current post13 covariance error."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from threadpoolctl import threadpool_limits
from covariance_score_subspace_v1 import project, tilt
from gaussian_next_mean_risk_metric_v1 import geometry, risk
from k3_radial_numpy import gaussian_raw
from radial_binary_compact import select

HERE = Path(__file__).resolve().parent
OUT = HERE / 'covariance_score_rank32_oracle_v1.json'
assert not OUT.exists() and not OUT.with_suffix('.npz').exists()
cert = json.loads((HERE / 'covariance_score_subspace_small_v1.json').read_text())
helper_hash = hashlib.sha256((HERE / 'covariance_score_subspace_v1.py').read_bytes()).hexdigest()
assert cert['passed'] and helper_hash == cert['source_sha256']
provenance = {}
inputs = []
for name in ['public_p2_third_cell_mini0057_v1.npz', 'public_p2_third_carry_inputs_v1.npz']:
    path = HERE / name
    receipt = json.loads(path.with_suffix('.json').read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert receipt['passed'] and digest == receipt['output_sha256']
    inputs.append(dict(np.load(path))); provenance[name] = digest
f, nxt = inputs
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
    sn, an, pn, gn, mn, *_ = gaussian_raw(w64.T @ mean, np.diag(pre))
    checks = dict(Cpre=bool(np.array_equal(pre, nxt['Cpre14'])))
    for key, value in [('sigma', sn), ('alpha', an), ('pdf', pn), ('gate', gn), ('gaussian_mean', mn)]:
        checks[key] = bool(np.array_equal(value, nxt[key + '14']))
    assert all(checks.values()), checks
    n, k = len(mean), 32
    assert n == 1024
    u, beta, factor, v0 = geometry(mean, C, 2 / n)
    half = np.sqrt(1 - beta) - 1
    inverse_half = 1 / np.sqrt(1 - beta) - 1
    bank_eigenvalues, bank_eigenvectors = np.linalg.eigh(tilt(C, u, half))
    Q_input = bank_eigenvectors[:, -k:]
    input_basis_hash = hashlib.sha256(Q_input.tobytes()).hexdigest()
    print(json.dumps(dict(stage='input_only_basis_frozen', input_basis_sha256=input_basis_hash,
                          geometry_byte_checks=checks)), flush=True)
    path = HERE / 'public_p2_higher_moments_pilot_v1/mlp_00057.npz'
    download = json.loads(path.with_suffix('.download.json').read_text())
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    assert download['passed'] and source_hash == download['sha256']
    with np.load(path, allow_pickle=False) as z:
        keys = ['global_index', 'mlp_name', 'n_samples', 'sample_seed', 'mean', 'post_m2', 'M11', 'pre_mean', 'pre_M11']
        data = {key: z[key] for key in keys}
    identity = {key: data[key].item() for key in keys[:4]}
    assert identity == dict(global_index=57, mlp_name='gregory-wilson', n_samples=100000000, sample_seed=151321001905)
    raw = data['M11'][13]
    true_mean = data['mean'][13]
    assert raw.dtype == np.float32 and raw.shape == (n, n) and true_mean.dtype == np.float64
    true_C = raw.astype(np.float64) - np.outer(true_mean, true_mean)
    true_C = (true_C + true_C.T) / 2
    E = true_C - C; E = (E + E.T) / 2
    raw_diag = np.diag(raw)
    diagonal_ulps = float(np.max(np.abs(raw_diag.astype(np.float64) - data['post_m2'][13]) /
                                  np.maximum(np.spacing(np.abs(raw_diag)).astype(np.float64), 1e-45)))
    next_true_mean = data['pre_mean'][14]
    true_pre = data['pre_M11'][14].astype(np.float64) - np.outer(next_true_mean, next_true_mean)
    affine_error = float(np.linalg.norm(true_pre - w64.T @ true_C @ w64) / np.linalg.norm(true_pre))
    storage_bound = .5 * np.spacing(np.abs(raw)).astype(np.float64)
    storage_bound = (storage_bound + storage_bound.T) / 2
    storage_ratio = float(np.linalg.norm(storage_bound) / np.linalg.norm(E))
    qualification = dict(raw_diagonal_max_ulps=diagonal_ulps, affine_pre14_covariance_relative=affine_error,
                         f32_storage_bound_over_error_rms=storage_ratio,
                         passed=diagonal_ulps <= 2 and affine_error <= 1e-4 and storage_ratio <= .005)
    assert qualification['passed'], qualification
    Z = tilt(E, u, half)
    evals, evecs = np.linalg.eigh(Z)
    selected = np.argsort(-np.abs(evals), kind='stable')[:k]
    Q_oracle = evecs[:, selected]
    target_risk = risk(E, u, beta, factor)
    expected_risk = factor * (2 * np.sum(Z * Z) + np.trace(Z) ** 2)
    risk_identity = float(abs(target_risk / expected_risk - 1))
    assert risk_identity <= 1e-10
    response = nxt['pdf14'] / (2 * nxt['sigma14'])
    target_value = np.sum(w64 * (E @ w64), axis=0) * response
    value_energy = float(np.sum(target_value * target_value))
    predictions = {}
    records = {}
    for name, Q in [('bank_geometry', Q_input), ('oracle_spectral', Q_oracle)]:
        pred_Z, core, shift = project(Z, Q)
        pred_C = tilt(pred_Z, u, inverse_half)
        error = E - pred_C
        pred_value = np.sum(w64 * (pred_C @ w64), axis=0) * response
        risk_ratio = risk(error, u, beta, factor) / target_risk
        value_ratio = float(np.sum((target_value - pred_value) ** 2) / value_energy)
        gradient = Q.T @ (pred_Z - Z) @ Q + .5 * np.trace(pred_Z - Z) * np.eye(k)
        gradient_relative = float(np.linalg.norm(gradient) / max(np.linalg.norm(core), 1e-30))
        assert gradient_relative <= 1e-10
        records[name] = dict(analytic_risk_residual_ratio=risk_ratio, actual_W14_response_residual_ratio=value_ratio,
                             ordinary_covariance_residual_ratio=float(np.sum(error * error) / np.sum(E * E)),
                             trace_correction=shift, subspace_stationarity_relative=gradient_relative,
                             passed=bool(risk_ratio <= .25 and value_ratio <= .25))
        predictions[name + '_covariance'] = pred_C
        predictions[name + '_next_mean'] = pred_value
    result = dict(passed=records['bank_geometry']['passed'], ordinal=57, network='gregory-wilson', role='train',
                  post_layer=13, next_mean_layer=14, fixed_rank=k, geometry_byte_checks=checks,
                  input_basis_frozen_before_target_decode=True, input_basis_sha256=input_basis_hash,
                  qualification_checks=qualification, identity=identity, decoded_source_keys=keys,
                  target_covariance_rms=float(np.sqrt(np.mean(E * E))), target_covariance_energy=float(np.sum(E * E)),
                  target_next_mean_rms=float(np.sqrt(value_energy / n)), target_risk=target_risk,
                  beta=beta, frozen_variance=v0, risk_tilt_identity_relative=risk_identity, bases=records,
                  tilted_error_frobenius_stable_rank=float(np.sum(evals * evals) / np.max(evals * evals)),
                  tilted_bank_covariance_eigenvalue_range=[float(bank_eigenvalues[0]), float(bank_eigenvalues[-1])],
                  tilted_bank_covariance_rank32_trace_fraction=float(np.sum(bank_eigenvalues[-k:]) / np.sum(bank_eigenvalues)),
                  source_sha256=source_hash, provenance=provenance, helper_sha256=helper_hash,
                  wall_s=time.time() - start, peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                  utc_finished=datetime.now(timezone.utc).isoformat(),
                  qualification='Single training-cell oracle representation audit of total covariance error. Bank basis fixed before decoding targets; core coefficients and oracle comparator use100M targets. No independent replica/noise correction, predictor, carrier write or closed-loop raw/adjusted score. No official1e9 gt_mean decoded;100M mean arrays decoded for centering, including unused rows. Fixed-basis optimum, not a global rank32 risk optimum.')
    np.savez(OUT.with_suffix('.npz'), target_covariance_error=E, bank_covariance=C, bank_mean=mean,
             target_next_mean=target_value, input_basis=Q_input, oracle_basis=Q_oracle,
             tilted_error_eigenvalues=evals, **predictions)
    result['output_sha256'] = hashlib.sha256(OUT.with_suffix('.npz').read_bytes()).hexdigest()
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
