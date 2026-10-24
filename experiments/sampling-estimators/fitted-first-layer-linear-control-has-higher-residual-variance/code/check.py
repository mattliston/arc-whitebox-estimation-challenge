"""One fixed, target-free antithetic first-layer CV capacity diagnostic.

Inert unless --execute is supplied. This is not an estimator, a measured
complete-cost test, or a scored candidate. Existing targets/models are never
loaded. See PROTOCOL.md and IMPLEMENTATION.md before executing.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback


HERE = Path(__file__).resolve().parent
NEW = HERE.parent
P2 = NEW.parent
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
MANIFEST_SHA = '9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50'
AP = Path('/Users/matt/anaconda3')
RANKS = (35, 100)
WIDTH, DEPTH = 1024, 16
NTRAIN, NTEST, BATCH = 4096, 4096, 128
BLOCKS, SPOT = 16, 8
TRAIN_SALT, TEST_SALT = 0x464C435654524149, 0x464C435654455354
SEED_MASK = (1 << 63) - 1
WALL_LIMIT, MEMORY_LIMIT = 300.0, 4_000_000_000
BLAS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
        'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
CONDITION_LIMIT = 1e8
CHOLESKY_REL_LIMIT = 1e-12
SPOT_RMS_LIMIT, SPOT_MAX_LIMIT = 1e-5, 1e-4
PAID_SUB3 = 6.556167934102733e-5
PAID_SUB1 = 2.1853893113675777e-5
PAIR_CLASSICAL_F = 32 * WIDTH * (2 * WIDTH - 1)
IDEAL_SUB3 = 3e-9 * (2 ** 41) / PAIR_CLASSICAL_F
IDEAL_SUB1 = 1e-9 * (2 ** 41) / PAIR_CLASSICAL_F
PAID_TOTAL_F, PAID_SETUP_F, PAID_PAIRS = 1e12, 300e9, 9938


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def binding(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), sha256=sha(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def utc():
    return datetime.now(timezone.utc).isoformat()


def swap_usage():
    # Read-only OS metadata, not a numerical worker or memory-changing command.
    return subprocess.run(['/usr/sbin/sysctl', '-n', 'vm.swapusage'],
                          check=True, capture_output=True, text=True, timeout=5).stdout.strip()


class Guard:
    def __init__(self, start):
        self.start = start
        self.torch = None
        self.samples = []

    def sample(self, stage, enforce=True):
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        driver = int(self.torch.mps.driver_allocated_memory()) if self.torch is not None else 0
        current = int(self.torch.mps.current_allocated_memory()) if self.torch is not None else 0
        item = dict(stage=stage, elapsed_s=time.monotonic() - self.start,
                    peak_process_rss_bytes=rss, current_mps_driver_bytes=driver,
                    current_mps_tensor_bytes=current,
                    conservative_combined_bytes=rss + driver)
        self.samples.append(item)
        if enforce:
            require(item['elapsed_s'] < WALL_LIMIT, 'Fixed 300-second diagnostic limit')
            require(rss + driver < MEMORY_LIMIT, 'Combined RSS plus current MPS driver limit')
        return item


def resolve_rows():
    manifest = STORE / 'manifest.json'
    require(sha(manifest) == MANIFEST_SHA, 'Canonical feature-store manifest changed')
    records = json.loads(manifest.read_text())['roles']['observer_fit']['row_manifests']
    pins, rows = {str(manifest.resolve()): binding(manifest)}, {}
    for rank in RANKS:
        match = [item for item in records if item['role_rank'] == rank]
        require(len(match) == 1, f'Unique canonical observer_fit rank {rank}')
        row_path = (STORE / match[0]['path']).resolve()
        require(row_path.is_relative_to(STORE.resolve()), 'Canonical row outside store')
        require(sha(row_path) == match[0]['sha256'], 'Canonical row manifest changed')
        row = json.loads(row_path.read_text())
        weights_path = row_path.parent / 'weights.npy'
        weight_pin = binding(weights_path)
        require(weight_pin['sha256'] == row['files']['weights.npy']['sha256'], 'Canonical weights changed')
        seed = int(row['mlp_seed'])
        rows[rank] = dict(rank=rank, mlp_id=match[0]['mlp_id'], mlp_seed=seed,
                          weights=weight_pin,
                          train_seed=(seed ^ TRAIN_SALT) & SEED_MASK,
                          test_seed=(seed ^ TEST_SALT) & SEED_MASK)
        require(rows[rank]['train_seed'] != rows[rank]['test_seed'], 'Independent stream seed collision')
        pins[str(row_path)] = binding(row_path)
        pins[str(weights_path)] = weight_pin
    require(len({rows[r][s] for r in RANKS for s in ('train_seed', 'test_seed')}) == 4,
            'Distinct fixed streams across ranks and splits')
    return rows, pins


def feature_gram(np, weights0):
    w = np.asarray(weights0, dtype=np.float64)
    pre_cov = w.T @ w
    sigma = np.sqrt(np.diag(pre_cov))
    require(np.isfinite(pre_cov).all() and np.all(sigma > 0), 'Finite positive first-layer scales')
    scale = sigma[:, None] * sigma[None, :]
    rho_raw = pre_cov / scale
    excursion = float(np.max(np.maximum(np.abs(rho_raw) - 1., 0.)))
    require(excursion <= 1e-12, 'First-layer Gaussian correlation outside roundoff range')
    rho = np.clip(rho_raw, -1., 1.)
    root = np.sqrt(np.maximum(1. - rho * rho, 0.))
    # sqrt(1-rho^2)-1 = -rho^2/(1+sqrt(1-rho^2)): no small-rho subtraction.
    covariance = scale / (2. * np.pi) * (rho * np.arcsin(rho) - rho * rho / (1. + root))
    covariance = .5 * (covariance + covariance.T)
    mean = sigma / np.sqrt(2. * np.pi)
    eigenvalues = np.linalg.eigvalsh(covariance)
    require(np.isfinite(eigenvalues).all() and eigenvalues[0] > 0, 'Analytic feature Gram must be SPD')
    condition = float(eigenvalues[-1] / eigenvalues[0])
    require(condition <= CONDITION_LIMIT, 'Analytic feature Gram conditioning guard')
    lower = np.linalg.cholesky(covariance)
    error = float(np.linalg.norm(lower @ lower.T - covariance) / np.linalg.norm(covariance))
    require(error <= CHOLESKY_REL_LIMIT, 'Analytic feature Gram Cholesky reconstruction')
    return mean, lower, dict(minimum_eigenvalue=float(eigenvalues[0]),
        maximum_eigenvalue=float(eigenvalues[-1]), spectral_condition=condition,
        cholesky_relative_frobenius_error=error, maximum_rho_excursion=excursion,
        feature_mean_min=float(mean.min()), feature_mean_max=float(mean.max()),
        formula='Cov(abs(X W0)/2), zero-mean Gaussian input, frozen float32 weights interpreted as real')


def cpu_reference(np, weights, inputs):
    """Independent CPU64 forward on precisely the same rounded input rows."""
    z = inputs.astype(np.float64) @ np.asarray(weights[0], dtype=np.float64)
    values = np.concatenate((np.maximum(z, 0.), np.maximum(-z, 0.)), axis=0)
    for layer in range(1, DEPTH):
        values = np.maximum(values @ np.asarray(weights[layer], dtype=np.float64), 0.)
    count = len(inputs)
    return dict(plus=values[:count], minus=values[count:],
                even=.5 * (values[:count] + values[count:]), feature=.5 * np.abs(z))


def acquire(np, torch, weights, device_weights, seed, count, guard, label, keep_raw):
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    even = np.empty((count, WIDTH), dtype=np.float64)
    features = np.empty_like(even)
    raw = np.empty_like(even) if keep_raw else None
    input_hash, feature_hash, branch_hash = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    spot = None
    began = time.monotonic()
    with torch.inference_mode():
        for offset in range(0, count, BATCH):
            # The seed and conversion are fixed. CPU64 checks use these same float32 rows.
            inputs = rng.standard_normal((BATCH, WIDTH)).astype(np.float32)
            input_hash.update(inputs.tobytes())
            tensor = torch.from_numpy(inputs).to('mps')
            z = tensor @ device_weights[0]
            values = torch.cat((torch.relu(z), torch.relu(-z)), dim=0)
            for layer in range(1, DEPTH):
                values = torch.relu(values @ device_weights[layer])
            branches = values.cpu().numpy().copy()
            first = z.cpu().numpy().copy()
            require(np.isfinite(branches).all() and np.isfinite(first).all(), 'Finite MPS paths')
            plus, minus = branches[:BATCH].astype(np.float64), branches[BATCH:].astype(np.float64)
            e = .5 * (plus + minus)
            h = .5 * np.abs(first.astype(np.float64))
            even[offset:offset + BATCH] = e
            features[offset:offset + BATCH] = h
            if keep_raw:
                raw[offset:offset + BATCH] = plus
            feature_hash.update(h.tobytes())
            branch_hash.update(branches.tobytes())
            if offset == 0:
                ref_start = time.monotonic()
                reference = cpu_reference(np, weights, inputs[:SPOT])
                spot = dict(mps=dict(plus=plus[:SPOT].copy(), minus=minus[:SPOT].copy(),
                            even=e[:SPOT].copy(), feature=h[:SPOT].copy()), cpu=reference,
                            cpu_reference_seconds=time.monotonic() - ref_start,
                            same_input_sha256=hashlib.sha256(inputs[:SPOT].tobytes()).hexdigest())
            del tensor, z, values
            guard.sample(f'{label}/batch_{offset // BATCH:02d}')
            if offset % (8 * BATCH) == 0:
                print(json.dumps(dict(stage=label, acquired=offset + BATCH, total=count,
                                      elapsed_s=time.monotonic() - guard.start)), flush=True)
    return even, features, raw, spot, dict(seed=seed, rows=count, batch=BATCH,
        input_sha256=input_hash.hexdigest(), feature_sha256=feature_hash.hexdigest(),
        branch_output_sha256=branch_hash.hexdigest(), wall_s=time.monotonic() - began,
        feature_dtype='float64 arithmetic on native float32 first preactivation',
        true_forward_dtype='MPS float32', antithetic_average_dtype='float64')


def fit_once(np, features, response, mean, lower):
    u = features - mean
    white = np.linalg.solve(lower, u.T).T
    white_mean, response_mean = white.mean(axis=0), response.mean(axis=0)
    centered = white - white_mean
    q, upper = np.linalg.qr(centered, mode='reduced')
    singular = np.linalg.svd(upper, compute_uv=False)
    require(np.isfinite(singular).all() and singular[-1] > 0, 'Full-rank TRAIN QR, no fallback')
    condition = float(singular[0] / singular[-1])
    require(condition <= CONDITION_LIMIT, 'TRAIN QR conditioning guard')
    theta = np.linalg.solve(upper, q.T @ (response - response_mean))
    coefficients = np.linalg.solve(lower.T, theta)
    intercept = response_mean - white_mean @ theta
    require(np.isfinite(coefficients).all() and np.isfinite(intercept).all(), 'Finite OLS coefficients')
    residual = response - u @ coefficients - intercept
    # TRAIN statistics are diagnostics only, never a selection or evidence gate.
    variance = float(np.sum((residual - residual.mean(axis=0)) ** 2) / ((NTRAIN - 1) * WIDTH))
    return coefficients, intercept, dict(qr_spectral_condition=condition,
        minimum_qr_singular_value=float(singular[-1]), maximum_qr_singular_value=float(singular[0]),
        centered_train_residual_variance=variance, fit_count=1, rank=WIDTH,
        coefficient_frobenius_norm=float(np.linalg.norm(coefficients)),
        fit='analytic-Sigma whitening, centered TRAIN-only reduced QR, no ridge or rank truncation')


def spot_diagnostics(np, spot, coefficients, mean):
    values = {}
    for key in ('plus', 'minus', 'even', 'feature'):
        delta = spot['mps'][key] - spot['cpu'][key]
        values[key] = dict(rms=float(np.sqrt(np.mean(delta * delta))), max_abs=float(np.max(np.abs(delta))))
    mps_r = spot['mps']['even'] - (spot['mps']['feature'] - mean) @ coefficients
    cpu_r = spot['cpu']['even'] - (spot['cpu']['feature'] - mean) @ coefficients
    delta = mps_r - cpu_r
    values['controlled_residual'] = dict(rms=float(np.sqrt(np.mean(delta * delta))),
        max_abs=float(np.max(np.abs(delta))))
    values.update(paths=SPOT, same_input_sha256=spot['same_input_sha256'],
        cpu_reference_seconds=spot['cpu_reference_seconds'],
        interpretation='Fixed-path numerical-scale diagnostic, not a population rounding bound')
    values['passed'] = (values['controlled_residual']['rms'] <= SPOT_RMS_LIMIT and
                        values['controlled_residual']['max_abs'] <= SPOT_MAX_LIMIT)
    return values


def trace_block(np, values):
    mean = values.mean(axis=0)
    centered = values - mean
    return dict(count=len(values), mean=mean, ss=float(np.sum(centered * centered)))


def merge_trace(np, blocks):
    first, *rest = blocks
    count, mean, ss = first['count'], first['mean'].copy(), first['ss']
    for block in rest:
        other = block['count']
        delta = block['mean'] - mean
        ss += block['ss'] + float(delta @ delta) * count * other / (count + other)
        mean += delta * other / (count + other)
        count += other
    return ss / ((count - 1) * WIDTH)


def interval_summary(np, estimate, leave_block_out):
    values = np.asarray(leave_block_out, dtype=np.float64)
    require(values.shape == (BLOCKS,) and np.isfinite(values).all() and math.isfinite(estimate),
            'Finite sixteen-block uncertainty statistics')
    se = float(np.sqrt((BLOCKS - 1.) / BLOCKS * np.sum((values - values.mean()) ** 2)))
    return dict(estimate=float(estimate), empirical_standard_error=se,
        empirical_lower_95=float(estimate - 1.96 * se), empirical_upper_95=float(estimate + 1.96 * se),
        delete_block_estimates=values.tolist(), blocks=BLOCKS, rows_per_block=NTEST // BLOCKS,
        uncertainty='Delete-one-contiguous-independent-input-block jackknife; asymptotic empirical 95%, not certified coverage')


def variance_summary(np, values):
    size = NTEST // BLOCKS
    blocks = [trace_block(np, values[j * size:(j + 1) * size]) for j in range(BLOCKS)]
    full = merge_trace(np, blocks)
    return interval_summary(np, full, [merge_trace(np, blocks[:j] + blocks[j + 1:]) for j in range(BLOCKS)])


def joint_block(np, residual, whitened):
    mean_r, mean_u = residual.mean(axis=0), whitened.mean(axis=0)
    r, u = residual - mean_r, whitened - mean_u
    return dict(count=len(residual), mean_r=mean_r, mean_u=mean_u,
                ss=float(np.sum(r * r)), cross=r.T @ u)


def merge_joint(np, blocks):
    first, *rest = blocks
    count = first['count']
    mean_r, mean_u = first['mean_r'].copy(), first['mean_u'].copy()
    ss, cross = first['ss'], first['cross'].copy()
    for block in rest:
        other = block['count']
        dr, du = block['mean_r'] - mean_r, block['mean_u'] - mean_u
        scale = count * other / (count + other)
        ss += block['ss'] + float(dr @ dr) * scale
        cross += block['cross'] + scale * np.outer(dr, du)
        mean_r += dr * other / (count + other)
        mean_u += du * other / (count + other)
        count += other
    return ss / ((count - 1) * WIDTH), cross / (count - 1)


def projection_floor(np, residual, whitened, guard):
    size = NTEST // BLOCKS
    blocks = []
    for j in range(BLOCKS):
        sl = slice(j * size, (j + 1) * size)
        blocks.append(joint_block(np, residual[sl], whitened[sl]))
        guard.sample(f'floor/cross_block_{j:02d}')

    def evaluate(omit=None):
        left = [b for j, b in enumerate(blocks[:BLOCKS // 2]) if j != omit]
        right = [b for j, b in enumerate(blocks[BLOCKS // 2:], BLOCKS // 2) if j != omit]
        v1, b1 = merge_joint(np, left)
        v2, b2 = merge_joint(np, right)
        explained = float(np.sum(b1 * b2) / WIDTH)
        return .5 * (v1 + v2) - explained, dict(half_variances=[v1, v2],
                    cross_half_projection_energy=explained)

    point, detail = evaluate()
    result = interval_summary(np, point, [evaluate(j)[0] for j in range(BLOCKS)])
    result.update(detail, independent_half_rows=NTEST // 2,
        negative_estimate=bool(point < 0),
        identity='Var(R)/n - ||Cov(R,u) Sigma^{-1/2}||_F^2/n; class target invariant to frozen TRAIN A',
        numerical_scope='Analytic ideal-Gaussian Sigma with finite-precision feature/forward evaluations; same-path rounding checked separately')
    if point < 0:
        result['status'] = 'inconclusive_negative_estimate'
    elif result['empirical_lower_95'] > IDEAL_SUB3:
        result['status'] = 'above_optimistic_sub3_classical_IID_yardstick'
    elif result['empirical_upper_95'] < IDEAL_SUB3:
        result['status'] = 'low_optimistic_floor_not_a_deployable_constructor'
    else:
        result['status'] = 'inconclusive_at_optimistic_sub3_yardstick'
    return result


def planning_decision(records):
    residuals = [item['test']['fitted_residual_variance'] for item in records]
    floors = [item['test']['optimistic_projection_floor'] for item in records]
    return dict(
        both_fixed_readouts_support_paid_sub3_investigation=all(x['empirical_upper_95'] < PAID_SUB3 for x in residuals),
        both_fixed_readouts_support_paid_sub1_investigation=all(x['empirical_upper_95'] < PAID_SUB1 for x in residuals),
        both_class_floors_above_optimistic_sub3_yardstick=all(
            x['estimate'] >= 0 and x['empirical_lower_95'] > IDEAL_SUB3 for x in floors),
        any_negative_floor=any(x['estimate'] < 0 for x in floors),
        scope='Empirical IID capacity/planning evidence only; no achieved score, complete-cost qualification or automatic follow-up',
        thresholds=dict(paid_sub3=PAID_SUB3, paid_sub1=PAID_SUB1,
            optimistic_sub3=IDEAL_SUB3, optimistic_sub1=IDEAL_SUB1),
        unpaid_planning=dict(total_flops=PAID_TOTAL_F, construction_flops=PAID_SETUP_F,
            correction_pairs=PAID_PAIRS, exact_classical_leading_pair_products=PAIR_CLASSICAL_F,
            assumed_pair_allowance=1.05 * PAIR_CLASSICAL_F, actual_complete_flops_measured=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--source-sha256')
    parser.add_argument('--protocol-sha256')
    args = parser.parse_args()
    if not args.execute:
        parser.print_help()
        return 0
    require(args.source_sha256 == sha(__file__), 'Explicit reviewed checker source hash required')
    require(args.protocol_sha256 == sha(HERE / 'PROTOCOL.md'), 'Explicit frozen protocol hash required')
    require(Path(sys.prefix).resolve() == AP.resolve(), 'Use the existing AP environment only')
    require(sys.platform == 'darwin', 'macOS RSS units/MPS convention')
    require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'nice-10 or lower scheduling priority')
    require(all(os.environ.get(key) == '1' for key in BLAS), 'Set all fixed CPU BLAS controls to 1 before launch')
    require(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') == '0', 'No hidden CPU MPS fallback')
    require('threadpoolctl' not in sys.modules, 'No AP plus threadpoolctl')
    out = HERE / 'result_v1'
    require(not out.exists(), 'Exclusive one-shot output; no overwrite or automatic resume')
    out.mkdir()
    start = time.monotonic()
    guard = Guard(start)
    result = dict(completed=False, numerically_qualified=False, scored_candidate=False,
        existing_truth_files_accessed=False, existing_fitted_models_read=False,
        new_generated_function_responses=True, started_utc=utc(), ranks=list(RANKS),
        train_rows=NTRAIN, test_rows=NTEST, batch=BATCH, device='mps', cpu_threads=1,
        source_bindings={}, fixture_bindings={}, records=[], artifacts={}, phase_seconds={})
    write(out / 'started.json', dict(result, requested_source_sha256=args.source_sha256,
          requested_protocol_sha256=args.protocol_sha256, environment={key: os.environ.get(key) for key in BLAS}))

    def deadline(_signal, _frame):
        raise TimeoutError('Frozen diagnostic wall limit; no retry')

    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, WALL_LIMIT)
    try:
        files = (Path(__file__), HERE / 'PROTOCOL.md', HERE / 'IMPLEMENTATION.md')
        annex = HERE / 'NUMERICAL_ANNEX.md'
        if annex.exists():
            files += (annex,)
        result['source_bindings'] = {str(p.resolve()): binding(p) for p in files}
        rows, result['fixture_bindings'] = resolve_rows()
        result['fixed_rows'] = rows
        result['swap_before'] = swap_usage()
        import numpy as np
        import torch
        require('threadpoolctl' not in sys.modules, 'Forbidden AP import occurred')
        require(torch.backends.mps.is_available(), 'One available MPS device required')
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        guard.torch = torch
        result['versions'] = dict(python=sys.version, numpy=np.__version__, torch=torch.__version__,
                                  executable=sys.executable)
        result['phase_seconds']['imports_and_provenance'] = time.monotonic() - start
        guard.sample('initialized')
        for rank in RANKS:
            record = dict(rank=rank, phase_seconds={})
            result['records'].append(record)
            began = time.monotonic()
            wn = np.load(rows[rank]['weights']['path'], mmap_mode='r', allow_pickle=False)
            require(wn.shape == (DEPTH, WIDTH, WIDTH) and wn.dtype == np.float32 and np.isfinite(wn).all(),
                    'Canonical fixed MLP weight shape/dtype/finite contract')
            wt = torch.tensor(np.asarray(wn), device='mps', dtype=torch.float32)
            mean, lower, record['feature_gram'] = feature_gram(np, wn[0])
            record['phase_seconds']['load_and_analytic_gram'] = time.monotonic() - began
            guard.sample(f'{rank}/analytic_gram')
            train_y, train_h, _, train_spot, record['train_acquisition'] = acquire(
                np, torch, wn, wt, rows[rank]['train_seed'], NTRAIN, guard, f'{rank}/TRAIN', False)
            began = time.monotonic()
            coefficients, intercept, record['fit'] = fit_once(np, train_h, train_y, mean, lower)
            record['phase_seconds']['fit'] = time.monotonic() - began
            record['train_numerical_spot'] = spot_diagnostics(np, train_spot, coefficients, mean)
            require(record['train_numerical_spot']['passed'], 'TRAIN same-path controlled-residual numerical guard')
            artifact = out / f'rank_{rank:04d}_coefficients.npz'
            with artifact.open('xb') as handle:
                np.savez(handle, coefficients=coefficients, training_intercept=intercept,
                         analytic_first_feature_mean=mean)
            result['artifacts'][artifact.name] = binding(artifact)
            seal = out / f'rank_{rank:04d}_training_seal.json'
            write(seal, dict(rank=rank, frozen_before_any_test_inputs=True,
                coefficients=result['artifacts'][artifact.name], train_acquisition=record['train_acquisition'],
                fit=record['fit'], numerical_spot=record['train_numerical_spot'],
                source_sha256=args.source_sha256, protocol_sha256=args.protocol_sha256,
                weights=rows[rank]['weights'], sealed_utc=utc()))
            result['artifacts'][seal.name] = binding(seal)
            del train_y, train_h, train_spot
            gc.collect()
            guard.sample(f'{rank}/TRAIN_sealed_before_TEST')
            test_y, test_h, plain, test_spot, record['test_acquisition'] = acquire(
                np, torch, wn, wt, rows[rank]['test_seed'], NTEST, guard, f'{rank}/TEST', True)
            record['test_numerical_spot'] = spot_diagnostics(np, test_spot, coefficients, mean)
            require(record['test_numerical_spot']['passed'], 'TEST same-path controlled-residual numerical guard')
            began = time.monotonic()
            u = test_h - mean
            residual = test_y - u @ coefficients - intercept
            whitened = np.linalg.solve(lower, u.T).T
            require(np.isfinite(residual).all() and np.isfinite(whitened).all(), 'Finite TEST residual/features')
            record['test'] = dict(plain_variance=variance_summary(np, plain),
                antithetic_variance=variance_summary(np, test_y),
                fitted_residual_variance=variance_summary(np, residual),
                optimistic_projection_floor=projection_floor(np, residual, whitened, guard))
            fitted = record['test']['fitted_residual_variance']
            fitted['paid_expected_adjusted_proxy'] = fitted['estimate'] / PAID_PAIRS * PAID_TOTAL_F / (2 ** 41)
            fitted['paid_proxy_upper_95'] = fitted['empirical_upper_95'] / PAID_PAIRS * PAID_TOTAL_F / (2 ** 41)
            record['phase_seconds']['test_statistics'] = time.monotonic() - began
            require(sha(artifact) == result['artifacts'][artifact.name]['sha256'], 'Readout changed after seal')
            record['numerically_qualified'] = True
            rank_receipt = out / f'rank_{rank:04d}_result.json'
            write(rank_receipt, record)
            result['artifacts'][rank_receipt.name] = binding(rank_receipt)
            print(json.dumps(dict(rank=rank, residual_variance=fitted['estimate'],
                residual_upper_95=fitted['empirical_upper_95'],
                class_floor=record['test']['optimistic_projection_floor']['estimate'])), flush=True)
            del wn, wt, coefficients, intercept, mean, lower, test_y, test_h, plain, test_spot, u, residual, whitened
            gc.collect()
            torch.mps.empty_cache()
            guard.sample(f'{rank}/complete')
        result['decision'] = planning_decision(result['records'])
        for group in ('source_bindings', 'fixture_bindings', 'artifacts'):
            require(all(sha(item['path']) == item['sha256'] for item in result[group].values()),
                    f'End-of-run unchanged hashes: {group}')
        result['swap_after'] = swap_usage()
        guard.sample('all_complete')
        result.update(completed=True, numerically_qualified=True, hashes_unchanged=True,
            actual_complete_deployment_flops_measured=False,
            construction_path_count_per_net=NTRAIN, independent_test_pairs_per_net=NTEST,
            diagnostic_cost_scope='All actual acquisition/fit/test wall and memory included; classical planning subtotals are not metered participant FLOPs')
    except BaseException:
        result['error'] = traceback.format_exc()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        guard.sample('final_receipt', enforce=False)
        result.update(finished_utc=utc(), wall_seconds=time.monotonic() - start,
            resource_samples=guard.samples,
            maximum_recorded_combined_bytes=max(x['conservative_combined_bytes'] for x in guard.samples),
            peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            maximum_recorded_mps_driver_bytes=max(x['current_mps_driver_bytes'] for x in guard.samples))
        write(out / 'result.json', result)
        digest = sha(out / 'result.json')
        write(out / 'completion.json', dict(completed=result['completed'],
            numerically_qualified=result['numerically_qualified'], result_sha256=digest,
            truth_files_accessed=False, finished_utc=utc()))
        print(json.dumps(dict(completed=result['completed'], numerically_qualified=result['numerically_qualified'],
                              result_sha256=digest, wall_seconds=result['wall_seconds'])), flush=True)
    return 0 if result['completed'] and result['numerically_qualified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
