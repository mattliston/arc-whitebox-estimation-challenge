"""Inert fixed signed-copula tree action; no fitting or participant rollout."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
OFFENV = NEW.parent / 'data_gate/.venv'
CAPTURE = NEW / 'rows_reflected_residual_copula_tree_v1/row_0035'
OLD = NEW / 'rows_two_memory_teacher_action_v2/row_0035'
END = NEW / 'copula_exact_endpoints_v1/result_v1'
OUT = HERE / 'result_v1'
N, BLOCK, STEPS = 1024, 16384, 26
WALL, RSS = 120, 1_000_000_000
BLAS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
PINS = {
    CAPTURE / 'result.json': 'dfe88ab975fdae184c7636ca197f04e136d6c64734f75840435e8cbfcc6e02fe',
    CAPTURE / 'causal_post12_pre13.npz': '09c00424bfccd50715f77f01181d7b7786bbdb75b22f870d29e1579c163b78a5',
    OLD / 'result.json': '3caa8661c514eb3da6b91fd92932e156f18a4530ba76bc8dc531398b65a02b54',
    OLD / 'pre_13.npz': 'dce092d8e7bb3e11afc5376b247bab332095b7ce79c5be81027300d7fa009eea',
    NEW / 'rows_current_transport_shadow_v2/row_0035/weights.npy': '14563b956b4e326aee081263077c0dd1dcb86037a0b7870dc58a5e33865f47f2',
    NEW / 'reflected_residual_copula_tree_information_v1/marginal_parameters.npz': '6e61c8cffe296e3326d89bf5dfd46bfad872b855b787350277c05878e740b404',
    NEW / 'reflected_residual_copula_tree_information_v1/result.json': '43ccad916ac76ea773763271756a144b644c15795a590c744ac82d2534aa4e05',
    NEW / 'judge_reflected_residual_copula_tree_v1.py': 'e15bf8c1f23f455ea985bfa333deb9357f45a3e0a2b364a77be00a112e7a3aba',
    END / 'result.json': '4d382153725f2778c53a9f5d5f6b3ae019c11904c22de745ce02097b8c7e0c96',
    END / 'endpoint_classification.npz': 'f2c362b2159d4f156ea58d2900c587b7589625d7560783de64c633f0f182d237',
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def binding(path):
    path = Path(path).resolve()
    before = path.stat()
    digest = sha(path)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            'Changed while hashing: ' + str(path))
    return dict(path=str(path), sha256=digest, bytes=after.st_size, mtime_ns=after.st_mtime_ns)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return sha(path)


def peak_rss():
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != 'darwin':
        peak *= 1024
    return peak


def resources(start):
    peak = peak_rss()
    require(peak <= RSS, 'Fixed1GB RSS cap')
    require(time.perf_counter() - start < WALL, 'Fixed120s wall cap')
    return peak


def authenticate():
    paths = dict(PINS)
    capture, old, endpoint = read(CAPTURE / 'result.json'), read(OLD / 'result.json'), read(END / 'result.json')
    require(capture['passed'] and old['passed'] and capture['rank'] == old['rank'] == 35,
            'Wrong qualified capture')
    own = capture['trace_ownership']
    require(not own['current_action_loss_in_causal_artifact'] and not own['inverse_evaluated']
            and not own['predictor_evaluated'] and own['action_loss_key'] == 'pre_K21_loss_only',
            'Wrong source/teacher boundary')
    require(own['old_v2_receipt_sha256'] == PINS[OLD / 'result.json'], 'Capture chronology changed')
    require(old['artifacts']['pre_13.npz']['sha256'] == PINS[OLD / 'pre_13.npz']
            and old['input_hashes']['prefix/weights.npy'] ==
            PINS[NEW / 'rows_current_transport_shadow_v2/row_0035/weights.npy'], 'Teacher/weight provenance')
    require(endpoint['passed'] and endpoint['numerical_completed'] and endpoint['hashes_unchanged']
            and not endpoint['teacher_access'], 'Endpoint prerequisite incomplete')
    require(endpoint['pairs']['strict_feasible'] == 345678 and
            endpoint['pairs']['below_lower'] == 18 and endpoint['pairs']['above_upper'] == 0
            and endpoint['pairs']['unresolved'] == 0, 'Different endpoint population')
    for collection in (capture['source_hashes'], own['checked_files']):
        for name, digest in collection.items():
            path = NEW / name
            require(path not in paths or paths[path] == digest, 'Conflicting source binding')
            paths[path] = digest
    for key, digest in endpoint['hashes'].items():
        if key in ('source', 'bounds', 'protocol'):
            filename = {'source': 'diagnose.py', 'bounds': 'interval_bounds.py', 'protocol': 'PROTOCOL.md'}[key]
            paths[NEW / 'copula_exact_endpoints_v1' / filename] = digest
        elif key == 'decimal':
            paths[NEW / 'current_response_scalar_certificate_v1/scalar_interval.py'] = digest
    sources = {}
    for path, expected in paths.items():
        item = binding(path)
        require(item['sha256'] == expected, 'Frozen input/source hash: ' + str(path))
        sources[str(path.resolve())] = item
    for path in (HERE / 'PROTOCOL.md', Path(__file__).resolve()):
        sources[str(path.resolve())] = binding(path)
    return sources


def rehash(sources):
    for path, expected in sources.items():
        require(binding(path) == expected, 'Changed source/input: ' + path)


def cdf(np, x):
    return np.asarray([.5 * math.erfc(-float(v) / math.sqrt(2)) for v in x], dtype=np.float64)


def price(np, r, ai, aj, gi, gj, sign, scale, nodes, weights, stats):
    """Literal old Price formula, evaluated at a declared quadrature order."""
    rr = sign * r
    integral = np.zeros_like(rr)
    for u, weight in zip(nodes, weights, strict=True):
        t = rr * u
        denominator = 1 - t * t
        integral += weight * (1-u) * np.exp(
            -(ai*ai - 2*t*ai*aj + aj*aj) / (2*denominator)) / np.sqrt(denominator)
    stats['pair_evaluations'] += int(len(r))
    stats['integrand_node_evaluations'] += int(len(r)) * len(nodes)
    return sign * scale * (rr*gi*gj + rr*rr*integral/(2*math.pi))


def plane(np, U, a, q):
    return (U*U)*q[None, :] + 2*(q*a)[:, None]*U.T


def action(np, U, q, residual, W, rows):
    X, WT = W.T @ U, W.T
    tree = (X[rows]*X[rows]) @ (WT*q).T + 2*(X[rows]*WT[rows]) @ (X*q).T
    E = residual - (2/3)*np.diag(np.diag(residual))
    Q = E @ W
    star = (W[:, rows]*W[:, rows]).T @ Q + 2*(W[:, rows]*Q[:, rows]).T @ W
    return tree + star


def synthetic(np, quadrature):
    R = np.array([[1, 1, 1, 0], [1, 1, -1, 0], [1, -1, 1, 0], [0, 0, 0, 1.]], dtype=np.float64)
    witness = np.array([-1., 1, 1, 0])
    require(float(witness @ R @ witness) == -3., 'Indefinite fixture changed')
    a, q = np.array([.75, 1., 1.25, .5]), np.array([.5, -.25, .125, -.75])
    U = a[:, None]*R
    E21 = np.array([[2, -1, 3, 4], [5, 1, -2, 3], [-4, 2, 3, 1], [1, -3, 2, -1.]])/32
    W = np.array([[1, 2, -1, 0], [-2, 1, 1, 3], [1, -1, 2, 1], [2, 0, -1, 1.]])/4
    B = np.array([[2, .25, 0, -.125], [.25, 1.5, .125, 0], [0, .125, 1, .25], [-.125, 0, .25, 1.25]])
    b = np.array([1., -2, 3, -1])/64
    repeated = plane(np, U, a, q)
    residual = E21 - repeated
    encoded = 3*residual.T - 2*np.diag(np.diag(residual))
    dense = np.zeros((4, 4, 4))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                dense[i, j, k] = (q[i]*U[j, i]*U[k, i] + q[j]*U[i, j]*U[k, j]
                    + q[k]*U[i, k]*U[j, k]
                    + ((i == j)*encoded[k, i] + (i == k)*encoded[j, i]
                       + (j == k)*encoded[i, j])/3)
    dense_plane = np.array([[dense[i, i, j] for j in range(4)] for i in range(4)])
    direct = np.einsum('abc,ai,bi,cj->ij', dense, W, W, W)
    actual = action(np, U, q, residual, W, np.arange(4))
    alive = np.array([True, False, True, True])
    for value in (direct, actual):
        diagonal = np.diag(value).copy()
        value *= alive[:, None]*alive[None, :]
        np.fill_diagonal(value, diagonal)
    Bpre, bpre = W.T @ B @ W, W.T @ b
    tr = np.diag(Bpre)[:, None]*bpre[None, :] + 2*Bpre*bpre[:, None]
    dense_trace = np.array([[[b[i]*B[j, k] + b[j]*B[i, k] + b[k]*B[i, j]
                             for k in range(4)] for j in range(4)] for i in range(4)])
    direct += np.einsum('abc,ai,bi,cj->ij', dense_trace, W, W, W)
    actual += tr
    plane_error = float(np.max(np.abs(dense_plane-E21)))
    action_error = float(np.max(np.abs(actual-direct)))
    require(plane_error <= 1e-12*max(1., float(np.max(np.abs(E21)))) and
            action_error <= 1e-12*max(1., float(np.max(np.abs(direct)))), 'Indefinite dense tensor fixture')
    r = np.array([-.8, -.3, 0., .2, .75])
    zero, half, one = np.zeros_like(r), np.full_like(r, .5), np.ones_like(r)
    analytic = r/4 + (np.sqrt(1-r*r) + r*np.arcsin(r)-1)/(2*math.pi)
    errors = {}
    for order in (32, 64):
        stats = dict(pair_evaluations=0, integrand_node_evaluations=0)
        value = price(np, r, zero, zero, half, half, one, one, *quadrature[order], stats)
        errors[str(order)] = float(np.max(np.abs(value-analytic)))
        require(errors[str(order)] <= 1e-12, 'Fixed alpha0 Price fixture')
    return dict(passed=True, indefinite_witness=-3., repeated_max_abs=plane_error,
                masked_action_trace_max_abs=action_error, price_alpha0_max_abs=errors,
                causal_arrays_decoded=False)


def execute():
    start = time.perf_counter()
    require(Path(sys.prefix).resolve() == OFFENV.resolve() and 'anaconda3' not in sys.executable,
            'Fixed OFFPY only')
    require(not any(k == 'torch' or k.startswith(('torch.', 'scipy')) for k in sys.modules),
            'No Torch/SciPy imports')
    require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'nice>=10 required')
    require(all(os.environ.get(k) == '1' for k in BLAS) and
            os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'Explicit BLAS1/no-bytecode launch')
    require(not OUT.exists(), 'Refusing overwrite or automatic retry')
    OUT.mkdir(exist_ok=False)
    result = dict(passed=False, completed=False, numerical_qualified=False, stage='preflight',
                  target_access=False, raw_score_claim=False, current409f_compatibility_claim=False,
                  probability_law_claim=False, latent_PSD_required=False, rank=35,
                  full_exact_Gaussian_third_tensor_claim=False,
                  post_layer=12, next_layer=13, started_utc=datetime.now(timezone.utc).isoformat(),
                  pid=os.getpid(), source_bindings={}, artifacts={},
                  price_work=dict(pair_evaluations=0, integrand_node_evaluations=0))
    sources = {}
    def alarm(signum, frame):
        raise TimeoutError('Fixed120s signed-tree diagnostic cap')
    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(WALL)
    try:
        sources = authenticate()
        result['source_bindings'] = sources
        result['versions'] = {name: importlib.metadata.version(name) for name in ('numpy', 'threadpoolctl')}
        require(result['versions'] == dict(numpy='2.4.6', threadpoolctl='3.6.0'), 'Frozen native versions')
        write(OUT / 'started.json', result)
        import numpy as np
        from numpy.polynomial.legendre import leggauss
        from threadpoolctl import threadpool_info, threadpool_limits
        np.seterr(over='raise', divide='raise', invalid='raise', under='ignore')
        quadrature = {}
        for order in (32, 64):
            nodes, weights = leggauss(order)
            quadrature[order] = ((nodes+1)*.5, weights*.5)
        with threadpool_limits(limits=1):
            result['threadpools'] = threadpool_info()
            require(all(p['num_threads'] == 1 for p in result['threadpools'] if p.get('user_api') == 'blas'),
                    'Reported BLAS pool differs from one thread')
            result['thread_attestation'] = 'Explicit BLAS1 launch; every reported pool agrees; empty report is not independent proof.'
            result['stage'] = 'synthetic'
            result['synthetic'] = synthetic(np, quadrature)
            write(OUT / 'synthetic.json', result['synthetic'])
            resources(start)
            print('Fixed indefinite tensor and Price fixtures passed; loading causal arrays.', flush=True)
            result['stage'] = 'causal_inputs'
            fields = {'post_C', 'post_B', 'post_b', 'post_m', 'post_variance', 'post_K21',
                      'post_alive', 'post_radial', 'pre13_Bpre', 'pre13_bpre', 'pre13_alive',
                      'pre13_alpha', 'pre13_sigma'}
            with np.load(CAPTURE / 'causal_post12_pre13.npz', allow_pickle=False) as z:
                require(set(z.files) == fields, 'Unexpected causal fields')
                state = {key: z[key] for key in fields}
            with np.load(NEW / 'reflected_residual_copula_tree_information_v1/marginal_parameters.npz', allow_pickle=False) as z:
                require(set(z.files) == {'live_indices', 'gamma', 'gaussian', 'alpha', 'eta', 'tau', 'sigma', 'a', 'q'}, 'Marginal fields')
                saved = {key: z[key] for key in z.files}
            with np.load(END / 'endpoint_classification.npz', allow_pickle=False) as z:
                endpoint = {key: z[key] for key in ('live_indices', 'ii', 'jj', 'classification',
                            'lower_lo', 'lower_hi', 'upper_lo', 'upper_hi', 'target_lo', 'target_hi')}
            require(all(np.isfinite(x).all() for x in (*state.values(), *saved.values(), *endpoint.values())), 'Nonfinite causal input')
            C, B, b, S = (state[key].astype(np.float64) for key in ('post_C', 'post_B', 'post_b', 'post_K21'))
            alive = state['post_alive']
            require(C.shape == B.shape == S.shape == (N, N) and b.shape == (N,) and
                    alive.shape == (N,) and alive.dtype == np.bool_, 'Wrong frozen state shapes')
            variance = np.diag(C).copy()
            require(np.array_equal(variance, state['post_variance']), 'Variance ownership')
            expected_B = C.copy()
            np.fill_diagonal(expected_B, variance*alive)
            require(np.array_equal(B, expected_B) and not np.any(b[~alive]), 'Actual trace metric/vector ownership')
            del expected_B
            indices = np.flatnonzero(alive)
            require(len(indices) == 832 and np.array_equal(indices, saved['live_indices']) and
                    np.array_equal(indices, endpoint['live_indices']), 'Changed live support')
            alpha, eta, tau, sigma, a, q = (saved[key].astype(np.float64) for key in ('alpha', 'eta', 'tau', 'sigma', 'a', 'q'))
            require(all(x.shape == (832,) for x in (alpha, eta, tau, sigma, a, q)) and
                    np.all(tau > 0) and np.all(sigma > 0) and np.all(np.abs(eta) == 1)
                    and not np.any(saved['gaussian']), 'Frozen non-Gaussian marginal domain')
            require(np.array_equal(sigma, np.sqrt(variance[indices])), 'Stored sigma owner')
            trace = np.diag(B)[:, None]*b[None, :] + 2*B*b[:, None]
            E21 = S-trace
            require(np.array_equal(np.diag(E21)[indices]/sigma**3, saved['gamma']), 'Trace-subtracted marginal owner')
            live_C = C[np.ix_(indices, indices)]
            auxiliary = (live_C+live_C.T)*.5
            ii, jj = np.triu_indices(832, 1)
            require(np.array_equal(ii, endpoint['ii']) and np.array_equal(jj, endpoint['jj']), 'Endpoint pair ordering')
            classification = endpoint['classification']
            require(classification.shape == ii.shape and int((classification == -1).sum()) == 18
                    and int((classification == 1).sum()) == 345678, 'Wrong fixed clip population')
            clipped = classification == -1
            require(np.all(eta[ii[clipped]]*eta[jj[clipped]] == 1) and
                    np.all(alpha[ii[clipped]] < 0) and np.all(alpha[jj[clipped]] < 0), 'Clipped endpoint semantics changed')
            target = auxiliary[ii, jj]/(sigma[ii]*sigma[jj])
            require(np.all(target >= endpoint['target_lo']) and np.all(target <= endpoint['target_hi']), 'Target differs from certified reference')
            g = cdf(np, alpha)
            r, c32, c64 = np.empty(len(ii)), np.empty(len(ii)), np.empty(len(ii))
            result['stage'] = 'fixed_clipped_inverse'
            result['inverse'] = dict(iterations=STEPS, inverse_order=64, check_order=32,
                                    clipped_count=18, live_count=832, pair_count=int(len(ii)),
                                    covariance_mutated=False, latent_PSD_checked=False, chunks=[])
            for begin in range(0, len(ii), BLOCK):
                end = min(begin+BLOCK, len(ii))
                i, j = ii[begin:end], jj[begin:end]
                free = ~clipped[begin:end]
                sign = eta[i]*eta[j]
                scale = (tau[i]/sigma[i])*(tau[j]/sigma[j])
                block_r = np.full(end-begin, -1., dtype=np.float64)
                low, high = np.full(int(free.sum()), -1.), np.full(int(free.sum()), 1.)
                for iteration in range(STEPS):
                    mid = (low+high)*.5
                    value = price(np, mid, alpha[i[free]], alpha[j[free]], g[i[free]], g[j[free]],
                                  sign[free], scale[free], *quadrature[64], result['price_work'])
                    below = value < target[begin:end][free]
                    low, high = np.where(below, mid, low), np.where(below, high, mid)
                block_r[free] = (low+high)*.5
                r[begin:end] = block_r
                for order, destination in ((32, c32), (64, c64)):
                    destination[begin:end] = price(np, block_r, alpha[i], alpha[j], g[i], g[j],
                        sign, scale, *quadrature[order], result['price_work'])
                quad = np.abs(c64[begin:end]-c32[begin:end])
                error = c64[begin:end]-target[begin:end]
                endpoint_gap = np.maximum(endpoint['lower_lo'][begin:end]-c64[begin:end],
                                          c64[begin:end]-endpoint['lower_hi'][begin:end])
                rec = dict(begin=begin, end=end, clipped=int((~free).sum()),
                    max_quadrature_discrepancy=float(quad.max()),
                    max_unclipped_target_residual=float(np.max(np.abs(error[free]))) if free.any() else 0.,
                    max_clipped_target_residual=float(np.max(np.abs(error[~free]))) if (~free).any() else 0.,
                    max_clipped_endpoint_discrepancy=max(0., float(np.max(endpoint_gap[~free]))) if (~free).any() else 0.,
                    elapsed_s=time.perf_counter()-start)
                result['inverse']['chunks'].append(rec)
                print(json.dumps(rec, sort_keys=True), flush=True)
                require(np.isfinite(block_r).all() and np.isfinite(error).all() and
                        rec['max_quadrature_discrepancy'] <= 1e-8 and
                        rec['max_unclipped_target_residual'] <= 1e-6 and
                        rec['max_clipped_endpoint_discrepancy'] <= 1e-8,
                        'Fixed inverse/independent-quadrature numerical guard')
                resources(start)
            result['inverse'].update(max_standardized_quadrature_discrepancy=float(np.max(np.abs(c64-c32))),
                max_unclipped_standardized_residual=float(np.max(np.abs((c64-target)[~clipped]))),
                max_clipped_standardized_residual=float(np.max(np.abs((c64-target)[clipped]))),
                all_pair_residual_relative=float(np.linalg.norm(c64-target)/max(float(np.linalg.norm(target)), 1e-300)),
                clipped_target_residuals=(c64-target)[clipped].tolist(),
                formal_abs_offdiagonal_max=float(np.max(np.abs(r))))
            R = np.eye(832)
            R[ii, jj], R[jj, ii] = r, r
            require(np.array_equal(R, R.T) and np.array_equal(np.diag(R), np.ones(832)), 'Formal R symmetry/diagonal')
            U, full_a, full_q = np.zeros((N, N)), np.zeros(N), np.zeros(N)
            U[np.ix_(indices, indices)] = a[:, None]*R
            full_a[indices], full_q[indices] = a, q
            tree21 = plane(np, U, full_a, full_q)
            residual = E21-tree21
            encoded = 3*residual.T-2*np.diag(np.diag(residual))
            restored = tree21+(encoded.T+2*np.diag(np.diag(encoded)))/3+trace
            difference = restored-S
            result['restoration'] = dict(relative_frobenius=float(np.linalg.norm(difference)/max(float(np.linalg.norm(S)), 1e-300)),
                normalized_max_abs=float(np.max(np.abs(difference))/max(float(np.max(np.abs(S))), 1e-300)))
            require(result['restoration']['relative_frobenius'] <= 1e-10, 'Repeated-plane restoration')
            del encoded, restored, difference
            result['stage'] = 'fixed_teacher_action_prediction'
            weights = np.load(NEW / 'rows_current_transport_shadow_v2/row_0035/weights.npy', mmap_mode='r', allow_pickle=False)
            require(weights.shape == (16, N, N) and weights.dtype == np.float32, 'Fixed weights shape/dtype')
            W, rows = weights[13].astype(np.float64), np.arange(0, N, 32)
            prediction = action(np, U, full_q, residual, W, rows)
            diagonal = prediction[np.arange(len(rows)), rows].copy()
            pre_alive = state['pre13_alive']
            require(pre_alive.shape == (N,) and pre_alive.dtype == np.bool_, 'Actual next mask')
            prediction *= pre_alive[rows, None]*pre_alive[None, :]
            prediction[np.arange(len(rows)), rows] = diagonal
            Bpre, bpre = state['pre13_Bpre'], state['pre13_bpre']
            require(Bpre.shape == (N, N) and bpre.shape == (N,), 'Actual next trace shape')
            prediction += np.diag(Bpre)[rows, None]*bpre[None, :] + 2*Bpre[rows]*bpre[rows, None]
            pre_alpha, pre_sigma = state['pre13_alpha'], state['pre13_sigma']
            require(pre_alpha.shape == pre_sigma.shape == (N,) and np.all(pre_sigma > 0), 'Actual next scalar units')
            weight = (.5*np.exp(-.5*pre_alpha[rows]**2)/(math.sqrt(2*math.pi)*pre_sigma[rows]))[:, None]*cdf(np, pre_alpha)[None, :]
            require(np.isfinite(prediction).all() and np.isfinite(weight).all(), 'Nonfinite fixed predictor')
            with (OUT / 'sealed_predictor.npz').open('xb') as stream:
                np.savez(stream, query_rows=rows, predicted_K21=prediction, consumer_weight=weight,
                    U=U, q=full_q, a=full_a, residual_star_plane=residual, formal_R=R,
                    live_indices=indices, ii=ii, jj=jj, clipped=clipped,
                    standardized_target=target, covariance32=c32, covariance64=c64)
            result['artifacts']['sealed_predictor.npz'] = binding(OUT / 'sealed_predictor.npz')
            result['numerical_qualified'] = True
            result['action_product_cost'] = dict(square_products=2, rectangular_products=4,
                query_rows=32, n=N, classical_multiply_adds=2*N*N*(2*N-1)+4*32*N*(2*N-1),
                fp64_meter_multiplier=2, complete_bill=False,
                excludes='Inverse, scalar functions, elementwise operations, checks, IO and future acquisition/rollout')
            rehash(sources)
            resources(start)
            seal = write(OUT / 'predictor_seal.json', dict(sealed_utc=datetime.now(timezone.utc).isoformat(),
                target_access=False, predictor=result['artifacts']['sealed_predictor.npz'],
                sources=sources, numerical_qualified=True, result_before_teacher=result))
            result['predictor_seal_sha256'] = seal
            result['artifacts']['predictor_seal.json'] = binding(OUT / 'predictor_seal.json')
            require(result['artifacts']['predictor_seal.json']['sha256'] == seal,
                    'Predictor seal changed before teacher access')
            # ONLY NOW decode the one original same-frame teacher plane.
            result['stage'], result['target_access'] = 'sole_teacher_fidelity', True
            with np.load(OLD / 'pre_13.npz', allow_pickle=False) as z:
                teacher = z['pre_K21_loss_only'][rows].astype(np.float64)
            require(teacher.shape == (32, N) and np.isfinite(teacher).all(), 'Fixed teacher plane')
            delta = prediction-teacher
            denominator = float(np.linalg.norm(weight*teacher))
            require(math.isfinite(denominator) and denominator > 0, 'Finite nonzero teacher norm required')
            fidelity = float(np.linalg.norm(weight*delta)/denominator)
            result['fidelity'] = dict(weighted_relative=fidelity, fixed_gate=.03,
                unweighted_relative=float(np.linalg.norm(delta)/max(float(np.linalg.norm(teacher)), 1e-300)),
                diagonal_relative=float(np.linalg.norm(delta[np.arange(32), rows])/max(float(np.linalg.norm(teacher[np.arange(32), rows])), 1e-300)),
                weighted_teacher_norm=denominator, teacher_norm=float(np.linalg.norm(teacher)),
                prediction_norm=float(np.linalg.norm(prediction)), passed=bool(math.isfinite(fidelity) and fidelity <= .03))
            with (OUT / 'teacher_fidelity_loss_only.npz').open('xb') as stream:
                np.savez(stream, query_rows=rows, teacher_K21_loss_only=teacher, error_loss_only=delta)
            result['artifacts']['teacher_fidelity_loss_only.npz'] = binding(OUT / 'teacher_fidelity_loss_only.npz')
            result['completed'] = True
            result['stage'] = 'complete'
            result['passed'] = result['fidelity']['passed']
            result['contextual_old_comparison'] = dict(practical=.337814, reference=.316703,
                paired_frame=False, raw_accuracy_comparison=False,
                note='Older covariance-metric replay; different covariance/trace arithmetic and physical-correlation squared legs.')
        require(not any(k == 'torch' or k.startswith(('torch.', 'scipy')) for k in sys.modules), 'Forbidden numerical package')
        rehash(sources)
        rehash({str(OUT / name): value for name, value in result['artifacts'].items()})
        result['peak_rss_bytes'] = resources(start)
        result['hashes_unchanged'] = True
    except BaseException:
        result['passed'] = False
        result['error'] = traceback.format_exc()
    finally:
        signal.alarm(0)
    result['peak_rss_bytes'] = peak_rss()
    result['elapsed_s'] = time.perf_counter()-start
    result['finished_utc'] = datetime.now(timezone.utc).isoformat()
    digest = write(OUT / 'result.json', result)
    write(OUT / 'completion.json', dict(passed=result['passed'], completed=result['completed'], result_sha256=digest))
    print(json.dumps({key: result.get(key) for key in ('passed', 'completed', 'numerical_qualified',
          'stage', 'elapsed_s', 'peak_rss_bytes', 'fidelity', 'error')}, sort_keys=True), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        raise SystemExit(execute())
    parser.print_help()
