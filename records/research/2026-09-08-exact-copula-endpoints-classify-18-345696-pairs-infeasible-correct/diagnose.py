"""One no-fit endpoint diagnostic, explicitly executed; frozen inputs stay intact."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
MAX_RSS = 512*(1 << 20)
FILES = {
    'marginals': BASE/'reflected_residual_copula_tree_information_v1/marginal_parameters.npz',
    'causal': BASE/'rows_reflected_residual_copula_tree_v1/row_0035/causal_post12_pre13.npz',
    'capture': BASE/'rows_reflected_residual_copula_tree_v1/row_0035/result.json',
    'old_judge': BASE/'judge_reflected_residual_copula_tree_v1.py',
    'old_result': BASE/'reflected_residual_copula_tree_information_v1/result.json',
    'decimal': BASE/'current_response_scalar_certificate_v1/scalar_interval.py',
    'protocol': HERE/'PROTOCOL.md',
    'bounds': HERE/'interval_bounds.py',
    'source': Path(__file__).resolve(),
}
EXPECTED = {
    'marginals': '6e61c8cffe296e3326d89bf5dfd46bfad872b855b787350277c05878e740b404',
    'causal': '09c00424bfccd50715f77f01181d7b7786bbdb75b22f870d29e1579c163b78a5',
    'capture': 'dfe88ab975fdae184c7636ca197f04e136d6c64734f75840435e8cbfcc6e02fe',
    'old_judge': 'e15bf8c1f23f455ea985bfa333deb9357f45a3e0a2b364a77be00a112e7a3aba',
    'decimal': '80a2cb4ec800648a4c7792fd019f9154f29d43b9dbc8d559b15d0329edc520b0',
}


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def execute(out):
    require(not out.exists(), 'Refusing to overwrite a diagnostic')
    require('/data_gate/.venv/' in sys.executable, 'Use the pinned OFFPY interpreter')
    require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'Require nice>=10')
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        require(os.environ.get(name) == '1', 'Set '+name+'=1 before Python')
    def expired(signum, frame):
        raise TimeoutError('Fixed 60s wall or55s CPU diagnostic cap')
    signal.signal(signal.SIGALRM, expired)
    signal.signal(signal.SIGXCPU, expired)
    signal.alarm(60)
    resource.setrlimit(resource.RLIMIT_CPU, (55, 60))
    hashes = {key: sha(path) for key, path in FILES.items()}
    for key, expected in EXPECTED.items():
        require(hashes[key] == expected, 'Frozen hash mismatch: '+key)
    capture = json.loads(FILES['capture'].read_text())
    old = json.loads(FILES['old_result'].read_text())
    require(capture['passed'] and old['passed'] is False, 'Capture/failure ownership')
    require(old['pair']['polynomial_bracket_failure_count'] == 810, 'Wrong old arm')
    out.mkdir(exist_ok=False)
    started = time.perf_counter()
    result = dict(passed=False, numerical_completed=False, stage='starting',
        scope='Old rank35 post12 exact frozen-marginal pair feasibility only',
        teacher_access=False, raw_score_claim=False, current409f_compatibility_claim=False,
        rank=35, post_layer=12, hashes=hashes, priority=os.getpriority(os.PRIO_PROCESS, 0),
        started_utc=datetime.now(timezone.utc).isoformat())
    write_json(out/'started.json', result)
    try:
        import numpy as np
        from interval_bounds import (Bounds, certify, endpoint_arrays, from_decimal,
                                     load_decimal, scalar_cdf)
        np.seterr(over='raise', divide='raise', invalid='raise', under='ignore')
        reference = load_decimal(FILES['decimal'])
        result['stage'] = 'scalar_certificate'
        result['certificate'] = certify(reference)
        write_json(out/'scalar_certificate.json', result['certificate'])
        print('Synthetic endpoint enclosures passed; loading frozen causal inputs.', flush=True)
        with np.load(FILES['marginals'], allow_pickle=False) as z:
            require(set(z.files) == {'live_indices','gamma','gaussian','alpha','eta','tau','sigma','a','q'},
                    'Unexpected marginal fields')
            saved = {name: z[name] for name in z.files}
        with np.load(FILES['causal'], allow_pickle=False) as z:
            C = z['post_C'].astype(np.float64)
            alive = z['post_alive']
        indices, alpha, eta, tau, sigma, gaussian = (saved[k] for k in
            ('live_indices', 'alpha', 'eta', 'tau', 'sigma', 'gaussian'))
        require(C.shape == (1024,1024) and alive.shape == (1024,), 'Wrong Phase2 shape')
        require(np.array_equal(indices, np.flatnonzero(alive)) and len(indices) == 832,
                'Wrong frozen live support')
        require(np.isfinite(C).all() and all(np.isfinite(x).all() for x in saved.values()),
                'Nonfinite frozen input')
        require(np.all(sigma > 0) and np.all(tau > 0) and np.all(np.abs(eta) == 1),
                'Invalid marginal scales/reflection')
        require(np.array_equal(sigma, np.sqrt(np.diag(C)[indices])), 'Sigma ownership')
        require(int(gaussian.sum()) == old['marginal']['gaussian_count'] == 0,
                'Unexpected Gaussian-limit coordinates; no alternate run')
        live = C[np.ix_(indices, indices)]
        aux = (live+live.T)*.5
        del C, live
        ii, jj = np.triu_indices(len(indices), 1)
        require(len(ii) == 345696, 'Pair contract')
        result['stage'] = 'certified_scalar_responses'
        cdfs, pdfs = [], []
        for i, alpha_i in enumerate(alpha):
            cdfs.append(scalar_cdf(reference, float(alpha_i)))
            pdfs.append(reference.normal_pdf(reference.point(float(alpha_i))))
            if i % 128 == 0:
                require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= MAX_RSS, 'RSS cap')
        cdf, pdf = from_decimal(cdfs), from_decimal(pdfs)
        print('All832 scalar CDF/PDF enclosures complete.', flush=True)
        # Reproduce literal old Hermite coefficient spelling, not a new inverse.
        h = np.zeros((len(indices), 13))
        h[:,1] = saved['a']/sigma
        hp, hc = np.zeros(len(indices)), np.ones(len(indices))
        density = np.exp(-.5*alpha*alpha)/math.sqrt(2*math.pi)
        for k in range(2,13):
            if k > 2:
                hp, hc = hc, -alpha*hc-(k-3)*hp
            h[:,k] = tau/sigma*eta**(k+1)*density*hc/math.sqrt(math.factorial(k))
        result['stage'] = 'all_pair_endpoints'
        classification = np.zeros(len(ii), dtype=np.int8)
        poly_bad = np.zeros(len(ii), dtype=np.bool_)
        lower_lo, lower_hi, upper_lo, upper_hi, target_lo, target_hi = [np.empty(len(ii)) for _ in range(6)]
        for begin in range(0,len(ii),16384):
            end = min(begin+16384,len(ii))
            i,j = ii[begin:end], jj[begin:end]
            old_target = aux[i,j]/(sigma[i]*sigma[j])
            coef = (h[i,1:]*h[j,1:]).T.copy()
            old_low, old_high = coef[-1].copy(), coef[-1].copy()
            for k in range(10,-1,-1):
                old_low, old_high = coef[k]-old_low, coef[k]+old_high
            old_low = -old_low
            poly_bad[begin:end] = (old_low > old_target) | (old_high < old_target)
            scale = (Bounds(tau[i])/Bounds(sigma[i]))*(Bounds(tau[j])/Bounds(sigma[j]))
            lo, hi = endpoint_arrays(alpha[i], alpha[j], pdf[i], pdf[j], cdf[i], cdf[j],
                                     scale, eta[i]*eta[j])
            target = Bounds(aux[i,j])/(Bounds(sigma[i])*Bounds(sigma[j]))
            require(np.all(lo.hi < hi.lo), 'Endpoint range enclosure is empty/uncertain')
            feasible = (target.lo > lo.hi) & (target.hi < hi.lo)
            below, above = target.hi < lo.lo, target.lo > hi.hi
            classification[begin:end] = np.where(feasible,1,np.where(below,-1,np.where(above,-2,0)))
            lower_lo[begin:end],lower_hi[begin:end] = lo.lo,lo.hi
            upper_lo[begin:end],upper_hi[begin:end] = hi.lo,hi.hi
            target_lo[begin:end],target_hi[begin:end] = target.lo,target.hi
            require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= MAX_RSS, 'RSS cap')
        require(int(poly_bad.sum()) == 810, 'Did not reproduce the frozen polynomial failure count')
        feasible, bad, unresolved = classification == 1, classification < 0, classification == 0
        opposite = eta[ii]*eta[jj] < 0
        violation_lower = np.maximum(lower_lo-target_hi,target_lo-upper_hi)
        worst = int(np.argmax(violation_lower))
        by_unit = np.bincount(np.concatenate((ii[bad],jj[bad])), minlength=len(indices))
        result['pairs'] = dict(total=len(ii), strict_feasible=int(feasible.sum()),
            strict_infeasible=int(bad.sum()), unresolved=int(unresolved.sum()),
            below_lower=int((classification == -1).sum()), above_upper=int((classification == -2).sum()),
            old_polynomial_failures=int(poly_bad.sum()),
            old_bad_exact_feasible=int((poly_bad&feasible).sum()),
            old_bad_exact_infeasible=int((poly_bad&bad).sum()),
            old_bad_opposite_reflection=int((poly_bad&opposite).sum()),
            exact_bad_opposite_reflection=int((bad&opposite).sum()),
            max_endpoint_interval_width=float(max((lower_hi-lower_lo).max(),(upper_hi-upper_lo).max())),
            largest_certified_standardized_violation=float(max(0,violation_lower.max())),
            worst_pair=[int(indices[ii[worst]]),int(indices[jj[worst]])],
            all_pairs_exact_feasible=bool(feasible.all()))
        result['coordinates'] = [dict(index=int(indices[k]), bad_pairs=int(by_unit[k]),
            alpha=float(alpha[k]), gamma=float(saved['gamma'][k]), eta=float(eta[k]))
            for k in np.flatnonzero(by_unit)]
        with (out/'endpoint_classification.npz').open('xb') as stream:
            np.savez(stream, live_indices=indices, ii=ii,jj=jj,classification=classification,
                old_polynomial_bad=poly_bad, lower_lo=lower_lo,lower_hi=lower_hi,
                upper_lo=upper_lo,upper_hi=upper_hi,target_lo=target_lo,target_hi=target_hi)
        result['stage'] = 'complete'
        result['numerical_completed'] = True
        result['passed'] = True
        result['numpy_version'] = np.__version__
        result['decision'] = ('Exact matching for frozen model ruled out; do not run an inverse.'
            if bad.any() else 'Pairwise condition cleared, not latent PSD or score evidence.'
            if feasible.all() else 'Endpoint feasibility unresolved; no inverse authorized by this diagnostic.')
    except BaseException as exc:
        result['exception_type'], result['exception'] = type(exc).__name__, str(exc)
        result['passed'] = False
    finally:
        signal.alarm(0)
        result['wall_s'] = time.perf_counter()-started
        result['peak_rss_bytes'] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        result['finished_utc'] = datetime.now(timezone.utc).isoformat()
        result['final_hashes'] = {key:sha(path) for key,path in FILES.items()}
        result['hashes_unchanged'] = hashes == result['final_hashes']
        result['passed'] = bool(result['passed'] and result['hashes_unchanged']
                                and result['peak_rss_bytes'] <= MAX_RSS and result['wall_s'] < 60)
        result['artifacts'] = {p.name:sha(p) for p in out.glob('*.npz')}
        write_json(out/'result.json', result)
        print(json.dumps({k:v for k,v in result.items() if k not in
              ('hashes','final_hashes','coordinates')}, allow_nan=False), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path, default=HERE/'result_v1')
    args = parser.parse_args()
    if args.execute:
        raise SystemExit(execute(args.output))
    print('INERT: one no-fit frozen-input endpoint diagnostic; --execute required.')
