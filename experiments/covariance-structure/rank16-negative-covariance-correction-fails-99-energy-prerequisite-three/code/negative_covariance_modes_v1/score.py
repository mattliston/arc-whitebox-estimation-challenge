import resource

from common import *
from geometry import response


def main():
    import numpy as np
    from threadpoolctl import threadpool_info
    assert all(x['num_threads'] == 1 for x in threadpool_info())
    verify();verify('geometry_sealed.json')
    spectrum = read(HERE/'spectrum_result.json');assert spectrum['passed']
    rows = []
    for rec, geometry in zip(captures(), spectrum['rows']):
        rank = rec['rank'];assert geometry['rank'] == rank
        W = np.load(rec['weights']['path'], allow_pickle=False)
        with np.load(truth(rank), allow_pickle=False) as z:
            assert int(z['P']) == 8_000_000
            true_C = z['mc_post_cov'].astype(np.float64)
            true_m = z['mc_post'].astype(np.float64)
            true_pre_C8 = z['mc_pre_cov'][8].astype(np.float64)
            true_pre_m8 = z['mc_pre_m'][8].astype(np.float64)
        affine_covariance_relative = float(np.linalg.norm(W[8].T @ true_C[7] @ W[8]-true_pre_C8)/np.linalg.norm(true_pre_C8))
        affine_mean_relative = float(np.linalg.norm(W[8].T @ true_m[7]-true_pre_m8)/np.linalg.norm(true_pre_m8))
        assert max(affine_covariance_relative, affine_mean_relative) <= 1e-4
        layers = [];payload = {}
        with np.load(rec['capture']['path'], allow_pickle=False) as cap, np.load(geometry['file']['path'], allow_pickle=False) as modes:
            for layer in LAYERS:
                C = cap[f'C_post_{layer:02d}'].astype(np.float64)
                Q, d = modes[f'Q_{layer}'], modes[f'd_{layer}'];correction = (Q*d) @ Q.T
                error = true_C[layer] - C;remaining = error - correction
                weights = W[layer+1].astype(np.float64)
                factor = cap[f'pdf_{layer+1:02d}'].astype(np.float64)/(2*cap[f'sigma_{layer+1:02d}'].astype(np.float64))
                y = factor*np.sum(weights*(error@weights), axis=0)
                z = response(np, Q, d, weights, factor)
                cov_base = float(np.mean(error*error));cov_new = float(np.mean(remaining*remaining))
                response_base = float(np.mean(y*y));response_new = float(np.mean((y-z)**2))
                row = dict(layer=layer, cov_base=cov_base, cov_new=cov_new, response_base=response_base,
                           response_new=response_new, covariance_ratio=cov_new/cov_base,
                           response_ratio=response_new/response_base)
                layers.append(row);payload[f'y_{layer}'] = y;payload[f'z_{layer}'] = z
                print(json.dumps(dict(rank=rank, **row)), flush=True)
        pooled = dict(covariance=sum(r['cov_new'] for r in layers)/sum(r['cov_base'] for r in layers),
                      response=sum(r['response_new'] for r in layers)/sum(r['response_base'] for r in layers))
        passed = pooled['covariance'] <= .98 and pooled['response'] <= .95
        dest = HERE/f'responses_{rank:04d}.npz';np.savez(dest, **payload)
        rows.append(dict(rank=rank, layers=layers, pooled=pooled, passed=passed, responses=pin(dest),
                         truth=pin(truth(rank)), affine_covariance_relative=affine_covariance_relative,
                         affine_mean_relative=affine_mean_relative))
    write(HERE/'score_result.json', dict(completed=True, passed=all(r['passed'] for r in rows), rows=rows,
          target_samples=8_000_000, ordinary_1B_targets_opened=False, fitted_coefficients=0,
          final_score=False, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    print(json.dumps(dict(proxy_passed=all(r['passed'] for r in rows), pooled=[r['pooled'] for r in rows])), flush=True)


if __name__ == '__main__':
    main()
