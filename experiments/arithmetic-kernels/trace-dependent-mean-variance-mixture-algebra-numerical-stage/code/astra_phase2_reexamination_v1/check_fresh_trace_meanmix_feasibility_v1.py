"""Necessary-PD screening on archived rounded bank tapes; no targets."""
from pathlib import Path
import hashlib
import json
import resource
import time
import numpy as np
from scipy.linalg import cholesky, solve_triangular, eigh
from threadpoolctl import threadpool_limits
from radial_mehler import make_nonlinear

HERE = Path(__file__).resolve().parent
OUT = HERE/'fresh_trace_meanmix_feasibility_v1.json'
assert not OUT.exists()
assert json.loads((HERE/'fresh_trace_meanmix_algebra_v2.json').read_text())['passed']
nonlinear = make_nonlinear(6, 2, True, True)
records = []
start = time.time()
with threadpool_limits(limits=1):
    for rank in (35, 100):
        folder = HERE/'rows_current_transport_shadow_v2'/f'row_{rank:04d}'
        tape = {k: np.load(folder/(k+'.npy'), mmap_mode='r') for k in
                ('pre_cov', 'mu', 'k21', 'k4', 'host', 'delta', 'weights', 'alive')}
        assert all(value.dtype == np.float32 for value in tape.values())
        b = np.zeros(1024)
        for layer in range(14):
            C = tape['pre_cov'][layer].astype(np.float64)
            mu = tape['mu'][layer].astype(np.float64)
            K = tape['k21'][layer].astype(np.float64)
            k4 = tape['k4'][layer].astype(np.float64)
            v = np.maximum(np.diag(C), 1e-10)
            selected = v >= max(float(v.max())*1e-5, 1e-9)
            radial_values = k4[selected]/v[selected]**2
            radial = float(np.median(radial_values))
            k22 = radial*(np.outer(v, v)+2*C*C)/3
            host, covariance, _, _, aux = nonlinear(mu, C, K, k4, k22, 'covariance')
            delta = tape['delta'][layer].astype(np.float64)
            covariance -= np.outer(host, delta)+np.outer(delta, host)+np.outer(delta, delta)
            alive = tape['alive'][layer] > .5
            wp = tape['weights'][layer].astype(np.float64)
            bpre = wp.T @ b
            b = (aux['gate']*bpre+radial*aux['pdf']*v/(6*aux['sigma']))*alive
            if layer < 7:
                continue
            next_v = np.maximum(np.diag(tape['pre_cov'][layer+1]).astype(np.float64), 1e-10)
            selection = next_v >= max(float(next_v.max())*1e-5, 1e-9)
            qs = tape['k4'][layer+1][selection].astype(np.float64)/next_v[selection]**2/3
            q = float(np.median(qs))
            active_cov = covariance[np.ix_(alive, alive)]
            active_cov = (active_cov+active_cov.T)/2
            active_b = b[alive]
            minimum = float(eigh(active_cov, subset_by_index=(0, 0), eigvals_only=True)[0])
            rec = dict(rank=rank, pre_layer=layer+1, active_size=int(alive.sum()),
                       q=q, q_relative_spread=float(np.max(np.abs(qs-q))/max(abs(q), 1e-30)),
                       active_covariance_min_eigenvalue=minimum,
                       inactive_b_max=float(np.max(np.abs(b[~alive]))) if (~alive).any() else 0.,
                       min_cold_variance=float(np.diag(covariance)[~alive].min()) if (~alive).any() else None,
                       host_from_rounded_tape_max_difference=float(np.max(np.abs(host-tape['host'][layer]))))
            try:
                factor = cholesky(active_cov, lower=True)
                whitened = solve_triangular(factor, active_b, lower=True)
                cost = float(whitened@whitened)
                rec.update(mahalanobis_b=cost, feasibility_ratio=cost/q,
                           ten_percent_necessary_margin_pass=bool(q > 0 and cost <= .9*q))
            except np.linalg.LinAlgError:
                rec.update(mahalanobis_b=None, feasibility_ratio=None,
                           ten_percent_necessary_margin_pass=False)
            records.append(rec)
            print(json.dumps(rec), flush=True)
result = dict(records=records,
              all_necessary_margins_pass=all(r['ten_percent_necessary_margin_pass'] for r in records),
              exact_probability_certificate=False,
              wall_s=time.time()-start,
              peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              qualification='Only existing F32 bank input tapes, ranks35/100, fixed pre-layers8..14. Reconstructed b and active post-covariance from rounded mu/C/K21/k4 and frozen deltas. This is a necessary-condition screen, not exact replay, a PSD certificate, or a value/score result. No target arrays read, no clipping of covariance or source strengths.')
OUT.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k != 'records'}, indent=2), flush=True)
