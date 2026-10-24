"""One fixed plane-read test; actual parent replay precedes targets."""
from pathlib import Path
import hashlib
import inspect
import json
import resource
import textwrap
import time
import numpy as np
from threadpoolctl import threadpool_limits
from radial_mehler import make_nonlinear
from k3_radial_numpy import gaussian_raw
from chi_radius_trace_shape_v1 import gaussian_planes

HERE = Path(__file__).resolve().parent
OUT = HERE/'chi_radius_trace_shape_value_r35_v1.json'
ART = OUT.with_suffix('.npz')
assert not OUT.exists() and not ART.exists()
assert json.loads((HERE/'chi_radius_trace_shape_certificate_v1.json').read_text())['passed']

original = make_nonlinear(6, 2, True, True)
source = textwrap.dedent(inspect.getsource(original))
source = source.replace('metric="identity"):', 'metric="identity",q31_override=None):')
needle = 'T31=t4[:,None]*rho if k31 else None'
assert source.count(needle) == 1
source = source.replace(needle, needle+'\n    if q31_override is not None: T31=q31_override')
env = dict(original.__globals__, gauss_order=6, edge_order=2, local=True, k31=True)
exec(source, env)
read = env['nonlinear']

cached = np.load(HERE/'current_pair_oracle_attribution_v2/inputs_rank_0035.npz')
exact = np.load(HERE/'fresh_trace_meanmix_exact_r35_v1.npz')
stage = []
arrays = {}
started = time.time()
with threadpool_limits(limits=1):
    for i, layer in enumerate(range(8, 15)):
        mu = exact[f'mu_{layer}']
        C = cached['Cpre'][i]
        K = cached['K21'][i]
        k4 = cached['k4'][i]
        v = np.diag(C)
        approximate_r = float(exact[f'q_{layer}'])*3
        choices = (approximate_r, np.nextafter(approximate_r, -np.inf), np.nextafter(approximate_r, np.inf))
        matches = [r for r in choices if np.array_equal(r*v*v, k4)]
        assert len(matches) == 1, (layer, 'original radial not uniquely recoverable', len(matches))
        r = matches[0]
        K22 = r*(np.outer(v, v)+2*C*C)/3
        parent = read(mu, C, K, k4, K22, 'covariance')
        assert np.array_equal(parent[0], exact[f'parent_mean_{layer}']), (layer, 'original mean')
        assert np.array_equal(parent[1], exact[f'parent_covariance_{layer}']), (layer, 'original covariance')
        sigma = gaussian_raw(mu, v)[0]
        rho = C/np.outer(sigma, sigma)
        np.fill_diagonal(rho, 0.)
        old_q31 = (k4/v**2)[:, None]*rho
        explicit = read(mu, C, K, k4, K22, 'covariance', q31_override=old_q31)
        assert np.array_equal(parent[0], explicit[0])
        assert np.array_equal(parent[1], explicit[1])
        shape = gaussian_planes(mu, (C+C.T)/2, K, r, bank_fourth_trace=float(K22.sum()))
        trace_relative = abs(shape['trace_new']-shape['trace_bank'])/max(abs(shape['trace_bank']), 1e-30)
        assert trace_relative <= 1e-11, (layer, 'trace')
        new_q31 = shape['k31']/(sigma[:, None]**3*sigma[None, :])
        np.fill_diagonal(new_q31, 0.)
        changed = read(mu, C, K, shape['k4'], shape['k22'], 'covariance', q31_override=new_q31)
        delta = changed[1]-parent[1]
        delta = (delta+delta.T)/2
        record = dict(layer=layer, original_call_mean_covariance_byte_exact=True,
                      explicit_original_K31_byte_exact=True,
                      recovered_original_radial=r,
                      original_radial_recovery_ulp_offset=choices.index(r),
                      trace_relative_error=trace_relative,
                      angular_radial=shape['angular_radial'],
                      input_C_symmetrization_max=float(np.max(np.abs((C+C.T)/2-C))),
                      changed_mean_rms=float(np.sqrt(np.mean((changed[0]-parent[0])**2))),
                      covariance_increment_rms=float(np.sqrt(np.mean(delta*delta))))
        stage.append(record)
        arrays[f'parent_covariance_{layer}'] = parent[1]
        arrays[f'changed_covariance_{layer}'] = changed[1]
        arrays[f'delta_covariance_{layer}'] = delta
        arrays[f'changed_mean_{layer}'] = changed[0]
        print(json.dumps(dict(input_qualification=record)), flush=True)

    # No current accuracy arrays are opened before all seven exact-read gates.
    old = np.load(HERE/'rows_covariance_innovation_probe_v2/row_0035/data.npz')
    truth = np.load(HERE.parent/'public_g19_attribution_v1/mps_cov2_rank0035.npz')
    ct = truth['mc_post_cov'].astype(np.float64)
    records = []
    for i, layer in enumerate(range(8, 15)):
        error = ct[layer]-arrays[f'parent_covariance_{layer}']
        error = (error+error.T)/2
        assert np.array_equal(error.astype(np.float32), old['target'][i]), layer
        error = old['target'][i].astype(np.float64)
        delta = arrays[f'delta_covariance_{layer}']
        weight = old['weight'][i].astype(np.float64)
        response = old['response'][i].astype(np.float64)
        def value(matrix):
            return np.sum(weight*(matrix@weight), axis=0)*response
        target, shift = value(error), value(delta)
        bce, nce = float(np.sum(error**2)), float(np.sum((error-delta)**2))
        bve, nve = float(target@target), float(np.sum((target-shift)**2))
        rec = dict(pair_layer=layer, value_layer=layer+1, original_target_byte_exact=True,
                   baseline_covariance_energy=bce, changed_covariance_energy=nce,
                   covariance_error_ratio=nce/bce,
                   baseline_value_energy=bve, changed_value_energy=nve,
                   next_mean_error_ratio=nve/bve,
                   value_cosine=float(target@shift/np.sqrt((target@target)*(shift@shift))),
                   value_relative_norm=float(np.linalg.norm(shift)/np.linalg.norm(target)))
        records.append(rec)
        arrays[f'delta_value_{layer}'] = shift
        print(json.dumps(rec), flush=True)
cr = sum(r['changed_covariance_energy'] for r in records)/sum(r['baseline_covariance_energy'] for r in records)
vr = sum(r['changed_value_energy'] for r in records)/sum(r['baseline_value_energy'] for r in records)
wins = sum(r['next_mean_error_ratio'] < 1 for r in records)
report = dict(rank=35, input_qualification=stage, records=records,
              pooled_covariance_error_ratio=cr, pooled_next_mean_error_ratio=vr,
              improved_value_cells=wins,
              information_gate=bool(cr <= .95 and vr <= .90 and wins >= 5),
              wall_s=time.time()-started,
              peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              qualification='Plane-read information diagnostic only; actual original mean/C/K31 replay and original current pre-frame K22 trace owned exactly. Same seven coherent rank35 cells, old covariance targets byte-exact, existing8M reference and fixed next-weight response. Current-layer mean feedback absent in both reads; previous history fixed. New mean and covariance centering recomputed by actual nonlinear map. No wide replay, new MC, amplitude/radius/layer sweep, full K4-to-K3 carrier, fit, metered bill or score.')
np.savez(ART, **arrays)
OUT.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('records', 'input_qualification')}, indent=2), flush=True)
