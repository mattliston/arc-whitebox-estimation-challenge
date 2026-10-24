"""Exact coherent input replay and fixed numerical gate, never score."""
from pathlib import Path
import hashlib
import json
import resource
import time
import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
import radial_compact_feed_fullpair_v1 as parent
from capture_gamma_pair_inputs_v2 import capture
from fresh_trace_meanmix_consumer_v1 import trace_increment

HERE = Path(__file__).resolve().parent
OUT = HERE/'fresh_trace_meanmix_exact_r35_v1.json'
ART = OUT.with_suffix('.npz')
assert not OUT.exists() and not ART.exists()
assert json.loads((HERE/'fresh_trace_meanmix_algebra_v2.json').read_text())['passed']
trace = {}
parent_covariances = {}
last_layer = -1
original_nonlinear = parent.env['nonlinear']


def capture_trace(layer, bpre, Bpre, radial):
    global last_layer
    last_layer = layer
    if 8 <= layer <= 14:
        trace[layer] = (bpre.copy(), Bpre.copy(), float(radial)/3)


def capture_parent_covariance(*args):
    output = original_nonlinear(*args)
    if 8 <= last_layer <= 14:
        parent_covariances[last_layer] = output[1].copy()
    return output


needle = '        m, C, target, radial_new, aux = nonlinear'
assert parent.PREDICT_SOURCE.count(needle) == 1
parent.PREDICT_SOURCE = parent.PREDICT_SOURCE.replace(
    needle, '        capture_trace(l,bpre,Bpre,radial)\n'+needle)
parent.env = dict(parent.env, capture_trace=capture_trace, nonlinear=capture_parent_covariance)
started = time.time()
records, arrays = [], {}
with threadpool_limits(limits=3):
    data = capture()
    print(json.dumps(dict(exact_original_call_replay_passed=True,
                          trace_layers=sorted(trace), wall_s=time.time()-started)), flush=True)
    for i, layer in enumerate(range(8, 15)):
        b, Braw, q = trace[layer]
        mu = data['mu'][i]
        Craw = data['Cpre'][i]
        C, B = (Craw+Craw.T)/2, (Braw+Braw.T)/2
        V = B-np.outer(b, b)/q
        D = C-B
        cmin, cmax = (float(eigh(C, subset_by_index=(j, j), eigvals_only=True)[0]) for j in (0, 1023))
        vmin = float(eigh(V, subset_by_index=(0, 0), eigvals_only=True)[0])
        dmin = float(eigh(D, subset_by_index=(0, 0), eigvals_only=True)[0])
        dm8, dr8 = trace_increment(mu, C, b, B, q, 8)
        dm32, dr32 = trace_increment(mu, C, b, B, q, 32)
        parent_mean = data['host'][i]
        dc8 = dr8-np.outer(parent_mean, dm8)-np.outer(dm8, parent_mean)-np.outer(dm8, dm8)
        dc32 = dr32-np.outer(parent_mean, dm32)-np.outer(dm32, parent_mean)-np.outer(dm32, dm32)
        mean_relative = float(np.linalg.norm(dm8-dm32)/max(np.linalg.norm(parent_mean+dm32), 1e-30))
        covariance_relative = float(np.linalg.norm(dc8-dc32)/max(np.linalg.norm(parent_covariances[layer]+dc32), 1e-30))
        correction_relative = float(np.linalg.norm(dc8-dc32)/max(np.linalg.norm(dc32), 1e-30))
        rec = dict(layer=layer, q=q, original_input_host_replay_passed=True,
                   input_covariance_symmetrization_max=float(np.max(np.abs(C-Craw))),
                   trace_metric_symmetrization_max=float(np.max(np.abs(B-Braw))),
                   min_C_eigenvalue=cmin, max_C_eigenvalue=cmax,
                   min_V_eigenvalue=vmin, min_D_eigenvalue=dmin,
                   strict_global_PSD=bool(cmin >= 0 and vmin >= 0 and dmin >= 0),
                   mean_8_vs_32_relative=mean_relative,
                   covariance_8_vs_32_relative=covariance_relative,
                   covariance_increment_8_vs_32_relative=correction_relative,
                   correction_mean_rms=float(np.sqrt(np.mean(dm8*dm8))),
                   correction_covariance_rms=float(np.sqrt(np.mean(dc8*dc8))),
                   numerical_gate=bool(mean_relative <= 1e-8 and covariance_relative <= 1e-8
                                       and correction_relative <= 1e-3))
        records.append(rec)
        arrays.update({f'{name}_{layer}': value for name, value in
                       (('mu', mu), ('C', C), ('B', B), ('b', b), ('q', np.array(q)),
                        ('parent_mean', parent_mean), ('delta_mean', dm8),
                        ('parent_covariance', parent_covariances[layer]),
                        ('delta_raw_second', dr8), ('delta_covariance', dc8))})
        print(json.dumps(rec), flush=True)
result = dict(rank=35, records=records,
              all_numerical_gates_pass=all(r['numerical_gate'] for r in records),
              all_strict_global_PSD=all(r['strict_global_PSD'] for r in records),
              accuracy_targets_scored=False,
              wall_s=time.time()-started,
              peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
              source_sha256=hashlib.sha256((HERE/'fresh_trace_meanmix_consumer_v1.py').read_bytes()).hexdigest(),
              qualification='Exact original coherent-history call inputs and host replay are certified by capture_gamma_pair_inputs_v2; history means/covariances are fixed inputs, no current output error/score computed. Input C and B explicitly symmetrized for the proposed probabilistic model, with changes reported. Strict global PSD and eight/32 numerical gates reported separately. Negative eigenvalues are not clipped; successful bivariate evaluations are not a global probability certificate.')
np.savez(ART, **arrays)
OUT.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k != 'records'}, indent=2), flush=True)
