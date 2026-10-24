"""Fixed full-plane consumer bridge from preserved q4 source artifacts."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import ndtr
from threadpoolctl import threadpool_limits
from residual_one_high_oracle_v1 import distinct_plane
from current_pair_oracle_math_v1 import exact_change

HERE = Path(__file__).resolve().parent
OUT = HERE / 'one_high_actual_value_r35_l12_v1.json'
ART = OUT.with_suffix('.npz')
assert not OUT.exists() and not ART.exists()
f = np.load(HERE / 'residual_one_high_inputs_r35_l12_v1.npz')
current = np.load(HERE / 'current_pair_oracle_attribution_v2/inputs_rank_0035.npz')
old = np.load(HERE / 'rows_covariance_innovation_probe_v2/row_0035/data.npz')
truth = np.load(HERE.parent / 'public_g19_attribution_v1/mps_cov2_rank0035.npz')
planes = np.load(HERE.parent / 'public_g19_attribution_v1/mps_planes_rank0035.npz')
idx = 13 - 8
mu, C, K, host = (current[k][idx].astype(np.float64) for k in ('mu', 'Cpre', 'K21', 'host'))
parity = float(np.max(np.abs(K[f['indices']] - f['model_next_K21'])))
assert parity < 1e-7, parity
s = np.sqrt(np.diag(C))
a = mu / s
pdf = np.exp(-a*a/2) / np.sqrt(2*np.pi)
g = ndtr(a)
common = (C / np.outer(s, s), s, a, pdf, g, pdf + a*g, host/s)
E = old['target'][idx].astype(np.float64)
wn = old['weight'][idx].astype(np.float64)
resp = old['response'][idx].astype(np.float64)


def value(mat):
    return np.sum(wn * (mat @ wn), axis=0) * resp


def measure(delta):
    dm, dc = exact_change(common, dS=delta / (s[:, None]**2 * s[None, :]))
    v = value(E)
    dv = value(dc)
    return dict(covariance_error_ratio=float(np.sum((E-dc)**2) / np.sum(E*E)),
                next_mean_error_ratio=float(np.sum((v-dv)**2) / np.sum(v*v)),
                value_correction_cosine=float(v@dv / np.sqrt((v@v)*(dv@dv))),
                value_correction_relative_norm=float(np.linalg.norm(dv)/np.linalg.norm(v)),
                current_one_step_mean_mse=float(np.mean((host + dm - truth['mc_post'][13])**2))), dc, dv


start = time.time()
results = {}
art = {}
with threadpool_limits(limits=4):
    for arm in ('oracle_A', 'oracle_B', 'exact_heavy', 'positive_path'):
        if arm.startswith('oracle_'):
            name = arm[-1]
            t = np.load(HERE / f'residual_one_high_truth_r35_l12_{name}_v1.npz')
            slices = t['slices'] - f['model_slices']
        else:
            file = ('cumulative_exact_heavy_source_r35_l12_v1.npz' if arm == 'exact_heavy'
                    else 'cumulative_positive_path_source_r35_l12_v1.npz')
            slices = np.load(HERE / file)['accumulated_source_slices']
        delta = distinct_plane(slices, f['U'], f['next_weight'], np.arange(1024))
        results[arm], dc, dv = measure(delta)
        art[arm + '_delta_K21'] = delta
        art[arm + '_delta_cov'] = dc
        art[arm + '_delta_value'] = dv
        print(json.dumps(dict(arm=arm, **results[arm])), flush=True)
    trueK = planes['K21_13'].astype(np.float64)
    np.fill_diagonal(trueK, truth['mc_pre_k3'][13])
    results['full_K3_reference'], _, _ = measure(trueK - K)
    delta_mean = .5*(art['oracle_A_delta_K21'] + art['oracle_B_delta_K21'])
    results['oracle_AB_average'], _, _ = measure(delta_mean)
gate = all(results[arm]['next_mean_error_ratio'] <= .75 and
           results[arm]['covariance_error_ratio'] <= 1
           for arm in ('oracle_A', 'oracle_B', 'exact_heavy'))
np.savez(ART, **art)
record = dict(rank=35, source_post_layer=12, pair_layer=13, value_layer=14,
              incoming_parent_plane_max_difference=parity,
              baseline_covariance_mse=float(np.mean(E*E)),
              baseline_next_mean_response_mse=float(np.mean(value(E)**2)),
              results=results, actual_value_bridge_gate=gate,
              wall_s=time.time()-start,
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              qualification='Full1024-row exact distinct q4 projection, natural-strength finite pair response. Original8M post13 covariance label is not independent of the original host trajectory; residual slice streams A/B are independent8M. Point diagnostic only, not a final-score ceiling, deployable residual source, FLOPs or adjusted score.')
OUT.write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record, indent=2), flush=True)
