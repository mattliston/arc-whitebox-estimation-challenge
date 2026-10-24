"""Fixed formal pair-closure value test; strict global-PD failure stays."""
from pathlib import Path
import hashlib
import json
import resource
import time
import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
OUT = HERE/'fresh_trace_meanmix_formal_value_r35_v1.json'
ART = OUT.with_suffix('.npz')
assert not OUT.exists() and not ART.exists()
certificate = json.loads((HERE/'fresh_trace_meanmix_exact_r35_v1.json').read_text())
assert certificate['all_numerical_gates_pass']
assert not certificate['all_strict_global_PSD']
saved = np.load(HERE/'fresh_trace_meanmix_exact_r35_v1.npz')
old = np.load(HERE/'rows_covariance_innovation_probe_v2/row_0035/data.npz')
truth = np.load(HERE.parent/'public_g19_attribution_v1/mps_cov2_rank0035.npz')
ct = truth['mc_post_cov'].astype(np.float64)
records, arrays = [], {}
start = time.time()
with threadpool_limits(limits=1):
    for i, layer in enumerate(range(8, 15)):
        base = saved[f'parent_covariance_{layer}']
        error = ct[layer]-base
        error = (error+error.T)/2
        assert np.array_equal(error.astype(np.float32), old['target'][i]), layer
        error = old['target'][i].astype(np.float64)
        delta = saved[f'delta_covariance_{layer}']
        delta = (delta+delta.T)/2
        w = old['weight'][i].astype(np.float64)
        response = old['response'][i].astype(np.float64)
        def value(matrix):
            return np.sum(w*(matrix@w), axis=0)*response
        target, shift = value(error), value(delta)
        base_covariance_energy = float(np.sum(error*error))
        changed_covariance_energy = float(np.sum((error-delta)**2))
        base_value_energy = float(target@target)
        changed_value_energy = float(np.sum((target-shift)**2))
        rec = dict(pair_layer=layer, value_layer=layer+1,
                   original_target_byte_exact=True, strict_global_PSD=False,
                   baseline_covariance_energy=base_covariance_energy,
                   changed_covariance_energy=changed_covariance_energy,
                   covariance_error_ratio=changed_covariance_energy/base_covariance_energy,
                   baseline_value_energy=base_value_energy,
                   changed_value_energy=changed_value_energy,
                   next_mean_error_ratio=changed_value_energy/base_value_energy,
                   value_cosine=float(target@shift/np.sqrt((target@target)*(shift@shift))),
                   value_relative_norm=float(np.linalg.norm(shift)/np.linalg.norm(target)))
        records.append(rec)
        arrays[f'delta_value_{layer}'] = shift
        print(json.dumps(rec), flush=True)
cr = sum(r['changed_covariance_energy'] for r in records)/sum(r['baseline_covariance_energy'] for r in records)
vr = sum(r['changed_value_energy'] for r in records)/sum(r['baseline_value_energy'] for r in records)
wins = sum(r['next_mean_error_ratio'] < 1 for r in records)
result = dict(rank=35, records=records, strict_global_PSD=False,
              pooled_covariance_error_ratio=cr,
              pooled_next_mean_error_ratio=vr, improved_value_cells=wins,
              information_gate=bool(cr <= .95 and vr <= .90 and wins >= 5),
              wall_s=time.time()-start,
              peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              qualification='Separate explicitly authorized FORMAL pair-moment closure diagnostic; exact-state global PSD failed and remains failed. Same seven coherent rank35 cells, original targets byte-exact, existing8M reference, fixed current-parent next Gaussian response. Natural coefficient1, no new samples, node/q/domain/strength sweeps, fit or state rollout. Not final MSE, adjusted score, independent-network validation, a positive joint distribution or a candidate.')
np.savez(ART, **arrays)
OUT.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k != 'records'}, indent=2), flush=True)
