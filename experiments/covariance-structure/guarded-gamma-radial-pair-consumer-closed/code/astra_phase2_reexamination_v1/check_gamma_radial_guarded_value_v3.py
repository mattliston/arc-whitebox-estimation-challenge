"""Qualify the guarded variant, then one existing-data seven-cell verdict."""
from pathlib import Path
import hashlib
import json
import resource
import time
import numpy as np
from threadpoolctl import threadpool_limits
from gamma_radial_guarded_v3 import guarded_pair
from gamma_radial_pair_v1 import CONDITIONAL

HERE = Path(__file__).resolve().parent
OUT = HERE/'gamma_radial_guarded_value_r35_v3.json'
ART = OUT.with_suffix('.npz')
assert not OUT.exists() and not ART.exists()
old_certificate = json.loads((HERE/'gamma_radial_pair_certificate_v2.json').read_text())
assert not old_certificate['passed']
failed = [r for r in old_certificate['records'] if not r['passed']]
assert len(failed) == 1 and failed[0]['kind'] == 'synthetic' and failed[0]['radial'] == .30
assert old_certificate['scale_moment_max_error'] <= 1e-12
assert all(r['passed'] for r in old_certificate['records'] if r not in failed)
start = time.time()
rng = np.random.default_rng(930064)
a = rng.normal(size=(12, 12))
c = a@a.T/12+np.eye(12)*.2
m = rng.normal(size=12)
k = rng.normal(size=(12, 12))*.001
v = np.diag(c)
for r in (-.02, .30):
    original = CONDITIONAL(m, c, k, r*v*v,
                           r*(np.outer(v, v)+2*c*c)/3, 'covariance')
    for count in (8, 32):
        trial = guarded_pair(m, c, k, r, count)
        assert np.array_equal(trial[0], original[0])
        assert np.array_equal(trial[1], original[1])
certificate = dict(passed=True, universal_v2_passed=False,
                   unchanged_current_seven_cell_numerical_gates=True,
                   unchanged_in_domain_synthetic_numerical_gates=True,
                   outside_domain_parent_byte_exact=True,
                   domain=[0, .10], nodes=8, numerical_reference_nodes=32)
print(json.dumps(dict(guarded_certificate=certificate)), flush=True)

# Accuracy files are opened only after the new method's explicit certificate.
saved = np.load(HERE/'gamma_radial_pair_certificate_v2.npz')
old = np.load(HERE/'rows_covariance_innovation_probe_v2/row_0035/data.npz')
truth = np.load(HERE.parent/'public_g19_attribution_v1/mps_cov2_rank0035.npz')
ct = truth['mc_post_cov'].astype(np.float64)
records, arrays = [], {}
with threadpool_limits(limits=4):
    for i, l in enumerate(range(8, 15)):
        base = saved[f'base_cov_{l}']
        changed = saved[f'mixture_cov_{l}']
        error = ct[l]-base
        error = (error+error.T)/2
        assert np.array_equal(error.astype(np.float32), old['target'][i]), l
        error = old['target'][i].astype(np.float64)
        delta = changed-base
        delta = (delta+delta.T)/2
        w = old['weight'][i].astype(np.float64)
        response = old['response'][i].astype(np.float64)
        value = lambda matrix: np.sum(w*(matrix@w), axis=0)*response
        target, shift = value(error), value(delta)
        bce, nce = float(np.sum(error**2)), float(np.sum((error-delta)**2))
        bve, nve = float(target@target), float(np.sum((target-shift)**2))
        rec = dict(pair_layer=l, value_layer=l+1, original_target_byte_exact=True,
                   baseline_covariance_energy=bce, changed_covariance_energy=nce,
                   covariance_error_ratio=nce/bce,
                   baseline_value_energy=bve, changed_value_energy=nve,
                   next_mean_error_ratio=nve/bve,
                   value_cosine=float(target@shift/np.sqrt((target@target)*(shift@shift))),
                   value_relative_norm=float(np.linalg.norm(shift)/np.linalg.norm(target)))
        records.append(rec)
        arrays[f'delta_cov_{l}'] = delta
        arrays[f'delta_value_{l}'] = shift
        print(json.dumps(rec), flush=True)
cr = sum(r['changed_covariance_energy'] for r in records)/sum(r['baseline_covariance_energy'] for r in records)
vr = sum(r['changed_value_energy'] for r in records)/sum(r['baseline_value_energy'] for r in records)
wins = sum(r['next_mean_error_ratio'] < 1 for r in records)
result = dict(rank=35, guarded_certificate=certificate, records=records,
              pooled_covariance_error_ratio=cr, pooled_next_mean_error_ratio=vr,
              improved_value_cells=wins,
              information_gate=cr<=.95 and vr<=.90 and wins>=5,
              wall_s=time.time()-start,
              peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
              source_sha256=hashlib.sha256((HERE/'gamma_radial_guarded_v3.py').read_bytes()).hexdigest(),
              qualification='One fixed guarded radial consumer; all seven original coherent rank35 cells and existing8M symmetric targets. Exact old target replay. No new Monte Carlo, fitted strength, range/node sweep or rollout. Not final raw error, adjusted score, independent-net replication, metered candidate or submission.')
np.savez(ART, **arrays)
OUT.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2), flush=True)
