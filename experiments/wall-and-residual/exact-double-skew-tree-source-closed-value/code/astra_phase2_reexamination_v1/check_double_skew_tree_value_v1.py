"""All seven frozen rank35 source cells: exact one-step physical source."""
from pathlib import Path
import hashlib
import json
import resource
import time
import numpy as np
from threadpoolctl import threadpool_limits
import radial_compact_feed_fullpair_v1 as parent
from radial_binary_compact import select
from current_pair_oracle_math_v1 import exact_change
from double_skew_tree_source_v1 import plane

HERE = Path(__file__).resolve().parent
OUT = HERE/'double_skew_tree_value_r35_v1.json'
ART = OUT.with_suffix('.npz')
assert not OUT.exists() and not ART.exists()
assert json.loads((HERE/'double_skew_tree_source_certificate_v1.json').read_text())['passed']
W = np.load(HERE/'rows_current_transport_shadow_v2/row_0035/weights.npy')
current = np.load(HERE/'current_pair_oracle_attribution_v2/inputs_rank_0035.npz')
old = np.load(HERE/'rows_covariance_innovation_probe_v2/row_0035/data.npz')
truth = np.load(HERE.parent/'public_g19_attribution_v1/mps_cov2_rank0035.npz')
mt, ct = truth['mc_post'].astype(np.float64), truth['mc_post_cov'].astype(np.float64)
records, arrays, source_planes = [], {}, {}
last_inputs = None
original_nonlinear = parent.env['nonlinear']
start = time.time()


class Done(Exception):
    pass


def nonlinear(*args):
    global last_inputs
    last_inputs = args
    return original_nonlinear(*args)


def hook(l, m, C, K, aux, w):
    if 8 <= l <= 14:
        idx = l-8
        for name, v in (('mu', aux['alpha']*aux['sigma']),
                        ('Cpre', aux['pre_cov']), ('K21', K), ('host', m)):
            assert np.array_equal(v, current[name][idx]), (l, name)
        E0 = ct[l]-C
        E0 = (E0+E0.T)/2
        assert np.array_equal(E0.astype(np.float32), old['target'][idx]), l
        E = old['target'][idx].astype(np.float64)
        delta = source_planes[l].copy()
        diag = np.diag(delta).copy()
        _, alive = select(aux['gate'], .01)
        delta *= alive[:,None]*alive[None,:]
        np.fill_diagonal(delta, diag)
        s, a, pdf, g = (aux[k] for k in ('sigma', 'alpha', 'pdf', 'gate'))
        common = (aux['pre_cov']/np.outer(s,s), s, a, pdf, g, pdf+a*g, m/s)
        dm, dc = exact_change(common, dS=delta/(s[:,None]**2*s[None,:]))
        args = list(last_inputs)
        assert np.array_equal(args[2], K)
        args[2] = K+delta
        finite = original_nonlinear(*args)
        mean_error = float(np.max(np.abs((finite[0]-m)-dm)))
        cov_error = float(np.max(np.abs((finite[1]-C)-dc)))
        assert max(mean_error, cov_error) < 1e-12, (l, mean_error, cov_error)
        wn, resp = old['weight'][idx].astype(np.float64), old['response'][idx].astype(np.float64)
        value = lambda matrix: np.sum(wn*(matrix@wn),axis=0)*resp
        v, dv = value(E), value(dc)
        bce, nce = float(np.sum(E*E)), float(np.sum((E-dc)**2))
        bve, nve = float(v@v), float(np.sum((v-dv)**2))
        rec = dict(source_layer=l-1, pair_layer=l, value_layer=l+1,
                   parent_and_label_byte_exact=True, finite_mean_max_error=mean_error,
                   finite_cov_max_error=cov_error, baseline_covariance_energy=bce,
                   changed_covariance_energy=nce, covariance_error_ratio=nce/bce,
                   baseline_value_energy=bve, changed_value_energy=nve,
                   next_mean_error_ratio=nve/bve,
                   value_cosine=float(v@dv/np.sqrt((v@v)*(dv@dv))),
                   value_relative_norm=float(np.linalg.norm(dv)/np.linalg.norm(v)))
        records.append(rec)
        arrays[f'delta_K21_{l}'] = delta
        arrays[f'delta_cov_{l}'] = dc
        arrays[f'delta_value_{l}'] = dv
        print(json.dumps(rec), flush=True)
    if l == 14:
        raise Done
    if 7 <= l <= 13:
        g, s, a, pdf = (aux[k] for k in ('gate', 'sigma', 'alpha', 'pdf'))
        d = pdf/s
        e = -a*d/s
        h = (a*a-1)*d/(s*s)
        _, alive = select(g, .01)
        source_planes[l+1] = plane(K, g, d, e, h, W[l+1].astype(np.float64), alive)
    return mt[l]-m if l >= 1 else np.zeros_like(m)


def cov_hook(l, C):
    return ct[l].copy()


source = parent.PREDICT_SOURCE
for before, after in (
    ('gate_floor=.001):', 'gate_floor=.001, cov_hook=None):'),
    ('            m = m+dm', '            m = m+dm\n        if cov_hook is not None:\n            C = cov_hook(l,C)'),
    ('        radial = radial_new', '        radial = radial_new\n        if cov_hook is not None:\n            C = cov_hook(l,C)')):
    assert source.count(before) == 1
    source = source.replace(before, after)
env = dict(parent.env, nonlinear=nonlinear)
exec(source, env)
with threadpool_limits(limits=4):
    try:
        env['predict'](W, hook=hook, cov_hook=cov_hook, gate_floor=.01,
                       cov32=True, metric='covariance')
    except Done:
        pass
assert len(records) == 7
cov_ratio = sum(r['changed_covariance_energy'] for r in records)/sum(r['baseline_covariance_energy'] for r in records)
val_ratio = sum(r['changed_value_energy'] for r in records)/sum(r['baseline_value_energy'] for r in records)
wins = sum(r['next_mean_error_ratio'] < 1 for r in records)
result = dict(rank=35, records=records, pooled_covariance_error_ratio=cov_ratio,
              pooled_next_mean_error_ratio=val_ratio, next_value_improved_cells=wins,
              information_gate=val_ratio<=.90 and cov_ratio<=.95 and wins>=5,
              wall_s=time.time()-start, peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
              source_sha256=hashlib.sha256((HERE/'double_skew_tree_source_v1.py').read_bytes()).hexdigest(),
              qualification='All7 predeclared rank35 coherent8M cells; exact one-step natural12-term source, no accumulation or fitted coefficient. Existing source-model states and labels share the original covariance acquisition. Finite pair-map response certified against full recomputation. Not final MSE, independent-network replication, metered cost or adjusted score.')
np.savez(ART, **arrays)
OUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2), flush=True)
