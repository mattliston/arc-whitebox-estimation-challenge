"""TAG fable-lead 2026-10-20: injection vs propagation from the PRE dumps alone: the error pushed from layer l to l+1 through a
first-order pair-map transmission (G dCpre_l G, G = diag(gate)) and the exact push W^T . W, compared with the measured dCpre_{l+1}:
residual = the error INJECTED at layer l (pair map + planes + corrections), pushed. Reports its size and its share along
mu_{l+1} mu_{l+1}^T, per layer, nets 35/100."""
import json
import numpy as np
from scipy.stats import norm
K = '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1/'
P2 = '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16'
STORE = P2 + '/learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
MAN = json.load(open(STORE + '/manifest.json'))['roles']['observer_fit']['row_manifests']
def weights(rank):
    p = STORE + '/' + MAN[rank]['path']; p = p[:-len('/row_manifest.json')] if p.endswith('row_manifest.json') else p
    return np.load(p + '/weights.npy')
for rank in (35, 100):
    pre = np.load(K + 'cpre_dump_rank%d_v1.npz' % rank); T = np.load(P2 + '/fable_covhunter_v1/mcall_rank%04d_P32M.npz' % rank); W = weights(rank)
    print('rank', rank)
    for l in range(1, 14):
        Cpre = pre['Cpre_%d' % l].astype(np.float64); mu = pre['mu_%d' % l].astype(np.float64)
        d_l = Cpre - np.asarray(T['cov_pre_%d' % l], dtype=np.float64)
        d_n = pre['Cpre_%d' % (l + 1)].astype(np.float64) - np.asarray(T['cov_pre_%d' % (l + 1)], dtype=np.float64); mun = pre['mu_%d' % (l + 1)].astype(np.float64)
        v = np.diag(Cpre); g = norm.cdf(mu / np.sqrt(np.maximum(v, 1e-12)))
        Wn = np.asarray(W[l + 1], dtype=np.float64)
        prop = Wn.T @ (np.outer(g, g) * d_l) @ Wn
        inj = d_n - prop
        n2 = float(mun @ mun); c = float(mun @ inj @ mun) / n2 ** 2; share = c * c * n2 * n2 / (inj ** 2).sum()
        cp = float(mun @ prop @ mun) / n2 ** 2
        print('L%2d->%2d |d_{l+1}| %.3e  propagated %.3e  injected %.3e (%.2f of |d_{l+1}|)  injected along mu mu^T: coef*|mu|^2/mean v %+.4f share %.3f | propagated coef %+.4f' % (
            l, l + 1, np.sqrt((d_n ** 2).sum()), np.sqrt((prop ** 2).sum()), np.sqrt((inj ** 2).sum()), np.sqrt((inj ** 2).sum() / (d_n ** 2).sum()),
            c * n2 / np.diag(pre['Cpre_%d' % (l + 1)].astype(np.float64)).mean(), share, cp * n2 / np.diag(pre['Cpre_%d' % (l + 1)].astype(np.float64)).mean()), flush=True)
print('DONE')
