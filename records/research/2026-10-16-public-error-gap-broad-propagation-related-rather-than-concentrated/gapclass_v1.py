# [opus-gapclass] Q1 concentration + Q2 proxy correlations per PREREG_gapclass_v1.md
import numpy as np, sys, json
sys.path.insert(0, '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_gapclass_v1')
from load_board_v1 import load
from scipy.stats import spearmanr
d, _ = load()
us = d['us']; L = ['marius', 'j2w', 'oliver']
cons = np.mean([np.log(d[k]['fin']) for k in d], axis=0)
prox_fixed = {'i_our_flops': us['flops'][:50], 'iii_difficulty': cons, 'iv_our_late_birth': np.log(us['fin'] / us['allm'])}
res = {}
for k in L:
    v = d[k]; gap = np.log(us['fin'] / v['fin']); dlt = us['fin'] - v['fin']
    o = np.argsort(-dlt); top10 = dlt[o[:10]].sum() / dlt.sum()
    pr = dict(prox_fixed, ii_leader_flops=v['flops'][:50], v_allm_gap=np.log(us['allm'] / v['allm']))
    rh = {p: round(spearmanr(gap, x)[0], 3) for p, x in pr.items()}
    res[k] = dict(agg_gap_pct=round(100 * np.log(us['fin'].mean() / v['fin'].mean()), 2), mean_log_gap_pct=round(100 * gap.mean(), 2),
                  sd_log_gap_pct=round(100 * gap.std(), 2), top10_share=round(top10, 3), top10_nets=[int(i) for i in o[:10]],
                  frac_nets_we_lose=round(float((dlt > 0).mean()), 2), rho=rh)
    print(k, json.dumps(res[k]))
# leader-vs-leader agreement on where we lose
g = {k: np.log(us['fin'] / d[k]['fin']) for k in L}
print('gap agreement rho m-j %.2f m-o %.2f j-o %.2f' % (spearmanr(g['marius'], g['j2w'])[0], spearmanr(g['marius'], g['oliver'])[0], spearmanr(g['j2w'], g['oliver'])[0]))
print('weak 15/40/16/14 gaps:', {k: [round(100 * g[k][i], 1) for i in (15, 40, 16, 14)] for k in L})
json.dump(res, open(sys.path[0] + '/GAPCLASS_v1.json', 'w'), indent=1)
