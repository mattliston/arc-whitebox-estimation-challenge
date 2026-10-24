"""[opus-ladder2] pooled ('aggregate') noise-free metric over the graded seeds: 100 log(sum U_c / sum U_b) + mean bill log; per set S8 / A80.
usage: agg_v1.py ARM"""
import json, math, sys
from pathlib import Path
import numpy as np
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
F = O/'analyst/fresh'; CTL = O/'lead/runs/opus_grow_v1/ctl_s2tr'; CTL80 = O/'lead/runs/opus_retune_v1/ctl_s2tr_a80'; D = Path(__file__).resolve().parent
util = lambda f: max(f / 2**41, .1)
def load(p):
    d = json.loads(Path(p).read_text())['rows'][0]
    with np.load(str(p).replace('.json', '_pred.npz')) as z: y = z[str(d['seed'])].astype(np.float64)
    return d['flops'], y
for arm in sys.argv[1:]:
    sets = {'S8': range(7103248, 7103256), 'A80': range(7103080, 7103096)}
    tot = {}
    for name, seeds in sets.items():
        Uc = Ub = 0.0; bl = []
        for s in seeds:
            p = D/f'screen_{arm}'/f'native_{s}.json'
            if not p.exists(): continue
            (fc, yc), (fb, yb) = load(p), load((CTL if s >= 7103248 else CTL80)/f'native_{s}.json')
            with np.load(F/f'truth_seed{s}.npz') as z: A, B = z['mc_post_A'][-1], z['mc_post_B'][-1]
            Uc += float(np.mean((yc[-1] - A) * (yc[-1] - B))); Ub += float(np.mean((yb[-1] - A) * (yb[-1] - B)))
            bl.append(100 * math.log(util(fc) / util(fb)))
        if bl:
            tot[name] = (Uc, Ub, bl)
            print(f'{arm} {name}: n {len(bl)} aggregate xMSE {100*math.log(Uc/Ub):+.3f} bill {sum(bl)/len(bl):+.3f} -> aggregate xnet {100*math.log(Uc/Ub)+sum(bl)/len(bl):+.3f}')
    if len(tot) == 2:
        Uc = sum(t[0] for t in tot.values()); Ub = sum(t[1] for t in tot.values()); bl = [b for t in tot.values() for b in t[2]]
        print(f'{arm} ALL: n {len(bl)} aggregate xMSE {100*math.log(Uc/Ub):+.3f} bill {sum(bl)/len(bl):+.3f} -> aggregate xnet {100*math.log(Uc/Ub)+sum(bl)/len(bl):+.3f}')
