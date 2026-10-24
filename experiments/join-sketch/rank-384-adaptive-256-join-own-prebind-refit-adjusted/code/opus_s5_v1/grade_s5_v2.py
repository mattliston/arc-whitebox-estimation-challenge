"""[opus-s5] S5 (VN+Full384+R256, own refit list) vs S4 (VN+R256, own refit list) per seed: noise-free U at L15, plain MSE, bill (both arms on their own census-fitted prebind lists)."""
import json, math, sys
from pathlib import Path
import numpy as np
W = Path(__file__).resolve().parent; F = W.parents[2]/'analyst/fresh'
def load(d, s):
    j = json.loads((d/f'native_{s}.json').read_text())['rows'][0]; return j, np.load(d/f'native_{s}_pred.npz')[str(s)][-1].astype(np.float64)
rows = []
for s in range(7103248, 7103256):
    if not ((W/'screen_S5pbt'/f'native_{s}.json').exists() and (W/'screen_S4pbt'/f'native_{s}.json').exists()): continue
    with np.load(F/f'truth_seed{s}.npz') as z: A, B = z['mc_post_A'][-1].astype(np.float64), z['mc_post_B'][-1].astype(np.float64)
    (jb, yb), (jv, yv) = load(W/'screen_S4pbt', s), load(W/'screen_S5pbt', s)
    U = lambda y: float(np.mean((y - A)*(y - B)))
    u = 100*math.log(U(yv)/U(yb)); b = 100*math.log(jv['flops']/jb['flops'])
    rows.append(dict(seed=s, xmse=u, mse=100*math.log(jv['mse_final']/jb['mse_final']), bill=b, net=u + b))
n = len(rows); m = lambda k: sum(r[k] for r in rows)/n
se = lambda k: math.sqrt(sum((r[k]-m(k))**2 for r in rows)/(n-1))/math.sqrt(n) if n > 1 else float('nan')
out = dict(n=n, **{f'{k}_mean': m(k) for k in ('xmse','mse','bill','net')}, **{f'{k}_se': se(k) for k in ('xmse','mse','bill','net')},
           xmse_lower=sum(r['xmse'] < 0 for r in rows), net_lower=sum(r['net'] < 0 for r in rows), rows=rows)
print(f"n={n} xMSE {out['xmse_mean']:+.3f}% (SE {out['xmse_se']:.3f}, {out['xmse_lower']}/{n} lower) plain {out['mse_mean']:+.3f}% bill {out['bill_mean']:+.3f}% NET {out['net_mean']:+.3f}% (SE {out['net_se']:.3f}, {out['net_lower']}/{n} lower)")
print('per-seed xMSE:', ' '.join(f"{r['xmse']:+.2f}" for r in rows)); print('per-seed bill:', ' '.join(f"{r['bill']:+.2f}" for r in rows))
if len(sys.argv) > 1:
    with open(W/sys.argv[1], 'x') as f: json.dump(out, f, indent=1)
