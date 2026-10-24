"""[opus-acc] Full384+gk4 (4f3cce8f) vs 2b108 on 7103080-7103103: noise-free cross-half U at L15, bill, xnet = U-log + util-log."""
import json, math, sys
from pathlib import Path
import numpy as np
W = Path(__file__).resolve().parent; R = W/'res'; F = W.parents[2]/'analyst/fresh'
seeds = list(range(7103080, 7103104)); rows = []
def load(tag, s):
    d = json.loads((R/f'{tag}_{s}.json').read_text())
    with np.load(R/f'{tag}_{s}_pred.npz') as z: y = z['y'][-1].astype(np.float64)
    return d, y
for s in seeds:
    if not ((R/f'x2b108_{s}.json').exists() and (R/f'xf384gk4_{s}.json').exists()): continue
    with np.load(F/f'truth_seed{s}.npz') as z: A, B = z['mc_post_A'][-1].astype(np.float64), z['mc_post_B'][-1].astype(np.float64)
    (b, yb), (c, yc) = load('x2b108', s), load('xf384gk4', s)
    Ub, Uc = float(np.mean((yb-A)*(yb-B))), float(np.mean((yc-A)*(yc-B)))
    ub, uc = b['flops']/2**41, c['flops']/2**41
    rows.append(dict(seed=s, U_log=100*math.log(Uc/Ub), MSE_log=100*math.log(c['mse_final']/b['mse_final']), bill_log=100*math.log(max(uc,.1)/max(ub,.1)),
                     Ub=Ub, Uc=Uc, util_b=ub, util_c=uc))
for r in rows: r['xnet'] = r['U_log'] + r['bill_log']
n = len(rows)
def ms(k):
    v = [r[k] for r in rows]; m = sum(v)/n; sd = math.sqrt(sum((x-m)**2 for x in v)/(n-1)) if n > 1 else float('nan'); return m, sd/math.sqrt(n)
agg = 100*(sum(r['Uc']*max(r['util_c'], .1) for r in rows)/sum(r['Ub']*max(r['util_b'], .1) for r in rows) - 1)
for r in rows: print(f"{r['seed']} xMSE(U) {r['U_log']:+.3f}  plainMSE {r['MSE_log']:+.3f}  bill {r['bill_log']:+.3f}  xnet {r['xnet']:+.3f}")
out = dict(n=n, **{f'{k}_mean_pct': ms(k)[0] for k in ('U_log','MSE_log','bill_log','xnet')}, **{f'{k}_SE_pct': ms(k)[1] for k in ('U_log','MSE_log','bill_log','xnet')},
           xnet_lower=sum(r['xnet'] < 0 for r in rows), U_lower=sum(r['U_log'] < 0 for r in rows), score_aggregate_pct=agg,
           oos_last8_xnet_mean_pct=(sum(r['xnet'] for r in rows[-8:])/8 if n == 24 else None), rows=rows)
print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, indent=0))
if n == 24 and len(sys.argv) > 1:
    with open(W/sys.argv[1], 'x') as f: json.dump(out, f, indent=1)
