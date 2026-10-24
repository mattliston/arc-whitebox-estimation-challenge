"""[opus-rankgb] grade V1/V2 vs GB128 (hunter screen_GB128) per seed: noise-free U = mean((y-A)(y-B)) at L15, plain MSE, bill; net = U + bill."""
import json, math, sys
from pathlib import Path
import numpy as np
W = Path(__file__).resolve().parent; L = W.parent; F = W.parents[2]/'analyst/fresh'; B = L/'opus_joinx_v1/screen_GB128'
def load(d, s):
    j = json.loads((d/f'native_{s}.json').read_text())['rows'][0]; y = np.load(d/f'native_{s}_pred.npz')[str(s)][-1].astype(np.float64); return j, y
out = {}
for v in ('V1_r7_320', 'V2_r13_320'):
    rows = []
    for s in range(7103248, 7103256):
        if not (W/f'screen_{v}'/f'native_{s}.json').exists(): continue
        with np.load(F/f'truth_seed{s}.npz') as z: A, Bh = z['mc_post_A'][-1].astype(np.float64), z['mc_post_B'][-1].astype(np.float64)
        (jb, yb), (jv, yv) = load(B, s), load(W/f'screen_{v}', s)
        U = lambda y: float(np.mean((y - A)*(y - Bh)))
        u = 100*math.log(U(yv)/U(yb)); m = 100*math.log(jv['mse_final']/jb['mse_final']); b = 100*math.log(jv['flops']/jb['flops'])
        rows.append(dict(seed=s, xmse=u, mse=m, bill=b, net=u + b))
    n = len(rows)
    if not n: continue
    st = lambda k: (sum(r[k] for r in rows)/n, (math.sqrt(sum((r[k] - sum(x[k] for x in rows)/n)**2 for r in rows)/(n - 1))/math.sqrt(n)) if n > 1 else float('nan'))
    out[v] = dict(n=n, **{f'{k}_mean': st(k)[0] for k in ('xmse','mse','bill','net')}, **{f'{k}_se': st(k)[1] for k in ('xmse','mse','bill','net')},
                  net_lower=sum(r['net'] < 0 for r in rows), xmse_lower=sum(r['xmse'] < 0 for r in rows), rows=rows,
                  prereg_stack=(st('net')[0] < -0.3))
    print(f"{v}: n={n} xMSE {out[v]['xmse_mean']:+.3f}% (SE {out[v]['xmse_se']:.3f}, {out[v]['xmse_lower']}/{n} lower) plain {out[v]['mse_mean']:+.3f}% bill {out[v]['bill_mean']:+.3f}% "
          f"NET {out[v]['net_mean']:+.3f}% (SE {out[v]['net_se']:.3f}, {out[v]['net_lower']}/{n} lower) -> stack: {out[v]['prereg_stack']}")
    print('   per-seed net:', ' '.join(f"{r['net']:+.2f}" for r in rows))
if len(sys.argv) > 1:
    with open(W/sys.argv[1], 'x') as f: json.dump(out, f, indent=1)
