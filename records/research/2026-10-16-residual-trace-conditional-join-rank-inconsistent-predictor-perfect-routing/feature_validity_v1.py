"""[opus-condrank] step 1: per-join Nystrom residual trace (exact, offline) on the base runs vs each seed's xMSE gain from the higher rank."""
import glob, json, math, sys
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parent; O = R.parents[2]
def resid(z, L):
    J, w, rows, Y = z[f'L{L}_J'], z[f'L{L}_w'], z[f'L{L}_rows'], z[f'L{L}_Y']
    Qp, core = z[f'L{L}_Qp'], z[f'L{L}_core']
    trG = float(np.sum((J * J) @ w)) + (float(np.sum((Qp @ core) * Qp)) if Qp.size else 0.0)
    S = Y[rows, :]; S = 0.5 * (S + S.T)
    trN = float(np.trace(np.linalg.lstsq(S, Y.T @ Y, rcond=1e-12)[0]))
    return trG - trN, (trG - trN) / trG, trG
def scr(p):
    d = json.loads(Path(p).read_text())['rows'][0]
    with np.load(str(p).replace('.json', '_pred.npz')) as z: return d['flops'], z[str(d['seed'])].astype(np.float64)
def xmse(c, b, s):
    with np.load(O/f'analyst/fresh/truth_seed{s}.npz') as t: A, B = t['mc_post_A'][-1].astype(np.float64), t['mc_post_B'][-1].astype(np.float64)
    U = lambda y: float(np.mean((y[-1] - A) * (y[-1] - B))); (fc, yc), (fb, yb) = scr(c), scr(b)
    return 100 * math.log(U(yc) / U(yb)), 100 * math.log(fc / fb)
def corr(x, y):
    x, y = np.asarray(x), np.asarray(y); rk = lambda v: np.argsort(np.argsort(v))
    return float(np.corrcoef(x, y)[0, 1]), float(np.corrcoef(rk(x), rk(y))[0, 1])
CT = O/'lead/runs/opus_grow_v1/ctl_s2tr'
for arm in ('r416', 'r448'):
    xs, ra, rr = [], [], []
    for s in range(7103248, 7103256):
        f = R/f'cap/jres_s2tr_{s}.npz'; c = O/f'lead/runs/opus_l13rank_v1/{arm}/native_{s}.json'
        if not (f.exists() and c.exists()): continue
        a, r, _ = resid(np.load(f), 13); x, b = xmse(c, CT/f'native_{s}.json', s); xs.append(x); ra.append(a); rr.append(r)
    if len(xs) > 2:
        print(f"L13 {arm} (S2tr, n={len(xs)}): xMSE per seed " + ' '.join(f'{v:+.2f}' for v in xs) + f" | resid rel {min(rr):.2e}..{max(rr):.2e}")
        print(f"   corr(xMSE, resid abs) pearson/spearman {corr(xs, ra)[0]:+.2f}/{corr(xs, ra)[1]:+.2f}   corr(xMSE, resid rel) {corr(xs, rr)[0]:+.2f}/{corr(xs, rr)[1]:+.2f}")
xs, ra, rr = [], [], []
for s in range(7103249, 7103253):
    f = R/f'cap/jres_b2b108_{s}.npz'; c = O/f'lead/runs/opus_lead_screen_v1/r7_448_screen4/native_{s}.json'
    b = next(iter(glob.glob(str(O/f'lead/runs/astra_smirnov_final_native_ten_v1/run_final_parity_s*_v1/native_{s}_ON.json'))), None)
    if not (f.exists() and c.exists() and b): continue
    a, r, _ = resid(np.load(f), 7); x, _b = xmse(c, b, s); xs.append(x); ra.append(a); rr.append(r)
if len(xs) > 2:
    print(f"L7 r448 (2b108, n={len(xs)}): xMSE per seed " + ' '.join(f'{v:+.2f}' for v in xs) + f" | resid rel {min(rr):.2e}..{max(rr):.2e}")
    print(f"   corr(xMSE, resid abs) {corr(xs, ra)[0]:+.2f}/{corr(xs, ra)[1]:+.2f}   corr(xMSE, resid rel) {corr(xs, rr)[0]:+.2f}/{corr(xs, rr)[1]:+.2f}")
