#!/usr/bin/env python3
"""TAG analyst A17 secondary test (2026-10-02): the three relations fixed in a17_preregistered.md (s_new and s_carry gradients NEGATIVELY,
k4a POSITIVELY correlated with colsd_deep = the spread of the column norms of W[8..15]) on gradient networks NOT used in the first pass.
The only such networks on disk are the trainer's older runs (run_A: pilot A, 340 gradients; run_L: 77): the OLD card (no F4, no K4L)
with the star basis held constant in the backward pass. The trainer showed that this backward pass is wrong for bscale, X0, X1, s_carry,
bsrc and k22 (memory p2-detached-star-basis-wrong-dial-gradients); s_new and the k4 band dials were not among the distorted ones. So this
is a weaker test than the one registered (other card, and void for s_carry); it is reported as such.
Gradient = d ln(final-layer MSE) / d dial; one intercept per parameter point (run, update); z-scored inside the point; pooled correlation
with the z-scored feature; t = r sqrt(dof / (1 - r^2)), dof = n - points - 1; one-sided p in the registered direction.
numpy + torch.load only, one thread. usage: a17_retest_oldcard.py -> a17_retest_oldcard.md"""
import os, sys, json, glob, re, time
for k in ('OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import numpy as np, torch
from scipy import stats
from pathlib import Path
torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent; O = HERE.parent; P2 = O.parent; TR = O / 'trainer'; t0 = time.time()
used = set(json.load(open(HERE / 'a17_conditional_dials.json'))['ranks'])
rows = []; seen = set()
for run in ('run_A', 'run_L'):
    for f in sorted(glob.glob(str(TR / run / 'results/g*.pt'))):
        m = re.search(r'g(\d+)_(\d+)_r(\d+)\.pt$', f); upd, rank = int(m.group(1)), int(m.group(3))
        if rank in used or rank in seen: continue
        z = torch.load(f, map_location='cpu', weights_only=True); seen.add(rank)
        rows.append(dict(rank=rank, point=(run, upd), rel={k: float(v) / (float(z['scale']) * float(z['mse'])) for k, v in z['dial_grads'].items()}))
pts = sorted(set(r['point'] for r in rows)); keep = [p for p in pts if sum(1 for r in rows if r['point'] == p) >= 6]; rows = [r for r in rows if r['point'] in keep]
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
cache = HERE / 'a17_colsd_cache.json'; C = json.load(open(cache)) if cache.exists() else {}
for r in rows:
    k = str(r['rank'])
    if k not in C:
        w = np.load(STORE / Path(man[r['rank']]['path']).parent / 'weights.npy', mmap_mode='r'); w2 = np.asarray(w[8:16], dtype=np.float64) ** 2
        C[k] = float(w2.sum(1).std(1).mean())                      # sd over the 1,024 columns of the squared column norm, mean over layers 8-15
json.dump(C, open(cache, 'w'))
N = len(rows); P = np.array([keep.index(r['point']) for r in rows]); x = np.array([C[str(r['rank'])] for r in rows])
def within(v):
    o = np.empty(N)
    for p in range(len(keep)): m = P == p; o[m] = (v[m] - v[m].mean()) / v[m].std()
    return o
xz = np.empty(N)
for p in range(len(keep)): m = P == p; xz[m] = x[m] - x[m].mean()
xz = xz / xz.std(); dof = N - len(keep) - 1
L = []; W = L.append
W(f"# A17 secondary test on the trainer's older runs ({time.strftime('%Y-%m-%d %H:%M', time.gmtime())} UTC)\n")
W(f"{N} networks not used in the first pass ({sum(1 for r in rows if r['point'][0] == 'run_A')} of pilot A, {sum(1 for r in rows if r['point'][0] == 'run_L')} of run L), {len(keep)} parameter points (updates). Old card (no F4, no K4L), star basis held constant in the backward pass: the trainer found that backward pass wrong for bscale, X0, X1, s_carry, bsrc; s_new and the k4 band dials were not among the distorted ones. Feature: spread of the column norms of W in layers 8-15 (mean {x.mean():.5f}, sd {x.std(ddof=1):.5f}).\n")
W('| dial | registered direction | first pass (72 networks) | correlation here | t | one-sided p in the registered direction | two-sided p |\n|---|---|---|---|---|---|---|')
REG = dict(s_new=(-1, '-0.40'), s_carry=(-1, '-0.40'), k4a=(+1, '+0.37')); res = {}
for k in list(REG) + [d for d in rows[0]['rel'] if d not in REG]:
    y = within(np.array([r['rel'][k] for r in rows])); r = float(np.corrcoef(xz, y)[0, 1]); t = r * np.sqrt(dof / (1 - r * r)); p2 = float(2 * stats.t.sf(abs(t), dof))
    if k in REG:
        p1 = float(stats.t.sf(-t, dof) if REG[k][0] < 0 else stats.t.sf(t, dof)); W(f"| {k} | {'negative' if REG[k][0] < 0 else 'positive'} | {REG[k][1]} | {r:+.3f} | {t:+.2f} | {p1:.3f} | {p2:.3f} |")
    else: W(f"| {k} | not registered | | {r:+.3f} | {t:+.2f} | | {p2:.3f} |")
    res[k] = dict(r=r, t=float(t), p_two_sided=p2)
W(f"\nA correlation of 0.40 would give t = {0.40 * np.sqrt(dof / (1 - 0.16)):.1f} here; of 0.15, t = {0.15 * np.sqrt(dof / (1 - 0.0225)):.1f}.")
json.dump(dict(n=N, points=len(keep), results=res), open(HERE / 'a17_retest_oldcard.json', 'w'), indent=1)
(HERE / 'a17_retest_oldcard.md').write_text('\n'.join(L) + '\n'); print('\n'.join(L)); print(round(time.time() - t0), 's')
