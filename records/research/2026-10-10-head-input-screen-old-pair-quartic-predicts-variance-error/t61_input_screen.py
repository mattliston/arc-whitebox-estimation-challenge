#!/usr/bin/env python3
"""TAG opus-physicist T61 (numpy): OPEN-LOOP SCREEN OF NEW HEAD INPUTS ON THE S42 TORCH CARD AGAINST TRUTH (lead 10-02 19:36Z, 10-10 02:11Z).
Dumps: s42dump_rank<r>.npz (t60_s42_dump.py: the trainer's S42 card, parity with his stored number). Truths: per-unit pre-activation v, k3, k4 of layer l from two independent draws
(public_g19_attribution_v1/mps_truth_rank<r>_{P32M,indep32M} for 35 / 100, {P8M,indep4M} for 150-500); post means of every layer from the organizers' 1e9 truth.
TARGETS per unit and layer l (5..15): the card's errors as they act on the layer's own post mean, first order:
   ev = (pdf / (2 sigma)) (v_card - v_true)      es = (k3_card - k3_true) (-alpha pdf / (6 v))      ek = sigma ((k4_card - k4_true) / v^2) (alpha^2 - 1) pdf / 24
   em = the post mean after the learned correction minus the truth (what is left for a new input to fix)
BASELINE regressed out of every target and candidate (per network and layer, least squares with a constant): what the head already sees, its 30 features (normalised as the head
normalises them), its 96 message columns and the S42 column (column 6); option 'state' adds the head's 64 output state numbers (what it made of them).
STATISTIC per candidate x, target y, network, layer: the partial R^2 of y on x after the baseline, with the truth noise taken out by the two draws (cov(x, yA) cov(x, yB) /
(var x cov(yA, yB)); em has one 1e9 truth), and the sign of the partial covariance. Reported per layer band: the mean partial R^2 over networks, how many networks share the sign of
the majority, and the SHARE OF THE POST-MEAN ERROR the candidate could take through that channel = partial R^2 x (energy of the channel's residual) / (energy of em's residual).
usage: t61_input_screen.py [state] [networks, default 35,100,150,200,250,300,400,500]"""
import sys, math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; P2 = HERE.parent.parent
args = [a for a in sys.argv[1:]]; USE_STATE = 'state' in args; NO6 = 'no6' in args; NOMSG = 'nomsg' in args; RESP = 'resp' in args; args = [a for a in args if a not in ('state', 'no6', 'nomsg', 'resp')]
NETS = [int(x) for x in args[0].split(',')] if args else [35, 100, 150, 200, 250, 300, 400, 500]
BANDS = (('5-8', (5, 6, 7, 8)), ('9-11', (9, 10, 11)), ('12-14', (12, 13, 14)), ('15', (15,)))
n = 1024


def truths(r):
    for a, b in (('P32M', 'indep32M'), ('P8M', 'indep4M')):
        pa, pb = P2 / f'public_g19_attribution_v1/mps_truth_rank{r:04d}_{a}.npz', P2 / f'public_g19_attribution_v1/mps_truth_rank{r:04d}_{b}.npz'
        if pa.exists() and pb.exists(): return np.load(pa), np.load(pb)
    return None, None


def resid(Y, B):
    """residuals of the columns of Y after least squares on B (with a constant)"""
    X = np.column_stack([np.ones(len(B)), B]); co, *_ = np.linalg.lstsq(X, Y, rcond=None); return Y - X @ co


RES = {}      # (candidate, target, band) -> list of (r2, sign, share)
BASE = {}     # band -> list of the baseline's in-sample R^2 of each target
for r in NETS:
    f = HERE / f's42dump_rank{r:04d}.npz'
    if not f.exists(): print('missing dump', r); continue
    D = np.load(f); TA, TB = truths(r); tp = np.load(P2 / f'fable_refit_v1/truth688/truthpost_rank{r:04d}.npz')['mc_post'].astype(np.float64)
    for band, layers in BANDS:
        for l in layers:
            if f'v_{l}' not in D.files or f'xn_{l}' not in D.files: continue
            v = D[f'v_{l}']; mu = D[f'mu_{l}']; s = np.sqrt(v); al = mu / s; ph = np.exp(-al * al / 2) / math.sqrt(2 * math.pi)
            k3 = D[f'k3_{l}']; k4 = D[f'k4_{l}']
            Y = {}
            if TA is not None:
                for nm, T in (('A', TA), ('B', TB)):
                    G = lambda key: (T if key in T.files else TA)[key][l].astype(np.float64)      # the 4M draws hold v only: k3, k4 then come from draw A for both (one draw)
                    vt, k3t, k4t = G('mc_pre_v'), G('mc_pre_k3'), G('mc_pre_k4')
                    Y['ev' + nm] = (ph / (2 * s)) * (v - vt); Y['es' + nm] = (k3 - k3t) * (-al * ph / (6 * v)); Y['ek' + nm] = s * ((k4 - k4t) / v ** 2) * (al * al - 1) * ph / 24
            Y['em'] = D[f'm_{l}'] - tp[l]
            Bm = [D[f'xn_{l}']] + ([] if NOMSG else [D[f'msgs_{l}']]) + ([] if NO6 else [D[f'cols_{l}'][:, 6:7]]) + ([D[f'state_out_{l}']] if USE_STATE else [])
            BASE.setdefault(band, []).append({k: 1 - (resid(y[:, None], np.column_stack(Bm))[:, 0] ** 2).sum() / ((y - y.mean()) ** 2).sum() for k, y in Y.items() if k in ('evA', 'esA', 'ekA', 'em')})
            Bm = np.column_stack(Bm).astype(np.float64); keep = Bm.std(0) > 0; Bm = Bm[:, keep]
            cand = {k: D[f'{k}_{l}'] for k in ('ls', 'qb', 'qk', 'kmu', 'kg', 'pv', 'pk3', 'pk4', 'pd', 'pdd') if f'{k}_{l}' in D.files}
            for pre in ('q', 'f', 'c', 'e', 's'):
                for a in ('1', '2', '3', '4', '56', '7'):
                    if f'{pre}{a}_{l}' in D.files: cand[pre + a] = D[f'{pre}{a}_{l}']
            cand['t4'] = k4 / v ** 2; cand['t3'] = k3 / (v * s)
            for k in ('m2', 'k3d2', 'gk', 'gb', 'xk4', 'rg', 'f15'):      # batch 6 (lead 10-10 02:13Z) through b6cols.py, raw (qold, eold, qb there = the screen's own)
                if f'b6_{k}_{l}' in D.files: cand['b6' + k] = D[f'b6_{k}_{l}']
            for pre in ('e', 's', 'q', 'f', 'c'):      # old content pooled: pairs aged 3 and more
                parts = [D[f'{pre}{a}_{l}'] for a in ('3', '4', '56', '7') if f'{pre}{a}_{l}' in D.files]
                if parts: cand[pre + 'old'] = np.sum(parts, 0)
            cols = D[f'cols_{l}']
            for k in range(cols.shape[1]):
                if (k != 6 or NO6) and cols[:, k].std() > 0: cand[f'col{k}'] = cols[:, k]
            if RESP and f'resp_{l}' in D.files:      # option resp: the candidates on the read's response rows only, 0 elsewhere (the native convention of column 6)
                rm = D[f'resp_{l}'] > 0.5; cand = {k: np.where(rm, x, 0.0) for k, x in cand.items() if not k.startswith('col')} | {k: x for k, x in cand.items() if k.startswith('col')}
            names = [k for k, x in cand.items() if np.isfinite(x).all() and x.std() > 0]
            Xc = resid(np.column_stack([cand[k] for k in names]), Bm); Yr = {k: resid(y[:, None], Bm)[:, 0] for k, y in Y.items()}
            emE = (Yr['em'] ** 2).sum()
            for i, nm in enumerate(names):
                x = Xc[:, i]; vx = (x * x).sum()
                if vx <= 0: continue
                for tg in ('ev', 'es', 'ek', 'em'):
                    if tg == 'em':
                        c = (x * Yr['em']).sum(); r2 = c * c / (vx * emE); share = r2; sg = np.sign(c)
                    else:
                        if tg + 'A' not in Yr: continue
                        ya, yb = Yr[tg + 'A'], Yr[tg + 'B']; cab = (ya * yb).sum()
                        if cab <= 0: continue
                        ca, cb = (x * ya).sum(), (x * yb).sum(); r2 = ca * cb / (vx * cab); sg = np.sign(ca + cb); share = r2 * cab / emE
                    RES.setdefault((nm, tg, band), []).append((r2, sg, share, r))
# table
for band, L in BASE.items(): print('baseline in-sample R^2, layers %s: %s' % (band, '  '.join('%s %.3f' % (k, np.mean([d[k] for d in L if k in d])) for k in ('evA', 'esA', 'ekA', 'em'))))
cands = sorted({k[0] for k in RES})
print('S42 torch card, networks %s; baseline = the head\'s 30 features + 96 messages + column 6%s; per band: mean partial R^2 (networks sharing the majority sign / networks), and the share of the post-mean error reachable through that channel' % (NETS, ' + its 64 state numbers' if USE_STATE else ''))
for tg, lab in (('ev', 'VARIANCE error (mean effect)'), ('es', 'SKEW error (mean effect)'), ('ek', 'KURTOSIS error (mean effect)'), ('em', 'POST-MEAN error after the head')):
    rows = []
    for c in cands:
        cells = []; score = 0.0
        for band, _ in BANDS:
            L = RES.get((c, tg, band), [])
            if not L: cells.append('        -           '); continue
            r2 = np.mean([x[0] for x in L]); sg = [x[1] for x in L]; maj = max(sum(1 for z in sg if z > 0), sum(1 for z in sg if z < 0)); sh = np.mean([x[2] for x in L])
            sgn = '+' if sum(1 for z in sg if z > 0) >= sum(1 for z in sg if z < 0) else '-'
            cells.append('%.3f %s(%d/%d) %4.1f%%' % (r2, sgn, maj, len(sg), 100 * sh)); score += sh if band in ('12-14', '15') else 0.0
        rows.append((score, c, cells))
    rows.sort(key=lambda z: -z[0])
    print('\n%s; columns: layers %s' % (lab, ' | '.join(b for b, _ in BANDS)))
    for score, c, cells in rows[:16]: print('  %-7s | %s' % (c, ' | '.join(cells)))
    for score, c, cells in rows:
        if c in ('col6', 'col9', 'col16', 'col17', 'col18', 'col19', 'col20') and (score, c, cells) not in rows[:16]: print('  %-7s | %s   (reference column)' % (c, ' | '.join(cells)))
