#!/usr/bin/env python3
"""TAG opus-reexam E15: is the non-isotropic part of the fourth-cumulant FEED into third-order births made of the exactly known
LOCAL fourth-order coefficients of the hubs born one or two layers earlier?  (open loop, physicist's truth and operators, network 35)
A hub e born at layer j (ReLU of a near-Gaussian z_e with mean mu, sd sigma) leaves a column pair p+- = s_e + t+- e_e in the bank.
Tree level, the post-activation fourth cumulant has, on that hub (s = gated covariance column, entry 0 at e):
   Q[e,a,b,c] = S1 s_a s_b s_c      S1 = d^3 E x / d mu^3   = -alpha pdf / v
   Q[e,e,a,b] = S2 s_a s_b          S2 = d^2 Var x / d mu^2 = 2 (g - g^2 - m d)
   Q[e,e,e,a] = S3 s_a              S3 = d kappa3 x / d mu  = 3 (1 - g)(E x^2 - 2 m^2)
   Q[e,e,e,e] = S4                  S4 = kappa4 x
(the rule already in the F4 candidate carries S1 only; a pair of columns can carry two numbers, the hub needs four).
At a later layer l the pair gives the transported legs  Wt = (pp+ - pp-)/(t+ - t-)  (the hub's unit vector) and
Y = (t+ pp- - t- pp+)/(t+ - t-)  (the covariance column), and the feed tensor into the births of layer l with hub k is
   (d_k/2) g_i g_j Q_ijkk = (gY)(gY) h1_k + [(gY)(gWt) + (gWt)(gY)] h2_k + (gWt)(gWt) h3_k
   h1 = (d/2)(2 S1 Y Wt + S2 Wt^2),  h2 = (d/2)(S1 Y^2 + 2 S2 Y Wt + S3 Wt^2),  h3 = (d/2)(S2 Y^2 + 2 S3 Y Wt + S4 Wt^2).
Scored exactly as the physicist's rule screen: share of the true feed's energy on the kink block of layer l+1 (pushed distinct
content), coefficients fitted on truth block 0 and scored on block 1 and the reverse.
Also: RIDE = the part of (h1, h2, h3) that two in-place column shifts can carry, (h1,h2,h3) = d+ (1,t+,t+^2) - d- (1,t-,t-^2),
least squares per hub in the metric of the three matrix types.
usage: e15_local_feed.py <rank> <layer> [threads=2]"""
import sys, json, math, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; PH = HERE / 'physicist'; P2 = HERE.parent
sys.path.insert(0, str(PH))
import numpy as np, torch
import t2_lib as L
from t4_lib import Op, T
torch.set_grad_enabled(False)
rank = int(sys.argv[1]); l = int(sys.argv[2]); torch.set_num_threads(int(sys.argv[3]) if len(sys.argv) > 3 else 2)
BP = np.load(PH / f't2_budget_rank{rank:04d}_planes.npz'); BI = np.load(PH / f't4births_rank{rank:04d}.npz')
MC = np.load(P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz')
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests'][rank]
W = T(np.load((STORE / man['path']).parent / 'weights.npy', mmap_mode='r')[l + 1])
COL = np.load(PH / f't4cols_rank{rank:04d}_L{l}.npz'); pp = T(COL['pp']); lam3 = COL['lam'].astype(np.float64); alive = COL['alive'].astype(np.float64)
stops = [0] + [int(x) for x in COL['stops']]
mu = MC[f'mean_pre_{l}'].astype(np.float64); C = MC[f'cov_pre_{l}'].astype(np.float64); v = np.diag(C).copy(); sd = np.sqrt(v); al = mu / sd
pdf = np.exp(-al * al / 2) / math.sqrt(2 * math.pi); g = np.array([math.erfc(-a / math.sqrt(2)) / 2 for a in al]); d = pdf / sd
Gm = T(alive * g)[:, None]; dd = T(d / 2)[:, None]
blk = BP[f'blk_L{l}']; F = [BP[f't_F4_L{l}_b{b}'].astype(np.float64)[blk] for b in (0, 1)]; FF = float(F[0] @ F[1])
Fiso = (L.feed_push_model(W.numpy(), d, g, C, 1.0, 0.0) - L.Pi(W.numpy(), L.feed_plane_model(d, g, C, 1.0, 0.0)))[blk]
t0 = time.time()


def hub_data(j):
    b = lambda k: BI[f'{k}_{j}']
    sel = b('sel'); a, s, gt, ph, m = (b(k)[sel].astype(np.float64) for k in ('alpha', 'sigma', 'gate', 'pdf', 'host'))
    tp, tm, coef = (b(k)[sel].astype(np.float64) for k in ('plus', 'minus', 'coef'))
    vv = s * s; dj = ph / s
    r2 = vv * ((1 + a * a) * gt + a * ph); r3 = vv * s * ((a ** 3 + 3 * a) * gt + (a * a + 2) * ph); r4 = vv * vv * ((a ** 4 + 6 * a * a + 3) * gt + (a ** 3 + 5 * a) * ph)
    S1 = -a * ph / vv; S2 = 2 * (gt - gt * gt - m * dj); S3 = 3 * (1 - gt) * (r2 - 2 * m * m)
    S4 = r4 - 4 * m * r3 + 6 * m * m * r2 - 3 * m ** 4 - 3 * (r2 - m * m) ** 2
    return tp, tm, coef, (S1, S2, S3, S4), a3_over_birth(a, ph, vv, coef, tp - tm)


def a3_over_birth(a, ph, vv, coef, gap):
    return (-a * ph / vv) / (coef * gap)


AGES = {'age1': (1, 1), 'age2': (2, 2), 'age34': (3, 4), 'age5p': (5, 99)}
planes = {}; ruleA = np.zeros(pp.shape[1]); info = {}
for name, (lo, hi) in AGES.items():
    Ys, Ws, tps, tms, Ss = [], [], [], [], [[], [], [], []]
    for j in range(l):
        if not lo <= l - j <= hi: continue
        tp, tm, coef, S, A = hub_data(j); k0 = stops[j]; K = (stops[j + 1] - k0) // 2; assert K == len(tp), (j, K, len(tp))
        pplus = pp[:, k0:k0 + K].to(torch.float64); pminus = pp[:, k0 + K:k0 + 2 * K].to(torch.float64); gap = T(tp - tm)[None, :]
        Ws.append((pplus - pminus) / gap); Ys.append((T(tp)[None, :] * pminus - T(tm)[None, :] * pplus) / gap); tps.append(tp); tms.append(tm)
        for q in range(4): Ss[q].append(S[q])
        ruleA[k0:k0 + K] = A; ruleA[k0 + K:k0 + 2 * K] = A
    if not Ys: continue
    Y = torch.cat(Ys, 1); Wt = torch.cat(Ws, 1); tp = np.concatenate(tps); tm = np.concatenate(tms); S1, S2, S3, S4 = (T(np.concatenate(x))[None, :] for x in Ss)
    a = Gm * Y; b = Gm * Wt; K = Y.shape[1]; one = torch.ones(K, dtype=torch.float64)
    yy, yw, ww = dd * Y * Y, dd * Y * Wt, dd * Wt * Wt
    def push(h1, h2, h3):
        """pushed distinct content of sum_e Sym(a,a,h1) + Sym-cross(a,b,h2) + Sym(b,b,h3)."""
        u = torch.cat((a, a + b, a - b, b), 1); om = torch.cat((h1, h2, h2, h3), 1); lam = torch.cat((one, one / 2, -one / 2, one))
        return Op(W, u, om).fwd(lam).numpy()[blk]
    z = torch.zeros_like(yy)
    H = {'S1': (2 * S1 * yw, S1 * yy, z), 'S2': (S2 * ww, 2 * S2 * yw, S2 * yy), 'S3': (z, S3 * ww, 2 * S3 * yw), 'S4': (z, z, S4 * ww)}
    for k_, h in H.items(): planes[f'{k_}_{name}'] = push(*h)
    # the part two in-place shifts can carry: per hub, projection of (h1,h2,h3) on span{(1,t+,t+^2), (1,t-,t-^2)} in the metric of the matrix types
    aa = (a * a).sum(0); bb = (b * b).sum(0); ab = (a * b).sum(0)
    Gam = torch.stack([torch.stack([aa * aa, 2 * aa * ab, ab * ab]), torch.stack([2 * aa * ab, 2 * (aa * bb + ab * ab), 2 * bb * ab]), torch.stack([ab * ab, 2 * bb * ab, bb * bb])]).permute(2, 0, 1)
    B = torch.stack([torch.stack([one, -one]), torch.stack([T(tp), -T(tm)]), torch.stack([T(tp) ** 2, -T(tm) ** 2])]).permute(2, 0, 1)       # K x 3 x 2
    BtG = B.transpose(1, 2) @ Gam; Pm = B @ torch.linalg.solve(BtG @ B, BtG)                                                                # K x 3 x 3
    htot = [sum(H[k_][q] for k_ in H) for q in range(3)]
    hr = [sum(Pm[:, q, r][None, :] * htot[r] for r in range(3)) for q in range(3)]
    planes[f'ALL_{name}'] = push(*htot); planes[f'RIDE_{name}'] = push(*hr)
    hS1 = H['S1']; hr1 = [sum(Pm[:, q, r][None, :] * hS1[r] for r in range(3)) for q in range(3)]; planes[f'RIDES1_{name}'] = push(*hr1)
    info[name] = dict(pairs=int(K), norm_a=float(aa.sqrt().median()), norm_b=float(bb.sqrt().median()), cos_ab=float((ab / (aa * bb).sqrt()).median()),
                      S_med_abs=[float(x.abs().median()) for x in (S1, S2, S3, S4)], wall=round(time.time() - t0, 1))
    print(json.dumps({name: info[name]}), flush=True)
opA = Op(W, Gm * pp.to(torch.float64), dd * pp.to(torch.float64) ** 2); planes['ruleA'] = opA.fwd(T(lam3 * ruleA)).numpy()[blk]


def cv(keys, fixed=()):
    """iso + keys with fitted coefficients, + fixed keys at coefficient 1 (subtracted from the target before the fit)."""
    base = sum(planes[k] for k in fixed) if fixed else 0.0
    X = np.stack([Fiso] + [planes[k] for k in keys], 1)
    co = [np.linalg.lstsq(X, F[b_] - base, rcond=None)[0] for b_ in (0, 1)]
    fit = [X @ co[b_] + base for b_ in (0, 1)]
    sc = np.mean([(2 * float(fit[b_] @ F[1 - b_]) - float(fit[b_] @ fit[b_])) / FF for b_ in (0, 1)])
    return round(float(sc), 4), [round(float(x), 4) for x in (co[0] + co[1]) / 2]


names = [n for n in AGES if f'ALL_{n}' in planes]; S = ('S1', 'S2', 'S3', 'S4')
out = dict(TAG='opus-reexam E15 local fourth-order feed', rank=rank, layer=l, kink_block=int(blk.sum()),
           norms_pct_of_feed={k: round(100 * float(np.linalg.norm(p_)) / math.sqrt(FF), 1) for k, p_ in planes.items()},
           cos_with_feed={k: round((float(p_ @ F[0]) + float(p_ @ F[1])) / 2 / max(float(np.linalg.norm(p_)) * math.sqrt(FF), 1e-300), 3) for k, p_ in planes.items()})
print(json.dumps(out), flush=True)
res = {}
res['iso'] = cv([])
res['iso+ruleA(fit)'] = cv(['ruleA'])
for upto in range(1, len(names) + 1):
    ns = names[:upto]; tag = '+'.join(ns)
    res[f'iso + physical ALL at 1 [{tag}]'] = cv([], fixed=[f'ALL_{n}' for n in ns])
    res[f'iso + ALL fitted per age [{tag}]'] = cv([f'ALL_{n}' for n in ns])
    res[f'iso + RIDE at 1 [{tag}]'] = cv([], fixed=[f'RIDE_{n}' for n in ns])
    res[f'iso + RIDE fitted per age [{tag}]'] = cv([f'RIDE_{n}' for n in ns])
    res[f'iso + S1 only fitted per age [{tag}]'] = cv([f'S1_{n}' for n in ns])
    res[f'iso + RIDE of S1 only fitted per age [{tag}]'] = cv([f'RIDES1_{n}' for n in ns])
res['iso + S1..S4 fitted separately, age1'] = cv([f'{s}_age1' for s in S])
res['iso + S1..S4 fitted separately, all ages'] = cv([f'{s}_{n}' for n in names for s in S])
for s in S: res[f'iso + {s} fitted per age, all ages'] = cv([f'{s}_{n}' for n in names])
res['iso + ruleA + ALL fitted per age'] = cv(['ruleA'] + [f'ALL_{n}' for n in names])
for k_, v_ in res.items(): print(json.dumps({k_: v_}), flush=True)
print(json.dumps(dict(done=True, wall=round(time.time() - t0, 1))), flush=True)
