#!/usr/bin/env python3
"""TAG analyst A20 tables: the error budget of the S40 torch card before (candidate 3) and after (candidate 3 + the pair-carried law),
from the files of a20_budget.py. All numbers are per network; 'share' = the part of the squared final-layer error (or of the squared
variance error) that disappears to first order when that input is made exact: -(change . error) / |error|^2, in %.
Truth noise: the variance arms use two independent 32M draws (A, B); squared sizes use A x B. Shares of the variance error are projected on
the error measured with the INDEPENDENT draw B (the planes and the 32M covariance may share samples with draw A).
usage: a20_table.py <tag> <rank> [<rank> ...] [<before suffix> <after suffix>]   -> text on stdout, a20_table_<tag>[_<before>_<after>].json
  the suffixes name the two cards compared (files a20_budget_rank<rank>_<tag><suffix>[p3|p4].npz); default c3 law; e.g. law s42."""
import json, math, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
tag = sys.argv[1]; ranks = [int(x) for x in sys.argv[2:] if x.isdigit()]; N = 1024
XB, XA = ([x for x in sys.argv[2:] if not x.isdigit()] + ['c3', 'law'])[:2] if len([x for x in sys.argv[2:] if not x.isdigit()]) in (0, 2) else sys.exit('give two card suffixes or none')
CARD = dict(c3='candidate 3', law='candidate 3 + the pair law = the S40 torch card', s42="the S42 torch card (trainer's second dial solve: law with the constant regressor, K31L4 ages 1-3, star by row sums)")
WHAT = "the law's" if (XB, XA) == ('c3', 'law') else "the step's"
BANDS = [('1-8', ['B1-8']), ('9-11', ['L9', 'L10', 'L11']), ('12', ['L12']), ('13', ['L13']), ('14', ['L14']), ('15', ['L15'])]
ARMS = ('C', 'mu', 'K21k', 'K21o', 'K31', 'K22', 'k3', 'k4', 'PL')
NAMES = dict(C='covariance', mu='mean', K21k='K21 kink rows', K21o='K21 other rows', K31='K31', K22='K22', k3='per-unit k3', k4='per-unit k4', PL='K21+K31+K22')
res = {}


def part1(Z):
    t, base = Z['truth'], Z['base']; e0 = base - t; E = float(e0 @ e0); r = dict(mse=E / N * 1e9, bands={})
    for band, names in BANDS:
        cA = sum(Z[f'{n}_v_A'] - base for n in names); cB = sum(Z[f'{n}_v_B'] - base for n in names)
        cs = sum(Z[f'{n}_s_AB'] - base for n in names); ck = sum(Z[f'{n}_k_AB'] - base for n in names); cv = (cA + cB) / 2
        r['bands'][band] = dict(v=-100 * float(cv @ e0) / E, s=-100 * float(cs @ e0) / E, k=-100 * float(ck @ e0) / E)
    eA, eB = Z['all_v_A'] - t, Z['all_v_B'] - t; r['all_v'] = float(eA @ eB) / N * 1e9
    r['all_s'] = float(((Z['all_s_AB'] - t) ** 2).mean()) * 1e9; r['all_k'] = float(((Z['all_k_AB'] - t) ** 2).mean()) * 1e9
    eA, eB = Z['all_vsk_A'] - t, Z['all_vsk_B'] - t; r['all_vsk'] = float(eA @ eB) / N * 1e9
    cs = [sum(Z[f'{n}_{a}'] - base for _, names in BANDS for n in names) for a in ('v_A', 'v_B', 's_AB', 'k_AB')]
    S = (cs[0] + cs[1]) / 2 + cs[2] + cs[3]; a = (Z['all_vsk_A'] + Z['all_vsk_B']) / 2 - base
    r['superposition'] = dict(cosine=float(S @ a) / float(np.linalg.norm(S) * np.linalg.norm(a)), size_ratio=float(np.linalg.norm(S) / np.linalg.norm(a)))
    # the card's inputs against truth, per layer
    vc, vA, vB = Z['card_v'], Z['true_v_A'], Z['true_v_B']; vt = (vA + vB) / 2; mu = Z['card_mu']
    al = mu / np.sqrt(vc); wgt = np.exp(-al * al / 2) / math.sqrt(2 * math.pi) / (2 * np.sqrt(vc))
    r['layers'] = {}
    for l in range(1, 16):
        xa, xb = (vc[l] - vA[l]) / vA[l], (vc[l] - vB[l]) / vB[l]
        g1c, g1t = Z['card_k3'][l] / vc[l] ** 1.5, Z['true_k3'][l] / vt[l] ** 1.5; g2c, g2t = Z['card_k4'][l] / vc[l] ** 2, Z['true_k4'][l] / vt[l] ** 2
        sg = np.sqrt(vc[l]); ph = np.exp(-al[l] ** 2 / 2) / math.sqrt(2 * math.pi)                         # first-order mean change: dv phi/(2 sigma); dg1 (-sigma alpha phi/6); dg2 sigma (alpha^2-1) phi/24
        r['layers'][l] = dict(v_rel_rms_pct=100 * math.sqrt(max(float(xa @ xb), 0) / N), v_mean_pct=100 * float((xa + xb).mean() / 2),
                              v_first_order_mse=1e9 * float((wgt[l] ** 2 * (vc[l] - vA[l]) * (vc[l] - vB[l])).mean()),
                              s_first_order_mse=1e9 * float(((sg * al[l] * ph / 6) ** 2 * (g1c - g1t) ** 2).mean()),
                              k_first_order_mse=1e9 * float(((sg * (al[l] ** 2 - 1) * ph / 24) ** 2 * (g2c - g2t) ** 2).mean()),
                              layer_mse=1e9 * float(((Z['pred_all'][l] - Z['truth_all'][l]) ** 2).mean()) if 'truth_all' in Z.files else float('nan'),
                              skew_rms=float(np.sqrt(((g1c - g1t) ** 2).mean())), kurt_rms=float(np.sqrt(((g2c - g2t) ** 2).mean())))
    return r


def part2(Z):
    t, base = Z['truth'], Z['base']; e0 = base - t; E = float(e0 @ e0); r = {}
    for l in (11, 12, 13, 14):
        if f'pm{l}_ev_A' not in Z.files: continue
        evA, evB = Z[f'pm{l}_ev_A'], Z[f'pm{l}_ev_B']; den = float(evA @ evB); row = dict(variance_error_rel_rms_pct=100 * math.sqrt(max(den, 0) / N), arms={})
        w, vA, vB = sens(Z, l + 1), Z['true_v_A'][l + 1], Z['true_v_B'][l + 1]; mA, mB = w * vA * evA, w * vB * evB; denw = float(mA @ mB)   # first-order error of the next layer's mean
        row['mean_error_from_variance_x1e9'] = 1e9 * denw / N
        for X in ARMS:
            dv = Z[f'pm{l}_{X}_dv']; c = Z[f'pm{l}_{X}'] - base; dm = w * dv * (vA + vB) / 2
            row['arms'][X] = dict(var_share=-100 * float(dv @ evB) / den, var_share_A=-100 * float(dv @ evA) / den, dv_rms_pct=100 * float(np.sqrt((dv ** 2).mean())),
                                  w_share=-100 * float(dm @ mB) / denw, w_share_A=-100 * float(dm @ mA) / denw,
                                  final_share=-100 * float(c @ e0) / E, final_alone=100 * float(2 * (c @ e0) + (c @ c)) / E)
        r[l] = row
    return r


P3 = ('K21', 'K21k', 'K21o', 'K31', 'K22', 'PL')


def sens(Z, l):
    """d(mean of layer l) / d(variance of layer l) per unit, Gaussian leading term: pdf(mu/sigma) / (2 sigma), at the card's own mean and variance."""
    al = Z['card_mu'][l] / np.sqrt(Z['card_v'][l]); return np.exp(-al * al / 2) / math.sqrt(2 * math.pi) / (2 * np.sqrt(Z['card_v'][l]))


def part3(Z):
    t, base = Z['truth'], Z['base']; e0 = base - t; E = float(e0 @ e0); vc, vA, vB = Z['card_v'], Z['true_v_A'], Z['true_v_B']; r = {}
    for X in P3:
        if f'allpm_{X}' not in Z.files: continue
        c = Z[f'allpm_{X}'] - base; row = dict(final_share=-100 * float(c @ e0) / E, mse=float(((Z[f'allpm_{X}'] - t) ** 2).mean()) * 1e9, size=100 * float(c @ c) / E, var_share={}, w_share={})
        if f'allpm_{X}_v' in Z.files:
            va = Z[f'allpm_{X}_v']                                              # layers 3..15
            for l in (9, 10, 11, 12, 13, 14, 15):
                evA, evB = (vc[l] - vA[l]) / vA[l], (vc[l] - vB[l]) / vB[l]; dv = (va[l - 3] - vc[l]) / ((vA[l] + vB[l]) / 2)
                row['var_share'][l] = -100 * float(dv @ evB) / float(evA @ evB)
                w = sens(Z, l); dm, mA, mB = w * (va[l - 3] - vc[l]), w * (vc[l] - vA[l]), w * (vc[l] - vB[l]); row['w_share'][l] = -100 * float(dm @ mB) / float(mA @ mB)
        r[X] = row
    return r


for rank in ranks:
    Zb, Za = [np.load(HERE / f'a20_budget_rank{rank:04d}_{tag}{x}.npz') for x in (XB, XA)]
    Z3 = [np.load(q) if q.exists() else Z for q, Z in ((HERE / f'a20_budget_rank{rank:04d}_{tag}{x}p3.npz', Z) for x, Z in ((XB, Zb), (XA, Za)))]
    assert all(float(np.abs(Z3[k]['base'] - (Zb, Za)[k]['base']).max()) == 0.0 for k in (0, 1))        # the add-on file is the same card run
    b, a = part1(Zb), part1(Za); res[rank] = dict(before=b, after=a, pm_before=part2(Zb), pm_after=part2(Za), p3_before=part3(Z3[0]), p3_after=part3(Z3[1]))
    print(f"\n================ network {rank}: final-layer MSE {b['mse']:.3f}e-9 before ({CARD.get(XB, XB)}), {a['mse']:.3f}e-9 after ({CARD.get(XA, XA)}): {100 * (a['mse'] / b['mse'] - 1):+.2f}%")
    print("PART 1. Mean formula inputs made exact (learned correction on). Share of the squared final error, % of that card's own error | the same in 1e-9 of final-layer MSE")
    print(" band  |  before: variance  skew  kurtosis   sum |  after: variance  skew  kurtosis   sum || 1e-9: before v / s / k  ->  after v / s / k   (change of the sum)")
    tb = np.zeros(3); ta = np.zeros(3)
    for band, _ in BANDS:
        x, y = b['bands'][band], a['bands'][band]; xb = np.array([x['v'], x['s'], x['k']]); ya = np.array([y['v'], y['s'], y['k']]); tb += xb; ta += ya
        mb, ma = xb * b['mse'] / 100, ya * a['mse'] / 100
        print(f" {band:>5} |        {xb[0]:+7.1f} {xb[1]:+6.1f} {xb[2]:+7.1f} {xb.sum():+7.1f} |       {ya[0]:+7.1f} {ya[1]:+6.1f} {ya[2]:+7.1f} {ya.sum():+7.1f} ||"
              f"   {mb[0]:+6.2f} {mb[1]:+6.2f} {mb[2]:+6.2f}  ->  {ma[0]:+6.2f} {ma[1]:+6.2f} {ma[2]:+6.2f}   ({ma.sum() - mb.sum():+.2f})")
    mb, ma = tb * b['mse'] / 100, ta * a['mse'] / 100
    print(f"   all |        {tb[0]:+7.1f} {tb[1]:+6.1f} {tb[2]:+7.1f} {tb.sum():+7.1f} |       {ta[0]:+7.1f} {ta[1]:+6.1f} {ta[2]:+7.1f} {ta.sum():+7.1f} ||"
          f"   {mb[0]:+6.2f} {mb[1]:+6.2f} {mb[2]:+6.2f}  ->  {ma[0]:+6.2f} {ma[1]:+6.2f} {ma[2]:+6.2f}   ({ma.sum() - mb.sum():+.2f})")
    print(f" not in the three inputs (formula, learned correction, cross terms): before {100 - tb.sum():+.1f}% = {(100 - tb.sum()) * b['mse'] / 100:.2f}e-9, after {100 - ta.sum():+.1f}% = {(100 - ta.sum()) * a['mse'] / 100:.2f}e-9")
    for key, label in (('all_v', 'variance exact at every layer'), ('all_s', 'skew exact at every layer'), ('all_k', 'kurtosis exact at every layer'), ('all_vsk', 'all three exact at every layer')):
        print(f"   final-layer MSE with {label:32s}: before {b[key]:7.3f}e-9 ({100 * (b[key] / b['mse'] - 1):+6.1f}%)   after {a[key]:7.3f}e-9 ({100 * (a[key] / a['mse'] - 1):+6.1f}%)")
    print(f"   superposition (sum of the band arms against the every-layer arm): before cosine {b['superposition']['cosine']:.3f} size ratio {b['superposition']['size_ratio']:.3f}; after {a['superposition']['cosine']:.3f} / {a['superposition']['size_ratio']:.3f}")
    print(f"   in sequence: the variance channel (base minus variance exact everywhere) before {b['mse'] - b['all_v']:.2f}e-9, after {a['mse'] - a['all_v']:.2f}e-9;"
          f" skew and kurtosis once the variance is exact (variance exact minus all three exact) before {b['all_v'] - b['all_vsk']:.2f}e-9, after {a['all_v'] - a['all_vsk']:.2f}e-9;"
          f" left with all three exact before {b['all_vsk']:.2f}e-9, after {a['all_vsk']:.2f}e-9")
    tr = Zb['truth']; cl = Za['base'] - Zb['base']; e0b = Zb['base'] - tr; avA, avB = Zb['all_v_A'] - Zb['base'], Zb['all_v_B'] - Zb['base']; av = (avA + avB) / 2
    print(f"   {WHAT} change of the final prediction: size {100 * float(cl @ cl) / float(e0b @ e0b):.1f}% of the squared error before, cosine with the error before {float(cl @ e0b) / math.sqrt(float(cl @ cl) * float(e0b @ e0b)):+.3f};"
          f" against the correction that exact variance at every layer makes: cosine {float(cl @ av) / math.sqrt(float(cl @ cl) * max(float(avA @ avB), 1e-300)):+.3f}, coefficient {float(cl @ av) / float(avA @ avB):+.3f}")
    print("   the card's per-unit inputs against truth, layers 9-15. First-order MSE that the input's error puts into THAT layer's mean, 1e-9 (variance / skew / kurtosis), and the variance error: relative rms %, mean %")
    for l in range(9, 16):
        x, y = b['layers'][l], a['layers'][l]
        print(f"     layer {l:2d}: before {x['v_first_order_mse']:5.2f} / {x['s_first_order_mse']:5.2f} / {x['k_first_order_mse']:5.2f}, variance {x['v_rel_rms_pct']:.4f} rms {x['v_mean_pct']:+.4f} mean"
              f"    after {y['v_first_order_mse']:5.2f} / {y['s_first_order_mse']:5.2f} / {y['k_first_order_mse']:5.2f}, variance {y['v_rel_rms_pct']:.4f} rms {y['v_mean_pct']:+.4f} mean")
    vcb, vca, vA, vB = Zb['card_v'], Za['card_v'], Zb['true_v_A'], Zb['true_v_B']
    print(f"   {WHAT} change of the per-unit variance (after minus before, relative): rms %, and the share of the squared variance error before that it removes (projection on the independent truth draw)")
    print("     " + '   '.join(f"L{l}: {100 * float(np.sqrt((((vca[l] - vcb[l]) / vB[l]) ** 2).mean())):.4f} / {-100 * float(((vca[l] - vcb[l]) / vB[l]) @ ((vcb[l] - vB[l]) / vB[l])) / float(((vcb[l] - vA[l]) / vA[l]) @ ((vcb[l] - vB[l]) / vB[l])):+.1f}%" for l in range(9, 16)))
    pb, pa = res[rank]['pm_before'], res[rank]['pm_after']
    print("PART 2. Pair map of layer l: ONE input made exact at that layer, the others the card's. 'variance error' = the error of the next layer's per-unit variance weighted by what it does to that layer's mean (pdf / 2 sigma).")
    print("   First number: share of that squared error (%); second: share of the squared final-layer error (%), the change carried to the end in closed loop. [..] = that error as MSE of the next layer's mean, 1e-9")
    print(" pair layer -> variance of layer | " + ' | '.join(NAMES[X] for X in ARMS))
    for l in (11, 12, 13, 14):
        if l not in pb or l not in pa: continue
        for name, p in (('before', pb), ('after ', pa)):
            print(f"   {l:2d} -> {l + 1:2d} {name} [{p[l]['mean_error_from_variance_x1e9']:.2f}] | " + ' | '.join(f"{p[l]['arms'][X]['w_share']:+6.1f} / {p[l]['arms'][X]['final_share']:+5.2f}" for X in ARMS))
    print("   the same in absolute terms: the part of that MSE (1e-9) carried by each input [share x the layer's error]")
    for l in (11, 12, 13, 14):
        if l not in pb or l not in pa: continue
        for name, p in (('before', pb), ('after ', pa)):
            q = p[l]['mean_error_from_variance_x1e9'] / 100
            print(f"   {l:2d} -> {l + 1:2d} {name} | total {q * 100:5.2f} | " + ' | '.join(f"{NAMES[X]} {p[l]['arms'][X]['w_share'] * q:+5.2f}" for X in ARMS))
    print("   unweighted (every unit counts the same; relative variance error rms % in brackets): share of the next layer's squared variance error (%)")
    for l in (11, 12, 13, 14):
        if l not in pb or l not in pa: continue
        for name, p in (('before', pb), ('after ', pa)):
            print(f"   {l:2d} -> {l + 1:2d} {name} [{p[l]['variance_error_rel_rms_pct']:.4f}] | " + ' | '.join(f"{p[l]['arms'][X]['var_share']:+6.1f}" for X in ARMS))
    print("PART 3. One plane exact at EVERY layer 2-14 (8M-sample planes: their noise enters at every layer, so the MSE is a pessimistic bound and the share is the cleaner number)")
    qb, qa = res[rank]['p3_before'], res[rank]['p3_after']
    for X in P3:
        if X in qb and X in qa:
            print(f"   {NAMES.get(X, 'K21 all rows'):14s}: share of the squared final error before {qb[X]['final_share']:+6.1f}% (MSE {qb[X]['mse']:.3f}e-9), after {qa[X]['final_share']:+6.1f}% (MSE {qa[X]['mse']:.3f}e-9)"
                  + (" | share of the squared variance error (weighted as in part 2) at layers 12 / 13 / 14 / 15: before " + ' / '.join(f"{qb[X]['w_share'][l]:+.0f}" for l in (12, 13, 14, 15)) if qb[X]['w_share'] else '')
                  + ("   after " + ' / '.join(f"{qa[X]['w_share'][l]:+.0f}" for l in (12, 13, 14, 15)) if qa[X]['w_share'] else ''))
    # PART 4: planes exact on bands of pair layers; second plane draw at layer 12
    print("PART 4. One plane exact on a BAND of pair layers (closed loop, 8M planes): first-order share of the squared final error / realized change of the final MSE / squared size of the change, all in % of that card's squared error")
    res[rank]['bands'] = {}
    for label, x, Zm in (('before', XB, Zb), ('after ', XA, Za)):
        q = HERE / f'a20_budget_rank{rank:04d}_{tag}{x}p4.npz'
        if not q.exists(): print(f"   {label}: no band file yet"); continue
        Z4 = np.load(q); assert float(np.abs(Z4['base'] - Zm['base']).max()) == 0.0
        t4, base4 = Z4['truth'], Z4['base']; e4 = base4 - t4; E4 = float(e4 @ e4); vc, vA, vB = Z4['card_v'], Z4['true_v_A'], Z4['true_v_B']; rr = {}
        for X in ('K31', 'K21k', 'PL'):
            cells = []
            for band in ('11-14', '8-10', '2-7'):
                if f'band{band}_{X}' not in Z4.files: continue
                c = Z4[f'band{band}_{X}'] - base4; sh, sz = -100 * float(c @ e4) / E4, 100 * float(c @ c) / E4; va = Z4[f'band{band}_{X}_v']; ws = []
                for i, l in enumerate((12, 13, 14, 15)):
                    w = sens(Z4, l); dm, mA, mB = w * (va[i] - vc[l]), w * (vc[l] - vA[l]), w * (vc[l] - vB[l]); ws.append(-100 * float(dm @ mB) / float(mA @ mB))
                rr[f'{X}_{band}'] = dict(share=sh, realized=sz - 2 * sh, size=sz, w_share=ws)
                cells.append(f"layers {band}: {sh:+5.1f} / {sz - 2 * sh:+5.1f} / {sz:4.1f} (variance error at 12-15: " + ' '.join(f"{y:+.0f}" for y in ws) + ")")
            print(f"   {label} {NAMES.get(X, X):14s} " + '   '.join(cells))
        if 'pm12n2_K31' in Z4.files:
            print(f"   {label}, pair layer 12 alone with TWO independent 8M draws of the planes: share (mean of the two) / realized with one draw / realized free of plane noise (size from the product of the two draws) / the noise's part of the size")
            for X in ('K31', 'K21k', 'K21o', 'K22', 'PL'):
                c1, c2 = Zm[f'pm12_{X}'] - base4, Z4[f'pm12n2_{X}'] - base4; sh = -100 * float((c1 + c2) @ e4) / 2 / E4; s1 = 100 * float(c1 @ c1) / E4; s2 = 100 * float(c2 @ c2) / E4; sx = 100 * float(c1 @ c2) / E4
                rr[f'{X}_12_two_draws'] = dict(share=sh, realized_one=(s1 + s2) / 2 - 2 * sh, realized_noise_free=sx - 2 * sh, noise_size=(s1 + s2) / 2 - sx)
                print(f"      {NAMES.get(X, X):14s} {sh:+5.2f} / {(s1 + s2) / 2 - 2 * sh:+5.2f} / {sx - 2 * sh:+5.2f} / {(s1 + s2) / 2 - sx:4.2f}")
        res[rank]['bands'][label.strip()] = rr
json.dump(res, open(HERE / (f'a20_table_{tag}.json' if (XB, XA) == ('c3', 'law') else f'a20_table_{tag}_{XB}_{XA}.json'), 'w'), indent=1, default=float)
