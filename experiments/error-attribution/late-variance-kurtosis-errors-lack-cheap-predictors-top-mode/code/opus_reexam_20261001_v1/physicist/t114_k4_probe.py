#!/usr/bin/env python3
"""TAG opus-physicist T114 (numpy, offline; lead 10-11 18:5xZ item 3): IS THE CARD'S PER-UNIT k4 STILL SYSTEMATICALLY LOW AT THE LATE LAYERS?
Card: the pair map's k4 input at layers 12-14 at S47e (the Nd point on the v11 simulator, pinned otr_m; t99_pindump_v2 dump of network 35: radial v^2 plus
K4L / F4C as the card forms it). Truth: the analyst's per-unit 32M-sample truths of network 35 (two independent halves A / B: mc_pre_k4, mc_pre_v, mc_pre_m).
Per layer and |alpha| band (alpha = the card's mu / sqrt(v)): units, median card / truth, the least-squares scale that would put the card on the truth,
and that scale weighted as the mean formula uses k4 (d m / d k4 = (alpha^2 - 1) pdf / (24 v s)); truth noise from A vs B.
usage: t114_k4_probe.py"""
import math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; O = HERE.parent
D = np.load(HERE / 't99_pindump_v2' / 'dumps' / 'P_35.npz')
A = np.load(O / 'analyst/fresh/pu/mps_truth_rank0035_A32M.npz'); B = np.load(O / 'analyst/fresh/pu/mps_truth_rank0035_B32M.npz')
bands = [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 99)]
for l in (12, 13, 14):
    C = D[f'C_{l}'].astype(np.float64); v = np.diag(C).clip(1e-12); s = np.sqrt(v); mu = D[f'mu_{l}'].astype(np.float64); al = mu / s
    kc = D[f'k4_{l}'].astype(np.float64); ta, tb = A['mc_pre_k4'][l].astype(np.float64), B['mc_pre_k4'][l].astype(np.float64); kt = (ta + tb) / 2
    w = ((al * al - 1) * np.exp(-al * al / 2) / math.sqrt(2 * math.pi) / (24 * v * s)) ** 2
    noise = np.median(np.abs(ta - tb) / 2 / np.abs(kt).clip(1e-30))
    print('\nlayer %d net 35: k4 excess t4 = k4 / v^2 median: card %.4f, truth %.4f; truth noise (half-split) median %.1f%% of |k4|; whole layer scale card -> truth: plain %.3f, mean-formula weighted %.3f' % (
        l, np.median(kc / v ** 2), np.median(kt / v ** 2), 100 * noise, (kc @ kt) / (kc @ kc), ((w * kc) @ kt) / ((w * kc) @ kc)))
    print('   |alpha| band | units | median card / truth | LS scale card -> truth | weighted LS scale | corr(card, truth)')
    for lo, hi in bands:
        m = (np.abs(al) >= lo) & (np.abs(al) < hi)
        if m.sum() < 10: continue
        r = kc[m] / kt[m]
        print('   %4.1f-%-4.1f | %4d | %7.3f | %7.3f | %7.3f | %6.3f' % (lo, hi if hi < 50 else float('inf'), int(m.sum()), float(np.median(r)), float(kc[m] @ kt[m] / (kc[m] @ kc[m])),
              float(((w[m] * kc[m]) @ kt[m]) / max((w[m] * kc[m]) @ kc[m], 1e-300)), float(np.corrcoef(kc[m], kt[m])[0, 1])))
