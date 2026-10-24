#!/usr/bin/env python3
"""TAG analyst A16 pair (2026-10-10): podium chance of a DESIGNATED PAIR of files under the rules' best-of-two ('Where a Participant has more than
one submission eligible for score-based prize consideration in a Phase, only that Participant's best-scoring such submission is ranked').
Inputs: a16_loss_risk.json our_score_sensitivity (P(podium) for our score x f, rivals' losses on, OUR OWN loss risk p_any already inside, so
P(beat the line | no loss of ours) = sens / (1 - p_any)), linearly interpolated in f; two files of the same physics: A (bill cheaper by dA %, loss L_A)
and S (the safe form, loss L_S). Losses independent of the networks' error draw; between A and S two cases: independent, or nested (S lost => A lost).
usage: a16_pair.py [dA% L_A L_S]   (default: the release's 03:02Z numbers, -0.39% bill at P(loss) 6.2% against the safe form at 1.1%)"""
import json, sys
from pathlib import Path
import numpy as np
d = json.loads((Path(__file__).resolve().parent / 'a16_loss_risk.json').read_text())
p_us = d['risk']['mliston']['p_any']
f, s = np.array(sorted((float(k), v[1]) for k, v in d['our_score_sensitivity'].items())).T
beat = lambda x: float(np.interp(x, f, s / (1 - p_us)))
dA, LA, LS = (float(a) for a in sys.argv[1:4]) if len(sys.argv) > 3 else (0.39, 0.062, 0.011)
print(f'our own loss risk inside the sensitivities: {p_us * 100:.2f}%; A = bill -{dA}% at P(loss) {LA * 100:.1f}%, S = safe form at {LS * 100:.1f}%')
print('position (S score x) | P(beat) S / A | S alone | A alone | pair, losses independent | pair, nested | A request risk price in the pair vs alone')
for x in (1.00, 0.97, 0.96, 0.95, 0.93):
    bs, ba = beat(x), beat(x * (1 - dA / 100))
    single_s, single_a = (1 - LS) * bs, (1 - LA) * ba
    pair_i, pair_n = (1 - LA) * ba + LA * (1 - LS) * bs, (1 - LA) * ba + (LA - LS) * bs
    print(f'x{x:.2f} | {bs * 100:.1f}% / {ba * 100:.1f}% | {single_s * 100:.1f}% | {single_a * 100:.1f}% | {pair_i * 100:.1f}% | {pair_n * 100:.1f}% | '
          f'{(ba - (1 - LS) * bs) / ba:.3f}')
