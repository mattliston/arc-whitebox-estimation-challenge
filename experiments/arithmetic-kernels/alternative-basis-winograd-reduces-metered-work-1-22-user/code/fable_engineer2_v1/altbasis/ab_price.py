#!/usr/bin/env python3
"""TAG fable-engineer2 (E2-ABP): PAPER price of Karstadt-Schwartz alternative-basis Strassen-Winograd on S13's in-place tree kernel, from
the tree-product census (ab_census_<tag>.json). Basis (derived; 12 adds/level): A' = (A11, A12, S2, S3) so the other left factors cost 3
adds (S4 = A12' - A21', A22 = A21' + A22', S1 = A21' + A11'), same for B; output decoded in (C11, C22-C12, C22-C21, C22) with 6 adds.
Per tree side with depth L and block volume V (A: mk, B: kn, C: mn; x P for batched), billed adds (flopscope bills adds at numel):
  today A/B: (4/3)((7/4)^L - 1) V;  C: (7/3)((7/4)^L - 1) V
  alt   A/B: level 1 fused (2 copies + 2 basis adds + 3 core = today's 3 copies + 4 adds) + basis 2 adds/level on the 4 plain chunks at
        levels 2..L ((L-1) V/2) + core 3 adds/level at levels 2..L  =>  delta = [(L-1)/2 - (((7/4)^L - 1)/3 - 1/4)] V, ops +(L-1)/call
  alt   C: core 6 adds/level at levels L..2, level 1 in standard basis 8 adds (vs 7), inverse basis 2 adds/level at levels 2..L on the
        output  =>  delta = [(L-1)/2 + 1/4 - (((7/4)^L - 1)/3 - 1/4)] V = [L/2 - ((7/4)^L - 1)/3] V, ops +L/call
Platform: user CPU ~ billed elementwise volume (transforms are memory-bound); ops x 0.41 ms. usage: ab_price.py <tag>"""
import json, sys, collections
from pathlib import Path
H = Path(__file__).resolve().parent
G = lambda L: ((7 / 4) ** L - 1) / 3
for net in json.load(open(H / f"ab_census_{sys.argv[1]}.json")):
    tot = net["flops"]; today = {"A": 0.0, "B": 0.0, "C": 0.0}; delta = {"A": 0.0, "B": 0.0, "C": 0.0}; dops = 0; byL = collections.Counter()
    for side, L, r, c, P, cnt in net["census"]:
        V = r * c * P * cnt
        if side in "AB":
            today[side] += (4 / 3) * ((7 / 4) ** L - 1) * V; delta[side] += ((L - 1) / 2 - (G(L) - 0.25)) * V; dops += (L - 1) * cnt
        else:
            today[side] += (7 / 3) * ((7 / 4) ** L - 1) * V; delta[side] += (L / 2 - G(L)) * V; dops += L * cnt
        byL[(side, L)] += cnt
    t = sum(today.values()); d = sum(delta.values()); addsub = net["vol"].get("add", 0) + net["vol"].get("subtract", 0)
    print(f"net r{net['rank']}: bill {tot/1e9:.1f} Gflop; tree transform adds today {t/1e9:.2f} Gflop ({100*t/tot:.2f}% of bill; measured add+subtract all ops {addsub/1e9:.2f})")
    print(f"   alt-basis delta: A {delta['A']/1e9:+.3f}, B {delta['B']/1e9:+.3f}, C {delta['C']/1e9:+.3f} Gflop -> {d/1e9:+.3f} Gflop = {100*d/tot:+.3f}% of the bill "
          f"({100*d/t:+.1f}% of tree transforms); ops {dops:+d}/net -> platform +{dops*0.41e-3:.2f} s")
    print(f"   depth mix (side, L): calls {dict(sorted(byL.items()))}")
