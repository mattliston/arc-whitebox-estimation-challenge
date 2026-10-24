"""TAG fable-lead 2026-10-19: parabola read of a dial FD pair at ARBITRARY steps dA, dB (not necessarily symmetric) on the store ranks,
plus the truth-half split of the per-rank slopes (how much of the per-network optimum spread is real structure vs truth-sample noise).
f(d) = a d + c d^2 (raw % vs the science base) through the two arms' per-rank raw differences against the reference run (same 95c
control; minus the base-vs-reference raw offset). Truth halves A/B (analyst/fresh/pu mps_truth_rank<r>_{A,B}32M.npz) and the arms'
pred_<rank>.bin files give the per-rank slopes on each half separately.
usage: fd_pair_asym_v1.py <ref run> <arm A dir> <dA> <arm B dir> <dB> [raw offset %, default -0.03]"""
import glob
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np

ref, armA, dA, armB, dB = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4], float(sys.argv[5])
offset = float(sys.argv[6]) if len(sys.argv) > 6 else -0.03
PU = Path(__file__).resolve().parents[3] / 'analyst/fresh/pu'


def grade(d):
    out = {}
    for name in ('candidate_all32', 'candidate7', 'candidate_heldout25'):
        p = Path(d) / name / 'GRADE_STORE_v1.json'
        if p.exists():
            for row in json.load(open(p)):
                out[int(row['rank'])] = row['raw_pct']
    return out


def preds(run):
    out = {}
    for p in glob.glob(run + '/candidate_*/pred_*.bin'):
        out[int(Path(p).stem.split('_')[1])] = np.fromfile(p, dtype='<f4').reshape(16, 1024).astype('d')[15]
    return out


def solve(fa, fb):
    # a dA + c dA^2 = fa ; a dB + c dB^2 = fb
    det = dA * dB * dB - dB * dA * dA
    a = (fa * dB * dB - fb * dA * dA) / det
    c = (dA * fb - dB * fa) / det
    return a, c


gr, ga, gb = grade(ref), grade(armA), grade(armB)
ranks = sorted(set(gr) & set(ga) & set(gb))
fa = {r: ga[r] - gr[r] - offset for r in ranks}
fb = {r: gb[r] - gr[r] - offset for r in ranks}
a, c = solve(st.mean(fa.values()), st.mean(fb.values()))
print('%d ranks | f(%+.4f) %+.3f%%  f(%+.4f) %+.3f%% | pooled slope %+.2f curvature %+.1f' % (len(ranks), dA, st.mean(fa.values()), dB, st.mean(fb.values()), a, c))
if c > 0:
    d = -a / (2 * c)
    print('pooled optimum d* %+.4f (scale x%.4f / additive %+.4f) predicted raw gain %+.3f%%' % (d, 1 + d, d, a * d + c * d * d))
per = [solve(fa[r], fb[r]) for r in ranks]
cbar = st.mean(x[1] for x in per)
dn = [-x[0] / (2 * cbar) for x in per]
print('per-rank slopes sd %.1f; per-net optimum (pooled curvature %.0f) sd %.4f; oracle per-net gain %+.3f%%' % (st.pstdev(x[0] for x in per), cbar, st.pstdev(dn), st.mean(x[0] * dd + x[1] * dd * dd for x, dd in zip(per, dn))))
# truth-half split
pr, pa, pb = preds(ref), preds(armA), preds(armB)
common = [r for r in ranks if r in pr and r in pa and r in pb and (PU / ('mps_truth_rank%04d_A32M.npz' % r)).exists()]
sA, sB = [], []
for r in common:
    A = np.load(PU / ('mps_truth_rank%04d_A32M.npz' % r))['mc_post'].astype('d')[15]
    B = np.load(PU / ('mps_truth_rank%04d_B32M.npz' % r))['mc_post'].astype('d')[15]
    for store, t in ((sA, A), (sB, B)):
        u0, ua, ub = (float(np.mean((p - t) ** 2)) for p in (pr[r], pa[r], pb[r]))
        store.append(solve(100 * (ua / u0 - 1), 100 * (ub / u0 - 1))[0])
if len(common) >= 6:
    mA, mB = st.mean(sA), st.mean(sB)
    rho = sum((x - mA) * (y - mB) for x, y in zip(sA, sB)) / math.sqrt(sum((x - mA) ** 2 for x in sA) * sum((y - mB) ** 2 for y in sB))
    noise = st.pstdev((x - y) / 2 for x, y in zip(sA, sB)); sig = st.pstdev((x + y) / 2 for x, y in zip(sA, sB))
    true = math.sqrt(max(sig * sig - noise * noise, 0.0))
    print('truth-half split on %d ranks: corr(slope_A, slope_B) %+.2f; noise sd %.1f, true-structure sd %.1f -> true optimum sd %.4f, true oracle gain ~ %+.3f%%'
          % (len(common), rho, noise, true, true / (2 * cbar) if cbar > 0 else float('nan'), -cbar * (true / (2 * cbar)) ** 2 if cbar > 0 else float('nan')))
