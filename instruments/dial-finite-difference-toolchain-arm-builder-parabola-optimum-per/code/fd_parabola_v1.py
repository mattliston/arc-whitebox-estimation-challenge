"""TAG fable-lead 2026-10-19: parabola read of a zero-bill dial finite-difference PAIR on the store ranks.
For dial scale s = 1 + d: f(d) = a d + c d^2 (raw % vs the base), fitted from the two arms (x(1-delta), x(1+delta)) per rank against
the reference run (same 95c control; per-rank raw_pct differences, minus the known raw offset of the science base vs the reference).
Prints a (slope %/unit), c (curvature), d* = -a/(2c) (the parabola's optimum), predicted raw gain at d*, the per-rank bootstrap se of
that gain, and the fraction of ranks whose own parabola predicts a gain.
usage: fd_parabola_v1.py <ref run dir> <minus arm dir> <plus arm dir> <delta> [raw offset % of base vs ref, default -0.03]
   or: fd_parabola_v1.py --scan <ref run dir> <K dir> <delta> <suffix pair e.g. _m,_p>   (every finished pair)"""
import json
import math
import random
import statistics as st
import sys
from pathlib import Path


def grade(d):
    out = {}
    for name in ('candidate_all32', 'candidate7', 'candidate_heldout25'):
        p = Path(d) / name / 'GRADE_STORE_v1.json'
        if p.exists():
            for row in json.load(open(p)):
                out[int(row['rank'])] = row['raw_pct']
    return out


def parabola(fm, fp, delta):
    a = (fp - fm) / (2 * delta)
    c = (fp + fm) / (2 * delta * delta)
    if c <= 0:
        return a, c, None, None
    d = -a / (2 * c)
    return a, c, d, a * d + c * d * d


def read_pair(ref, minus, plus, delta, offset):
    gr, gm, gp = grade(ref), grade(minus), grade(plus)
    ranks = sorted(set(gr) & set(gm) & set(gp))
    if len(ranks) < 4:
        return None
    fm = [gm[r] - gr[r] - offset for r in ranks]
    fp = [gp[r] - gr[r] - offset for r in ranks]
    a, c, d, gain = parabola(st.mean(fm), st.mean(fp), delta)
    # bootstrap the pooled gain over ranks
    rnd = random.Random(0)
    gains = []
    for _ in range(400):
        idx = [rnd.randrange(len(ranks)) for _ in ranks]
        _a, _c, _d, g = parabola(st.mean(fm[i] for i in idx), st.mean(fp[i] for i in idx), delta)
        gains.append(g if g is not None else 0.0)
    se = st.pstdev(gains)
    per = [parabola(x, y, delta)[3] for x, y in zip(fm, fp)]
    helped = sum(1 for g in per if g is not None and g < 0)
    # gain of the pooled optimum evaluated per rank (the honest paired read of the move d)
    at_d = [((y - x) / (2 * delta)) * d + ((y + x) / (2 * delta * delta)) * d * d for x, y in zip(fm, fp)] if d is not None else None
    return dict(n=len(ranks), fm=st.mean(fm), fp=st.mean(fp), a=a, c=c, d=d, gain=gain, se=se, helped=helped,
                at_d_mean=(st.mean(at_d) if at_d else None), at_d_lower=(sum(1 for g in at_d if g < 0) if at_d else None))


def show(label, r, delta):
    if r is None:
        print('%-14s not enough common ranks' % label)
        return
    opt = 'x%.4f' % (1 + r['d']) if r['d'] is not None else 'no minimum (c<=0)'
    g = ('%+.3f%% (boot se %.3f)' % (r['gain'], r['se'])) if r['gain'] is not None else '-'
    print('%-14s n=%2d  f(-%.2f) %+.3f  f(+%.2f) %+.3f | slope %+.2f curv %+.1f | optimum %s  predicted raw %s  ranks helped %d/%d'
          % (label, r['n'], delta, r['fm'], delta, r['fp'], r['a'], r['c'], opt, g, r['helped'], r['n']))


if sys.argv[1] == '--scan':
    ref, kdir, delta = sys.argv[2], Path(sys.argv[3]), float(sys.argv[4])
    sm, sp = sys.argv[5].split(',')
    offset = float(sys.argv[6]) if len(sys.argv) > 6 else -0.03
    for m in sorted(kdir.glob('science32_*%s_m15' % sm)):
        tag = m.name[len('science32_'):-len('%s_m15' % sm)]
        p = kdir / ('science32_%s%s_m15' % (tag, sp))
        if (m / 'RESULT32_v1.json').exists() and (p / 'RESULT32_v1.json').exists():
            show(tag, read_pair(ref, m, p, delta, offset), delta)
else:
    ref, minus, plus, delta = sys.argv[1:5]
    offset = float(sys.argv[5]) if len(sys.argv) > 5 else -0.03
    show(Path(minus).name, read_pair(ref, minus, plus, float(delta), offset), float(delta))
