"""TAG fable-lead 2026-10-19: does a per-network feature predict the network's optimal k4 band scale? Inputs: the feature probe rows
(probe_k4feat_v1) and a finished FD pair (x(1-delta), x(1+delta)) graded on the store ranks against the reference run.
Per rank: slope a_n = (f+ - f-)/(2 delta), curvature c_n = (f+ + f-)/(2 delta^2); the per-net optimum uses the POOLED curvature
(d_n = -a_n / (2 cbar), less noisy than the per-rank one). For every band-mean feature: Pearson r with d_n, and the LEAVE-ONE-OUT
linear conditional dial d_hat_n = b0 + b1 x_n (fitted without rank n), scored by the rank's own parabola a_n d + c_n d^2 at d_hat_n
versus at the common optimum dbar = mean(d_n). Prints the realizable LOO gain per feature (negative = helps).
usage: k4feat_analysis_v1.py <probe rows.jsonl> <ref run> <minus arm> <plus arm> <delta> <band layers e.g. 8,9,10,11> [raw offset %]"""
import json
import math
import statistics as st
import sys
from pathlib import Path

rows_path, ref, minus, plus, delta, layers = sys.argv[1:7]
delta = float(delta); layers = [int(x) for x in layers.split(',')]
offset = float(sys.argv[7]) if len(sys.argv) > 7 else -0.03


def grade(d):
    out = {}
    for name in ('candidate_all32', 'candidate7', 'candidate_heldout25'):
        p = Path(d) / name / 'GRADE_STORE_v1.json'
        if p.exists():
            for row in json.load(open(p)):
                out[int(row['rank'])] = row['raw_pct']
    return out


feat = {}
for line in open(rows_path):
    r = json.loads(line)
    per = [r['layers'][str(l)] for l in layers if str(l) in r['layers']]
    if len(per) != len(layers):
        continue
    keys = [k for k in per[0] if isinstance(per[0][k], (int, float)) and k != 'n']
    f = {k: st.mean(p[k] for p in per) for k in keys if all(p.get(k) is not None for p in per)}
    for l, p in zip(layers, per):   # per-layer copies too
        for k in keys:
            if p.get(k) is not None:
                f['L%d_%s' % (l, k)] = p[k]
    feat[int(r['rank'])] = f
gr, gm, gp = grade(ref), grade(minus), grade(plus)
ranks = sorted(set(gr) & set(gm) & set(gp) & set(feat))
print('%d ranks with features and both arms' % len(ranks))
a = {r: ((gp[r] - gr[r] - offset) - (gm[r] - gr[r] - offset)) / (2 * delta) for r in ranks}
c = {r: ((gp[r] - gr[r] - offset) + (gm[r] - gr[r] - offset)) / (2 * delta * delta) for r in ranks}
cbar = st.mean(c.values())
d = {r: -a[r] / (2 * cbar) for r in ranks}
dbar = st.mean(d.values())
print('pooled curvature %.0f, per-net optimum d_n: mean %+.4f sd %.4f (common optimum dbar %+.4f)' % (cbar, dbar, st.pstdev(d.values()), dbar))


def gain_at(r, x):
    return a[r] * x + c[r] * x * x


base_gain = st.mean(gain_at(r, dbar) for r in ranks)
oracle_gain = st.mean(gain_at(r, d[r]) for r in ranks)
print('raw gain at the common optimum %+.3f%%; at each net\'s own optimum (oracle) %+.3f%%' % (base_gain, oracle_gain))


def fit(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    b1 = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / vx if vx > 0 else 0.0
    return my - b1 * mx, b1


results = []
names = sorted(set.intersection(*[set(feat[r]) for r in ranks]))
for k in names:
    xs = [feat[r][k] for r in ranks]; ys = [d[r] for r in ranks]
    if st.pstdev(xs) == 0:
        continue
    mx, my = st.mean(xs), st.mean(ys)
    rho = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    loo = []
    for i, r in enumerate(ranks):
        b0, b1 = fit(xs[:i] + xs[i + 1:], ys[:i] + ys[i + 1:])
        dh = b0 + b1 * xs[i]
        dh = max(-0.15, min(0.15, dh))
        loo.append(gain_at(r, dh))
    results.append((st.mean(loo), rho, k))
results.sort()
print('feature                          LOO conditional-dial raw gain   Pearson r with d_n   (common-optimum gain %+.3f%%)' % base_gain)
for g, rho, k in results[:14]:
    print('  %-30s %+.3f%%   r %+.2f' % (k, g, rho))
print('  ... worst: %s' % ', '.join('%s %+.3f' % (k, g) for g, rho, k in results[-3:]))
