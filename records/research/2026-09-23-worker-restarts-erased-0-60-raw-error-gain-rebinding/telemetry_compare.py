#!/usr/bin/env python3
"""TAG fable-engineer2 (E2-T): per-MLP platform telemetry comparison of two graded submissions (fable_fire_v1/eval_<sid>.json from
read_eval_page.py): mean/median/max of wall, kernel, overhead, comms, residual and util; paired differences by MLP-index block (the
first 5 = each worker's cold first MLP); top-10 residuals; residual percentiles; residual-vs-overhead correlation.
usage: telemetry_compare.py <sid_new> <sid_ref>"""
import json, math, statistics as st, sys
from pathlib import Path
F = Path(__file__).resolve().parent.parent / "fable_fire_v1"
def load(sid):
    d = json.load(open(F / f"eval_{sid}.json"))["evaluation"]
    return {m["mlp_index"]: m for m in d["results"]["per_mlp"] if m.get("telemetry")}, d["results"]["aggregates"]["public"]
new, agg_n = load(sys.argv[1]); ref, agg_r = load(sys.argv[2])
assert all(new[i]["mlp_name"] == ref[i]["mlp_name"] for i in new if i in ref)
T = lambda m, k: m["telemetry"][k]
K = [("participant_predict_wall_s", "wall"), ("flopscope_server_kernel_wall_s", "kernel"), ("flopscope_server_overhead_wall_s", "overhead"),
     ("flopscope_server_comms_wall_s", "comms"), ("residual_wall_time_s", "residual")]
for lab, runs, agg in (("new " + sys.argv[1], new, agg_n), ("ref " + sys.argv[2], ref, agg_r)):
    cells = []
    for k, n in K:
        v = sorted(T(m, k) for m in runs.values()); cells.append(f"{n} {st.mean(v):.2f}/{v[len(v)//2]:.2f}/{v[-1]:.2f}")
    u = st.mean(T(m, "flops_used") / 2**41 for m in runs.values())
    print(f"{lab:12s} " + "  ".join(cells) + f"  util {u:.5f}  score {agg.get('score')}")
blocks = [(0, 5), (5, 10), (10, 20), (20, 40), (40, 70), (70, 100)]
print("block   wall d (sd)     overhead d   residual new/ref (ms)   d")
for a, b in blocks:
    idx = [i for i in range(a, b) if i in new and i in ref]
    dw = [T(new[i], K[0][0]) - T(ref[i], K[0][0]) for i in idx]; do = [T(new[i], K[2][0]) - T(ref[i], K[2][0]) for i in idx]
    rn = [1000 * T(new[i], K[4][0]) for i in idx]; rr = [1000 * T(ref[i], K[4][0]) for i in idx]
    print(f"{a:3d}-{b-1:<3d}  {st.mean(dw):+6.2f} ({st.pstdev(dw):4.1f})    {st.mean(do):+6.2f}      {st.mean(rn):6.1f} / {st.mean(rr):6.1f}      {st.mean(rn)-st.mean(rr):+6.1f}")
cold = [T(new[i], K[0][0]) - T(ref[i], K[0][0]) for i in range(5)]
m, s = st.mean(cold), st.stdev(cold)
print(f"cold wall diffs {[round(x, 1) for x in cold]} mean {m:+.2f} s sd {s:.2f} t {m / (s / math.sqrt(5)):.2f}")
dres = [1000 * (T(new[i], K[4][0]) - T(ref[i], K[4][0])) for i in new if i in ref]
print(f"residual diff all MLPs: mean {st.mean(dres):+.1f} ms, positives {sum(1 for x in dres if x > 0)}/{len(dres)}")
top = sorted(new, key=lambda i: -T(new[i], K[4][0]))[:10]
print("top-10 residual new (index, ms, ref ms, wall s):", [(i, round(1000 * T(new[i], K[4][0])), round(1000 * T(ref[i], K[4][0])), round(T(new[i], K[0][0]), 1)) for i in top])
vn = sorted(1000 * T(m, K[4][0]) for m in new.values()); vr = sorted(1000 * T(m, K[4][0]) for m in ref.values())
print(f"residual p95/p99/max: new {vn[94]:.0f}/{vn[98]:.0f}/{vn[-1]:.0f} ms  ref {vr[94]:.0f}/{vr[98]:.0f}/{vr[-1]:.0f} ms")
wn = sorted(T(m, K[0][0]) for m in new.values()); print(f"wall top-5 new: {[round(x, 1) for x in wn[-5:]]}")
xs = [T(new[i], K[2][0]) for i in range(5, 100) if i in new]; ys = [1000 * T(new[i], K[4][0]) for i in range(5, 100) if i in new]
mx, my = st.mean(xs), st.mean(ys)
c = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
print(f"warm MLPs corr(overhead wall, residual) = {c:+.2f}")
