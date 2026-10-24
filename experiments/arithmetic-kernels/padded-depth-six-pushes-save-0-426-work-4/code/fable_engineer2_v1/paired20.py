#!/usr/bin/env python3
"""TAG fable-engineer2 (E2-P20): digest two concurrent 20-net runs (run_mem_caches.sh <cand> <nets> <tag>) into a paired table:
per net handles, live peak, server CPU, residual, round trips and util for candidate vs reference, plus paired summaries.
usage: paired20.py <cand_tag> <ref_tag>"""
import json, sys, statistics as st
from pathlib import Path
E = Path(__file__).resolve().parent
def load(tag):
    recs = [json.loads(l.split(": ", 1)[1]) for l in open(E / f"{tag}.out") if l.startswith(f"{tag}: ")]
    wins = [w for w in (json.loads(l) for l in open(E / f"memwin_{tag}.jsonl")) if w.get("window")]   # skip empty-label windows (control-file race)
    return recs, wins
c, cw = load(sys.argv[1]); r, rw = load(sys.argv[2])
n = min(len(c), len(r), len(cw), len(rw))
print(f"{'net':>4} {'handles c/r':>15} {'peak MB c/r':>15} {'srvCPU c/r (d%)':>22} {'resid c/r (ms)':>20} {'rt c/r':>13} {'util d%':>8}")
cpu, res, ut = [], [], []
for i in range(n):
    a, b = c[i], r[i]
    d_cpu = 100 * (a['server_cpu_s'] / b['server_cpu_s'] - 1); cpu.append(d_cpu)
    d_res = 1000 * (a['residual_s'] - b['residual_s']); res.append(d_res)
    d_ut = 100 * (a['util'] / b['util'] - 1); ut.append(d_ut)
    print(f"{a['rank']:>4} {cw[i]['handles']:>7}/{rw[i]['handles']:<7} {cw[i]['peak_live_MB']:>7.0f}/{rw[i]['peak_live_MB']:<7.0f} "
          f"{a['server_cpu_s']:>6.1f}/{b['server_cpu_s']:<6.1f}({d_cpu:+5.1f}) {1000*a['residual_s']:>5.0f}/{1000*b['residual_s']:<5.0f}({d_res:+4.0f}) "
          f"{a['round_trips']:>6}/{b['round_trips']:<6} {d_ut:+7.3f}")
print(f"paired server CPU: mean {st.mean(cpu):+.2f}% median {st.median(cpu):+.2f}%; residual: mean {st.mean(res):+.1f} ms, cand max "
      f"{1000*max(x['residual_s'] for x in c[:n]):.0f} ms; util: mean {st.mean(ut):+.3f}%; handles after {n}: {cw[n-1]['handles']} vs {rw[n-1]['handles']} "
      f"(max {max(x['handles'] for x in cw[:n])} vs {max(x['handles'] for x in rw[:n])}); peak {max(x['peak_live_MB'] for x in cw[:n]):.0f} vs {max(x['peak_live_MB'] for x in rw[:n]):.0f} MB")
