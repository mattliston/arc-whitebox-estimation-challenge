#!/usr/bin/env python3
"""TAG fable-engineer2 (E2-NSK): platform server kernel ns per kflop on MLP indices shared by all given runs (indices 5+ skip the
first worker-cold block), median and mean, plus paired kernel/wall deltas of the first run vs each other run. Eval JSONs: this
folder's copy if present, else fable_fire_v1's. usage: nskflop.py <sid_new> <sid_ref1> [sid_ref2 ...]"""
import json, sys, statistics as st
from pathlib import Path
H = Path(__file__).resolve().parent; F = H.parent.parent / "fable_fire_v1"
def load(sid):
    p = H / f"eval_{sid}.json"
    if not p.exists(): p = F / f"eval_{sid}.json"
    ev = json.load(open(p))["evaluation"]
    return {m["mlp_index"]: m["telemetry"] for m in ev["results"]["per_mlp"] if m.get("telemetry")}, ev.get("progress")
runs = {sid: load(sid) for sid in sys.argv[1:]}
common = sorted(set.intersection(*(set(r[0]) for r in runs.values())) - set(range(5)))
print(f"shared indices 5+: {len(common)}" + (f" ({common[0]}..{common[-1]})" if common else ""))
for sid, (t, prog) in runs.items():
    ns = [1e12 * t[i]["flopscope_server_kernel_wall_s"] / t[i]["flops_used"] for i in common]
    print(f"{sid}: ns/kflop median {st.median(ns):.1f} mean {st.mean(ns):.1f}; kernel mean {st.mean(t[i]['flopscope_server_kernel_wall_s'] for i in common):.2f} s; "
          f"wall max {max(t[i]['participant_predict_wall_s'] for i in t):.1f} s over {len(t)} MLPs; progress {prog}")
new = sys.argv[1]
for sid in sys.argv[2:]:
    a, b = runs[new][0], runs[sid][0]
    dk = [a[i]["flopscope_server_kernel_wall_s"] / b[i]["flopscope_server_kernel_wall_s"] - 1 for i in common]
    dw = [a[i]["participant_predict_wall_s"] - b[i]["participant_predict_wall_s"] for i in common]
    du = [a[i]["flops_used"] / b[i]["flops_used"] - 1 for i in common]
    print(f"{new} vs {sid}: kernel {100*st.mean(dk):+.2f}% (median {100*st.median(dk):+.2f}%), wall {st.mean(dw):+.2f} s, flops {100*st.mean(du):+.2f}%")
