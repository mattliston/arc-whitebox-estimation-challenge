#!/usr/bin/env python3
"""TAG fable-engineer (E-61 platform check): compare two eval pages' setup and cold-MLP telemetry, to verify the setup()-time tree
prebinding on the platform. Prints setup_executions (worker, wall, flops, error), the cold MLPs (those with a setup wall) with predict
flops, predict wall and residual, and the warm MLPs' residual mean / max, for a base page and a new page.
usage: setup_telemetry.py <base eval json> <new eval json>"""
import sys, json, statistics as st
def load(p):
    ev = json.load(open(p))["evaluation"]; res = ev["results"]
    per = [m["telemetry"] for m in res["per_mlp"] if m.get("telemetry")]
    return res, per
for label, p in (("base", sys.argv[1]), ("new", sys.argv[2])):
    res, per = load(p)
    print(f"== {label}: {p}")
    agg = res.get("aggregates", {}).get("public", {})
    print("   public:", {k: agg.get(k) for k in ("score", "final_layer_mse", "mean_compute_utilization", "n_failed_mlps")})
    for s in res.get("setup_executions", []):
        print(f"   setup {s.get('worker_id')} gen {s.get('generation')}: wall {s.get('wall_s'):.3f} s  flops {s.get('flops'):,}  kernel {s.get('kernel_ns', 0)/1e9:.3f} s  comms {s.get('comms_ns', 0)/1e9:.3f} s  {s.get('error_code')}")
    cold = [t for t in per if t.get("participant_setup_wall_s") is not None]
    warm = [t for t in per if t.get("participant_setup_wall_s") is None]
    print(f"   MLPs {len(per)}: cold {len(cold)}, warm {len(warm)}")
    for t in cold:
        print(f"   cold: flops {t['flops_used']/1e9:8.2f} G  predict {t['participant_predict_wall_s']:6.2f} s  residual {t['residual_wall_time_s']:.3f} s")
    if cold:
        print(f"   cold flops mean {st.mean(t['flops_used'] for t in cold)/1e9:.2f} G; warm flops mean {st.mean(t['flops_used'] for t in warm)/1e9:.2f} G")
    r = [t["residual_wall_time_s"] for t in warm]
    print(f"   warm residual mean {st.mean(r):.3f} max {max(r):.3f}; all residual max {max(t['residual_wall_time_s'] for t in per):.3f}; roundtrip max {max(t['participant_roundtrip_wall_s'] for t in per):.1f} s")
