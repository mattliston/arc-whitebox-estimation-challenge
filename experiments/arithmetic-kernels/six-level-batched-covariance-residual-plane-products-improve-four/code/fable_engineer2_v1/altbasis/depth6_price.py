#!/usr/bin/env python3
"""TAG fable-engineer2 (E2-D6P): price table for moving depth-5 alt-basis tree products to depth 6, from depth6_ab_shadow_v2 output.
Per class (kind, m, k, n; W5 port split into creator / reuse calls) and per net: calls, bill change (per steady call and % of the net's
native bill), local user-CPU change, op-call change; then
  platform wall change = 0.78 x (local user change / server user per net) x K_worst + op calls x 0.41 ms   (project_wall_v3 central:
      r = 0.78, K_worst of the base page, 0.41 ms per op call; server user per net = mean 'user' of the given porter-runner optime file);
  residual change = op calls x 5.8 us (platform residual per request);
ranked by bill saved per platform wall-second, with the cumulative take-the-top-k line, and bill per residual-ms.
usage: depth6_price.py <depth6_ab_TAG.json> [optime_file_for_server_user=optime_e_S14.json] [base_sid=332195]"""
import sys, json, collections, statistics as st
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; E = HERE.parent
runs = json.load(open(HERE / sys.argv[1] if not Path(sys.argv[1]).is_absolute() else sys.argv[1]))
ufile = sys.argv[2] if len(sys.argv) > 2 else "optime_e_S14.json"; sid = sys.argv[3] if len(sys.argv) > 3 else "332195"
U = st.mean(x["user"] for x in json.load(open(E / "platfit" / ufile)))
t = [m["telemetry"] for m in json.load(open(E.parent / "fable_fire_v1" / f"eval_{sid}.json"))["evaluation"]["results"]["per_mlp"] if m.get("telemetry")]
pred = np.array([x["participant_predict_wall_s"] for x in t]); fl = np.array([x["flops_used"] for x in t])
cold = np.array([x.get("participant_setup_wall_s") is not None for x in t], float)
X = np.column_stack([np.ones(len(pred)), (fl - fl.mean()) / max(fl.std(), 1.0), cold]); fit = X @ np.linalg.lstsq(X, pred, rcond=None)[0]
Kw = float(np.mean([x["flopscope_server_kernel_wall_s"] for x in t])) * float(fit.max()) / float(pred.mean())
R, OP, RES = 0.78, 0.41e-3, 5.8e-6
wall = lambda du, dq: R * du / U * Kw + dq * OP
print(f"base #{sid}: K_worst {Kw:.2f} s; server user per net {U:.2f} s ({ufile}); 1 local user-s = {R*Kw/U:.3f} platform s; 1 op call = 0.41 ms wall, 5.8 us residual")
for run in runs:
    calls = [c for c in run["calls"] if not c.get("skip")]; skip = [c for c in run["calls"] if c.get("skip")]
    B = run["native_bill"]
    print(f"\nnet r{run['rank']}: native bill {B/1e9:.2f} G (parity {run['ref_bill']/1e9 if run['ref_bill'] else float('nan'):.2f} G, equal={run['native_bill']==run['ref_bill']}), "
          f"predictions bit-identical to parity: {run['bit_identical']}; native user {run['native_user']:.1f} s, ops {run['native_ops']}; "
          f"{len(calls)} depth-5 tree products shadowed, {len(skip)} skipped {sorted(set(tuple(c['key']) for c in skip))}")
    cls = collections.defaultdict(list)
    for c in calls: cls[tuple(c["key"])].append(c)
    rows = []
    for key, cs in cls.items():
        d = lambda c, f: c["L6"][f] - c["L5"][f]
        steady = [c for c in cs if not c["first"]] or cs
        db = sum(d(c, "bill") for c in cs); du = sum(d(c, "user") for c in cs); dq = sum(d(c, "ops") for c in cs)
        w = wall(du, dq); r = dq * RES * 1e3
        rows.append(dict(key=key, n=len(cs), b5=st.mean(c["L5"]["bill"] for c in steady), db1=st.mean(d(c, "bill") for c in steady),
                         u5=st.mean(c["L5"]["user"] for c in steady), du1=st.mean(d(c, "user") for c in steady), dq1=st.mean(d(c, "ops") for c in steady),
                         first_db=sum(d(c, "bill") for c in cs if c["first"]) - sum(1 for c in cs if c["first"]) * st.mean(d(c, "bill") for c in steady),
                         first_dq=sum(d(c, "ops") for c in cs if c["first"]) - sum(1 for c in cs if c["first"]) * st.mean(d(c, "ops") for c in steady),
                         pct=100 * db / B, du=du, dq=dq, w=w, r=r, mr=max(c["maxrel"] for c in cs),
                         sites=sorted(collections.Counter(c["site"] for c in cs).items(), key=lambda x: -x[1])[:3]))
    rows.sort(key=lambda x: (x["pct"] >= 0, -(-x["pct"] / x["w"]) if x["w"] > 0 else -1e9))
    print(f"  {'class (kind m k n)':34s} {'calls':>5s} {'L5 G/call':>9s} {'dG/call':>8s} {'L5 ms':>6s} {'dms':>6s} {'dops':>5s} | {'bill %':>7s} {'user s':>7s} {'ops':>6s} "
          f"{'wall s':>7s} {'resid ms':>8s} | {'%/wall-s':>8s} {'%/res-ms':>8s} {'maxrel':>7s} | cum bill%  wall s  resid ms")
    cb = cw = cr = 0.0
    for x in rows:
        cb += x["pct"]; cw += x["w"]; cr += x["r"]
        pw = -x["pct"] / x["w"] if x["w"] > 0 else float("inf"); pr = -x["pct"] / x["r"] if x["r"] > 0 else float("inf")
        print(f"  {' '.join(map(str, x['key'])):34s} {x['n']:5d} {x['b5']/1e9:9.3f} {x['db1']/1e9:+8.4f} {1e3*x['u5']:6.1f} {1e3*x['du1']:+6.1f} {x['dq1']:+5.0f} | "
              f"{x['pct']:+7.3f} {x['du']:+7.2f} {x['dq']:+6.0f} {x['w']:+7.2f} {x['r']:+8.2f} | {pw:8.3f} {pr:8.3f} {x['mr']:7.1e} | {cb:+8.3f} {cw:+7.2f} {cr:+8.2f}")
        print(f"  {'':34s} sites {x['sites']}; first-call extra: bill {x['first_db']/1e9:+.3f} G, ops {x['first_dq']:+.0f}")
