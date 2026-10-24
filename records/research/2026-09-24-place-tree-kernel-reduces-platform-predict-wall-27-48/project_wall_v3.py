#!/usr/bin/env python3
"""TAG fable-engineer2 (E2-PW v3 = project_wall_v2 recalibrated on S13 #332195 vs S11 #332182, a -40% change):
 - platform kernel % change = r x paired local USER-CPU % change; r band 0.60 / 0.78 / 0.95 (covers push5p6's two platform runs 0.57 / 0.94; S13: 0.784 = -31.53% platform kernel vs
   -40.20% local user; push5p6 two runs 0.57 / 0.94);
 - the change is applied to the WORST MLP's kernel, K_worst = K_mean x fit_max / predict_mean of the base page (the expected worst is the
   heaviest MLP's fitted wall), not to the mean kernel;
 - op-call change x 0.41 ms (S13: overhead + comms fell 0.71 s for -1,732 op calls; band 0.30 / 0.41 / 0.50 ms);
 - scenarios are ordered by the resulting expected worst (pessimistic = highest), so additions and reductions are both labelled right
   (v2 labelled only reductions right).
Base from an eval page (--base=<sid>, default 332195 = S13, the bank). Inputs as v2: --pair=NAME:cand.json:ref.json (platfit/ optime_split_v3
files), --item=NAME:bill:pess_s:cent_s:opt_s, --combo=LABEL:A+B, --fit=NAME:bill:lo_s:hi_s. --backtest=<pair>:<target_sid> prints the
projection of a pair measured against the base vs the target page's actual expected worst.
usage: project_wall_v3.py [--base=SID] --pair=... [--item=...] [--combo=...] [--fit=...] [--backtest=NAME:SID]"""
import json, sys, statistics as st
from pathlib import Path
import numpy as np
T = Path(__file__).resolve().parent; H = T.parent / "platfit"; F = T.parent.parent / "fable_fire_v1"
args = sys.argv[1:]; get = lambda key: [a.split("=", 1)[1] for a in args if a.startswith(f"--{key}=")]
R = (0.60, 0.78, 0.95); OPS = (0.30e-3, 0.41e-3, 0.50e-3)
def page(sid):
    p = T / f"eval_{sid}.json"
    if not p.exists(): p = F / f"eval_{sid}.json"
    t = [m["telemetry"] for m in json.load(open(p))["evaluation"]["results"]["per_mlp"] if m.get("telemetry")]
    pred = np.array([x["participant_predict_wall_s"] for x in t]); rt = np.array([x["participant_roundtrip_wall_s"] for x in t])
    fl = np.array([x["flops_used"] for x in t]); cold = np.array([x.get("participant_setup_wall_s") is not None for x in t], float)
    X = np.column_stack([np.ones(len(pred)), (fl - fl.mean()) / max(fl.std(), 1.0), cold]); fit = X @ np.linalg.lstsq(X, pred, rcond=None)[0]
    K = float(np.mean([x["flopscope_server_kernel_wall_s"] for x in t]))
    return dict(ew=float(fit.max() + np.median(rt - pred)), K=K, Kw=K * float(fit.max()) / float(pred.mean()))
base = (get("base") or ["332195"])[0]; B = page(base)
comp = {}
for p in get("pair"):
    name, cf, rf = p.split(":")
    a = json.load(open(H / cf)); b = json.load(open(H / rf))
    du = st.mean(x["user"] / y["user"] - 1 for x, y in zip(a, b)); se = st.stdev(x["user"] / y["user"] - 1 for x, y in zip(a, b)) / len(a) ** 0.5
    dc = st.mean(sum(v[0] for v in x["ops"].values()) - sum(v[0] for v in y["ops"].values()) for x, y in zip(a, b))
    fl = st.mean(x["flops"] / y["flops"] - 1 for x, y in zip(a, b))
    outs = sorted(r * du * B["Kw"] + dc * o for r in R for o in OPS)
    comp[name] = dict(bill=fl, s=[outs[-1], R[1] * du * B["Kw"] + dc * OPS[1], outs[0]], note=f"user {100*du:+.2f}% (se {100*se:.2f}%), op calls {dc:+.0f}/net")
for it in get("item"):
    name, b_, p_, c_, o_ = it.split(":"); comp[name] = dict(bill=float(b_), s=[float(p_), float(c_), float(o_)], note="paper item")
def line(label, bill, s, note=""):
    ew = [B["ew"] + x for x in s]
    print(f"{label:34s} bill {100*bill:+.2f}%  expected worst {ew[0]:.1f} / {ew[1]:.1f} / {ew[2]:.1f} s  room {100-ew[0]:.1f} / {100-ew[1]:.1f} / {100-ew[2]:.1f} s  {note}")
    return ew
print(f"base #{base}: expected worst {B['ew']:.1f} s, mean kernel {B['K']:.2f} s, worst-MLP kernel {B['Kw']:.2f} s; r {R}, op cost {OPS} s")
for n, v in comp.items(): line(n, v["bill"], v["s"], v["note"])
fits = [f.split(":") for f in get("fit")]
for cb in get("combo"):
    label, parts = cb.split(":"); parts = parts.split("+"); bill = 1.0
    for p in parts: bill *= 1 + comp[p]["bill"]
    ew = line(label, bill - 1, [sum(comp[p]["s"][i] for p in parts) for i in range(3)], "(ADDITIVE)" if len(parts) > 1 else "")
    for fname, fb, lo, hi in fits:
        lo, hi = float(lo), float(hi)
        print(f"    + {fname} (+{lo}..+{hi} s): " + " / ".join(f"{e+lo:.1f}-{e+hi:.1f}" for e in ew))
for bt in get("backtest"):
    name, sid = bt.split(":"); act = page(sid)["ew"]; s = comp[name]["s"]
    print(f"BACKTEST {name} on #{base} -> #{sid}: projected {B['ew']+s[0]:.1f} / {B['ew']+s[1]:.1f} / {B['ew']+s[2]:.1f} s vs ACTUAL {act:.1f} s")
