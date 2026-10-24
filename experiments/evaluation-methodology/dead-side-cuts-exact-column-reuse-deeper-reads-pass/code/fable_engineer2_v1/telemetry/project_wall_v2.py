#!/usr/bin/env python3
"""TAG fable-engineer2 (E2-PW v2 = project_wall.py + paired USER-CPU inputs): expected-worst projection on a fired base (default S11
#332182: 97.6 s by the analyst's method, mean platform kernel 84.91 s). Three scenarios (pess / central / opt) per component:
  --pair=NAME:cand.json:ref.json  paired in-process census (optime_split_v3 files in platfit/): user-CPU % change -> platform kernel at
        ratio 0.57 / 0.75 / 1.0 (push5p6 calibration: local user +5.3% vs platform +3.0/+5.0%); op-call change x 0.22 ms; bill = flops %.
  --census=NAME:flops:cpu:rt      paired server census (fractions; rt per net): kernel = GEMM-only (28% time / 84% flops) / flops-
        proportional / server CPU; rt x 0.22 ms.
  --item=NAME:bill:pess_s:cent_s:opt_s  paper item in platform seconds (e.g. six-level reads, the exact flag).
  --combo=LABEL:NAME+NAME+...     sums the components per scenario (ADDITIVE: optimistic for overlapping cuts).
  --fit=NAME:bill:lo_s:hi_s       checks whether a paper item of lo..hi platform seconds fits under 100 s on each combo.
usage: project_wall_v2.py [--base=EW,K] --pair=... --census=... --item=... --combo=... --fit=..."""
import json, sys, statistics as st
from pathlib import Path
H = Path(__file__).resolve().parent.parent / "platfit"
args = sys.argv[1:]; get = lambda key: [a.split("=", 1)[1] for a in args if a.startswith(f"--{key}=")]
EW, K = (float(x) for x in (get("base") or ["97.6,84.91"])[0].split(","))
comp = {}
for p in get("pair"):
    name, cf, rf = p.split(":")
    a = json.load(open(H / cf)); b = json.load(open(H / rf))
    du = st.mean(x["user"] / y["user"] - 1 for x, y in zip(a, b)); sd = st.stdev(x["user"] / y["user"] - 1 for x, y in zip(a, b)) / len(a) ** 0.5
    dc = st.mean(sum(v[0] for v in x["ops"].values()) - sum(v[0] for v in y["ops"].values()) for x, y in zip(a, b))
    fl = st.mean(x["flops"] / y["flops"] - 1 for x, y in zip(a, b)); rt = dc * 0.22e-3
    comp[name] = dict(bill=fl, s=[r * du * K + rt for r in (0.57, 0.75, 1.0)],
                      note=f"user {100*du:+.2f}% (se {100*sd:.2f}%, {len(a)} nets), op calls {dc:+.0f}/net")
for c in get("census"):
    name, fl, cpu, rt = c.split(":"); fl, cpu, rt = float(fl), float(cpu), float(rt) * 0.22e-3
    comp[name] = dict(bill=fl, s=[fl / 0.84 * 0.28 * K + rt, fl * K + rt, cpu * K + rt], note=f"server census: flops {100*fl:+.2f}%, CPU {100*cpu:+.1f}%")
for it in get("item"):
    name, b, p, c, o = it.split(":"); comp[name] = dict(bill=float(b), s=[float(p), float(c), float(o)], note="paper item")
def line(label, bill, s, note=""):
    ew = [EW + x for x in s]
    print(f"{label:40s} bill {100*bill:+.2f}%  expected worst {ew[0]:.1f} / {ew[1]:.1f} / {ew[2]:.1f} s  room {100-ew[0]:.1f} / {100-ew[1]:.1f} / {100-ew[2]:.1f} s  {note}")
    return ew
print(f"base: expected worst {EW} s, kernel {K} s; scenarios pess / central / opt")
for n, v in comp.items():
    line(n, v["bill"], v["s"], v["note"])
fits = [f.split(":") for f in get("fit")]
for cb in get("combo"):
    label, parts = cb.split(":"); parts = parts.split("+")
    bill = 1.0
    for p in parts: bill *= 1 + comp[p]["bill"]
    s = [sum(comp[p]["s"][i] for p in parts) for i in range(3)]
    ew = line(label, bill - 1, s, "(ADDITIVE)" if len(parts) > 1 else "")
    for fname, fb, lo, hi in fits:
        lo, hi = float(lo), float(hi)
        verdict = ["fits" if e + hi <= 100 else ("fits at the low end" if e + lo <= 100 else "does NOT fit") for e in ew]
        print(f"    + {fname} (bill {100*float(fb):+.2f}%, +{lo}..+{hi} s): {ew[0]+lo:.1f}-{ew[0]+hi:.1f} / {ew[1]+lo:.1f}-{ew[1]+hi:.1f} / {ew[2]+lo:.1f}-{ew[2]+hi:.1f} s -> "
              + " / ".join(verdict))
