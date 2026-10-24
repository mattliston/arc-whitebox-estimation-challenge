#!/usr/bin/env python3
"""TAG fable-engineer (E-38): the frame-BINDING bill per net, by binding op, by layout and by frame side. In-process; wraps
_exact_view, _read5_view and _cp3_bind_arena_views and attributes the flops each call charges (the billed reshape) to
(op, layout, side = calling frame builder). Nets run back to back in one process like one platform worker; the first is cold.
usage: bind_census.py <candidate.py> <ranks> <tag>   -> fable_engineer_v1/bind_census_<tag>.json"""
import sys, json, types, collections
from pathlib import Path
import numpy as np
import flopscope as flops
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand = Path(sys.argv[1]).resolve(); ranks = [int(x) for x in sys.argv[2].split(",")]; tag = sys.argv[3]
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
mod = types.ModuleType("bind_census_mod"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), "exec"), mod.__dict__)
ctx_ref = [None]; acc = collections.defaultdict(lambda: [0, 0])
SPEC_NAMES = {}
for nm in dir(mod):
    v = getattr(mod, nm)
    if nm.startswith("_PACKED_") or nm.startswith("_BB_SPEC_"):
        if isinstance(v, tuple) and v and isinstance(v[0], tuple) and len(v[0]) == 3:
            SPEC_NAMES[id(v)] = nm
def spec_layout(spec):
    for nm in dir(mod):
        v = getattr(mod, nm)
        if (nm.startswith("_PACKED_") or nm.startswith("_BB_SPEC_")) and isinstance(v, tuple) and spec in v:
            return nm
    return "?"
LAYOUT_OF_SPEC = {}
def caller():
    f = sys._getframe(2)
    while f is not None:
        n = f.f_code.co_name
        if n not in ("_exact_view", "_read5_view", "wrapped", "<genexpr>", "<listcomp>"):
            return n
        f = f.f_back
    return "-"
def wrap(name, spec_of):
    orig = getattr(mod, name)
    def wrapped(*a, **k):
        before = ctx_ref[0]._flops_used
        r = orig(*a, **k)
        billed = ctx_ref[0]._flops_used - before
        lay = spec_of(a)
        key = (name, lay, caller())
        acc[key][0] += 1; acc[key][1] += billed
        return r
    setattr(mod, name, wrapped)
def ev_spec(a):
    spec = a[1]
    k = (spec[0], spec[1], tuple(spec[2]))
    if k not in LAYOUT_OF_SPEC:
        LAYOUT_OF_SPEC[k] = spec_layout(spec)
    return LAYOUT_OF_SPEC[k]
wrap("_exact_view", ev_spec)
wrap("_read5_view", lambda a: "READ5_CHILD")
wrap("_cp3_bind_arena_views", lambda a: SPEC_NAMES.get(id(a[1]), "?"))
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
est = mod.Estimator(); out = []
for rank in ranks:
    W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
    acc.clear()
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        ctx_ref[0] = ctx
        est.predict(MLP([W[l] for l in range(16)], f"bc-{rank}"), 2**41)
    bill = ctx.summary_dict()["flops_used"]
    tot = sum(v[1] for v in acc.values())
    rows = sorted(((k, v[0], v[1]) for k, v in acc.items()), key=lambda x: -x[2])
    rec = dict(rank=rank, bill=bill, util=round(bill / 2**41, 5), bind_flops=tot, bind_pct=round(tot / bill * 100, 3),
               rows=[(list(k), n, b, round(b / bill * 100, 3)) for k, n, b in rows])
    out.append(rec)
    print(json.dumps({k: v for k, v in rec.items() if k != "rows"}), flush=True)
    for k, n, b, pct in rec["rows"][:18]:
        print(f"   {pct:6.3f}%  {n:5d} calls  {k}")
json.dump(out, open(HERE / f"bind_census_{tag}.json", "w"), indent=1)
