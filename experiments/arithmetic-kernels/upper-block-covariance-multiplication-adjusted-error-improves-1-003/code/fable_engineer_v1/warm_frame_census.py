#!/usr/bin/env python3
"""TAG fable-engineer (E-44, from frame_census.py E-42): which frames do WARM nets bind (nets back to back in ONE process, first cold), and what does each cost? For pricing view re-keying.
Every billed binding reshape (_exact_view, _read5_view) is attributed to the innermost frame builder active (_a_frame, _b_frame,
_o_frame, _bh_o_frame, _bb_o_frame, _read5_child_frame, _read5_packet_frames, _bb_level1) and its key (layout, dims). Also records
the cold net's other one-time costs: the first-predict bill vs the same net's second predict in the same process (warm repeat).
usage: warm_frame_census.py <candidate.py> <ranks> <tag>   -> fable_engineer_v1/warm_frame_census_<tag>.json"""
import sys, json, types, collections
from pathlib import Path
import numpy as np
import flopscope as flops
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand = Path(sys.argv[1]).resolve(); tag = sys.argv[3]
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
mod = types.ModuleType("frame_census_mod"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), "exec"), mod.__dict__)
ctx_ref = [None]; stack = []; acc = collections.defaultdict(lambda: [0, 0])
def layout(views):
    for nm in ("_SCRATCH_PUSH5", "_SCRATCH_PUSH6"):
        if views is getattr(mod, nm, None): return nm[9:]
    S = mod._SCRATCH
    if S is not None:
        for i, nm in ((1, "PUSH"), (2, "READ"), (3, "SQ3"), (4, "SQ4"), (5, "SQ5"), (6, "SQ6")):
            if views is S[i]: return nm
    for P, v in getattr(mod, "_BB_VIEWS", {}).items():
        if views is v: return f"BB{P}"
    return "other"
def builder(name, key_of):
    orig = getattr(mod, name)
    def w(*a, **k):
        stack.append((name, key_of(a)))
        try:
            return orig(*a, **k)
        finally:
            stack.pop()
    setattr(mod, name, w)
builder("_a_frame", lambda a: (layout(a[0]), a[1], a[2], a[3]))
builder("_b_frame", lambda a: (layout(a[0]), a[1], a[2], a[3]))
builder("_o_frame", lambda a: (layout(a[0]), a[1], a[2], a[3]))
builder("_bh_o_frame", lambda a: (layout(a[0]), a[1], a[2], a[3]))
builder("_bb_o_frame", lambda a: (layout(a[0]), a[1], a[2], a[3]))
builder("_read5_child_frame", lambda a: ("READ5", a[1], a[2], a[3], a[4]))
builder("_read5_packet_frames", lambda a: (layout(a[3]), a[0], a[1], a[2]))
builder("_bb_level1", lambda a: (layout(a[0]), a[1]))
def billed(name):
    orig = getattr(mod, name)
    def w(*a, **k):
        b0 = ctx_ref[0]._flops_used
        r = orig(*a, **k)
        top = stack[-1] if stack else ("-", ())
        v = acc[(top[0], top[1])]; v[0] += 1; v[1] += ctx_ref[0]._flops_used - b0
        return r
    setattr(mod, name, w)
billed("_exact_view"); billed("_read5_view")
ranks = [int(x) for x in sys.argv[2].split(",")]
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
est = mod.Estimator(); out = []
for r in ranks:
    W = np.load(STORE / Path(man[r]["path"]).parent / "weights.npy")
    acc.clear()
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        ctx_ref[0] = ctx
        est.predict(MLP([W[l] for l in range(16)], f"wfc-{r}"), 2**41)
    bill = ctx.summary_dict()["flops_used"]
    frames = sorted(((list(k[1]), k[0], v[0], v[1]) for k, v in acc.items()), key=lambda x: -x[3])
    rec = dict(rank=r, bill=bill, binding_pct=round(sum(f[3] for f in frames) / bill * 100, 3), frames=frames)
    out.append(rec)
    print(json.dumps({k: v for k, v in rec.items() if k != "frames"}), flush=True)
json.dump(out, open(HERE / f"warm_frame_census_{tag}.json", "w"), indent=0)
