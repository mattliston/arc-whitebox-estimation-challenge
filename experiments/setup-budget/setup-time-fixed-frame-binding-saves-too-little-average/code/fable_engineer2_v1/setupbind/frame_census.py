#!/usr/bin/env python3
"""TAG fable-engineer2 (E2-FC v2: bill = reshape charges only, reqs = reshape requests per key; v1 attributed product compute to leaf binders): FRAME-SHAPE CENSUS for census-driven setup binding. In-process (server's flopscope + numpy), nets run
back to back like one worker. Every OUTERMOST call of a frame binder is recorded with a key = (binder, layout tag, int args); hits cost 0,
misses carry the bill of the views they bind; nested binders are attributed to the outermost one. Output per net: every key the net
NEEDED with the bill it paid for it (0 when already bound), so a coverage curve (bill avoided vs keys pre-bound in setup) can be computed on held-out nets.
usage: frame_census.py <cand.py> <ranks> <tag>  -> fable_engineer2_v1/setupbind/frame_census_<tag>.json"""
import sys, json, types, time
from pathlib import Path
import numpy as np, flopscope as flops
from flopscope import _budget
HERE = Path(__file__).resolve().parent; P2 = HERE.parent.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand, ranks, tag = Path(sys.argv[1]).resolve(), [int(x) for x in sys.argv[2].split(',')], sys.argv[3]
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
mod = types.ModuleType("frame_census_mod"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), "exec"), mod.__dict__)
BINDERS = ["_bh_fwd", "_bh_o_frame", "_bh_leafx", "_bb_o_frame", "_bb_level1", "_bb_views", "_packed_frame", "_read5_packet_frames",
           "_read5_child_frame", "_read5_leaf_packets", "_a_frame", "_b_frame", "_o_frame", "_ul_mode", "_cp3_bind_arena_views"]
depth = [0]; rec = []; cur = [None]; reshape_by_key = {}; nreq_by_key = {}
_orig_charge = _budget.BudgetContext._charge_op
def _charge(self, op_name, flop_cost, subscripts, shapes, **kw):   # v2: the binding bill = reshape charges, attributed to the key
    b0 = self._flops_used; r = _orig_charge(self, op_name, flop_cost, subscripts, shapes, **kw)
    if op_name == "reshape" and cur[0] is not None:
        reshape_by_key[cur[0]] = reshape_by_key.get(cur[0], 0) + (self._flops_used - b0); nreq_by_key[cur[0]] = nreq_by_key.get(cur[0], 0) + 1
    return r
_budget.BudgetContext._charge_op = _charge
def tag_of(x):
    if isinstance(x, (int, np.integer)): return int(x)
    if isinstance(x, tuple):
        for name in ("_SCRATCH_PUSH5", "_SCRATCH_PUSH6"):
            if x is getattr(mod, name, None): return name
        S = getattr(mod, "_SCRATCH", None)
        if S is not None:
            for i, v in enumerate(S):
                if x is v: return f"_SCRATCH[{i}]"
        for P, v in getattr(mod, "_BB_VIEWS", {}).items():
            if x is v: return f"_BB_VIEWS[{P}]"
        return f"tuple{len(x)}"
    if hasattr(x, "shape"): return "arr" + str(tuple(int(s) for s in x.shape))
    return type(x).__name__
def wrap(name):
    f0 = getattr(mod, name, None)
    if f0 is None: return
    def w(*args, **kw):
        outer = depth[0] == 0
        if outer:
            key = (name,) + tuple(tag_of(a) for a in args); cur[0] = key; reshape_by_key[key] = 0; nreq_by_key[key] = 0
        depth[0] += 1
        try:
            return f0(*args, **kw)
        finally:
            depth[0] -= 1
            if outer:
                rec.append((key, reshape_by_key.pop(key, 0), nreq_by_key.pop(key, 0))); cur[0] = None   # hits (cost 0) too: the net NEEDED this key
    setattr(mod, name, w)
for n in BINDERS: wrap(n)
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, nm): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, nm
est = mod.Estimator(); out = []
for i, rank in enumerate(ranks):
    W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy"); rec.clear(); t0 = time.time()
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        est.predict(MLP([W[l] for l in range(16)], f"fc-{rank}"), 2**41)
    tot = ctx.summary_dict()["flops_used"]
    need = {}; reqs = {}
    for k, c, nq in rec:
        ks = "|".join(map(str, k)); need[ks] = max(need.get(ks, 0), c); reqs[ks] = max(reqs.get(ks, 0), nq)   # cost > 0 only where this net bound the key
    out.append(dict(i=i, rank=rank, flops=tot, bind_flops=sum(need.values()), n_keys=len(need), n_bound=sum(1 for c in need.values() if c > 0),
                    keys=need, reqs=reqs, wall=round(time.time() - t0, 1)))
    print(json.dumps(dict(i=i, rank=rank, util=round(tot / 2**41, 6), bind_pct=round(100 * sum(need.values()) / tot, 3), n_keys=len(need),
                          n_bound=sum(1 for c in need.values() if c > 0))), flush=True)
json.dump(out, open(HERE / f"frame_census_{tag}.json", "w"))
