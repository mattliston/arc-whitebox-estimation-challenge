#!/usr/bin/env python3
"""TAG fable-engineer (E-41): symmetric-output census. flopscope bills a symmetric-output einsum (Gram X^T X, weighted Gram X^T D X) at
half the dense cost. This finds every product of a candidate whose OUTPUT is symmetric, and whether its operands form a Gram pair
(a = b^T), a weighted Gram pair (a = b^T diag(d) or b = diag(d) a^T), or neither (e.g. the covariance push w^T (C w)).
In-process, nets back to back in one process (first cold). Wrapped entry points (outermost call only, nested calls are not re-counted):
_mm, _mm_batch (per output, bill split evenly), _shared_old384_mm, _join_rows_mm, _resident_gram512. Bill = budget delta of the call.
Symmetry and operand checks read plain-ndarray views (not billed).
usage: sym_census.py <candidate.py> <ranks e.g. 35,100> <tag>   -> fable_engineer_v1/sym_census_<tag>.json"""
import sys, json, types, collections
from pathlib import Path
import numpy as np
import flopscope as flops
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand = Path(sys.argv[1]).resolve(); ranks = [int(x) for x in sys.argv[2].split(",")]; tag = sys.argv[3]
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
mod = types.ModuleType("sym_census_mod"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), "exec"), mod.__dict__)
cand_name = str(cand)
WRAPPED = ("_mm", "_mm_batch", "_shared_old384_mm", "_join_rows_mm", "_resident_gram512")
PLUMB = set(WRAPPED) | {"_w5_port_call", "<listcomp>", "<genexpr>", "wrapped"}
ctx_ref = [None]; depth = [0]; recs = []

def raw(x):
    return np.asarray(x).view(np.ndarray)

def caller(f):
    while f is not None:
        if f.f_code.co_filename == cand_name and f.f_code.co_name not in PLUMB:
            return f"{f.f_code.co_name}:{f.f_lineno}"
        f = f.f_back
    return "-"

def classify(a, b, out):
    a, b, g = raw(a), raw(b), raw(out)
    m, n = g.shape[0], g.shape[1]
    if m != n:
        return "rect"
    g64 = g.astype(np.float64)
    scale = float(np.abs(g64).max()) or 1.0
    if float(np.abs(g64 - g64.T).max()) > 1e-4 * scale:
        return "square"
    if a.shape == b.T.shape and np.array_equal(a, b.T):
        return "gram"
    if a.shape == b.T.shape:
        bt = b.T.astype(np.float64); a64 = a.astype(np.float64)
        for x, y, axis in ((a64, bt, 0), (b.astype(np.float64), a64.T, 1)):   # a = b^T diag(d)  |  b = diag(d) a^T
            mask = np.abs(y) > 1e-30
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(mask, x / np.where(mask, y, 1.0), np.nan)
            hi, lo = np.nanmax(r, axis=axis), np.nanmin(r, axis=axis)
            ok = np.isfinite(hi) & np.isfinite(lo)
            if ok.any() and np.all(np.abs(hi - lo)[ok] <= 1e-5 * np.maximum(np.abs(hi), np.abs(lo))[ok] + 1e-30):
                return "wgram"
    return "symout"

def wrap(name):
    orig = getattr(mod, name)
    def wrapped(*args, **kw):
        if depth[0]:
            return orig(*args, **kw)
        depth[0] += 1
        where = caller(sys._getframe(1)); b0 = ctx_ref[0]._flops_used
        try:
            out = orig(*args, **kw)
        finally:
            depth[0] -= 1
        bill = ctx_ref[0]._flops_used - b0
        pairs = args[0] if name == "_mm_batch" else [(args[0], args[1])]
        outs = out if name == "_mm_batch" else [out]
        for (a, b), o in zip(pairs, outs):
            recs.append(dict(fn=name, caller=where, m=int(o.shape[0]), k=int(a.shape[-1]), n=int(o.shape[1]),
                             bill=bill / len(outs), cls=classify(a, b, o), dtype=str(o.dtype)))
        return out
    setattr(mod, name, wrapped)
for nm in WRAPPED:
    if hasattr(mod, nm):
        wrap(nm)

man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
est = mod.Estimator(); result = []
for rank in ranks:
    W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
    recs.clear()
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        ctx_ref[0] = ctx
        pred = np.asarray(est.predict(MLP([W[l] for l in range(16)], f"sc-{rank}"), 2**41), dtype=np.float64)
    np.save(HERE / f"sym_census_{tag}_pred_r{rank}.npy", pred)
    bill = ctx.summary_dict()["flops_used"]
    by = collections.defaultdict(lambda: [0, 0.0, set()])
    for r in recs:
        v = by[(r["cls"], r["caller"])]; v[0] += 1; v[1] += r["bill"]; v[2].add((r["m"], r["k"], r["n"]))
    rows = sorted(((k, v[0], v[1], sorted(v[2])[:6]) for k, v in by.items()), key=lambda x: -x[2])
    cls = collections.Counter()
    for r in recs: cls[r["cls"]] += r["bill"]
    rec = dict(rank=rank, bill=bill, util=round(bill / 2**41, 5), covered=round(sum(r["bill"] for r in recs) / bill, 4),
               by_class={k: round(v / bill * 100, 3) for k, v in cls.items()},
               rows=[(list(k), n, int(b), round(b / bill * 100, 3), [list(s) for s in shp]) for k, n, b, shp in rows])
    result.append(rec)
    print(json.dumps({k: v for k, v in rec.items() if k != "rows"}), flush=True)
    for (c, where), n, b, pct, shp in rec["rows"]:
        if c != "rect" and pct >= 0.01:
            print(f"   {pct:6.3f}%  {c:7s} {n:4d}x  {where:34s} {shp}", flush=True)
json.dump(result, open(HERE / f"sym_census_{tag}.json", "w"), indent=1)
