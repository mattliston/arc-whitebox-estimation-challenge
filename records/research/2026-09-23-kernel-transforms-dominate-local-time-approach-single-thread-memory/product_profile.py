#!/usr/bin/env python3
"""TAG fable-engineer (E-27): per-CALL-SITE and per-PRODUCT time profile of one candidate, in-process with the SERVER's numpy/OpenBLAS
(run with astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python, 1 BLAS thread). Every numpy call flopscope makes goes
through _call_numpy; this wraps it (as time_profile.py) and attributes its seconds three ways:
  sites     innermost candidate frame 'function:line' x numpy callable (kernel vs body by function name)
  products  innermost active _mm call; key = (kind, m, k, n, depth, path) from the depth decision and kernel entry it made
            (tiling parents keep only their glue); bill per product from the budget context
  callers   first body frame above the product ('which body line asked for this product')
Nets run in ONE process (first cold, later warm). No candidate code is changed.
usage: product_profile.py <candidate.py> <ranks e.g. 35,100> <tag>   -> fable_engineer_v1/product_profile_<tag>.json"""
import sys, json, types, time, collections
from pathlib import Path
import numpy as np
import flopscope as flops, flopscope._budget as fb
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand = Path(sys.argv[1]).resolve(); ranks = [int(x) for x in sys.argv[2].split(",")]; tag = sys.argv[3]
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
mod = types.ModuleType("product_profile_mod"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), "exec"), mod.__dict__)
cand_name = str(cand)
KERNEL = {"_packed_product_cached", "_bh_forward", "_bh_leaf", "_bh_backward", "_read5_leaf_packets", "_packed_read5",
          "_w5_port_product_body", "_bh_o_frame", "_bh_fwd", "_bh_leafx", "_exact_view", "_a_frame", "_b_frame", "_o_frame",
          "_read5_packet_frames", "_read5_child_frame", "_read5_view", "_packed_frame", "_cp3_bind_arena_views", "_mm",
          "_square_product", "_packed_product_4", "_packed_product_5", "_packed_product_3", "_packed_product_6", "_w5_port_product",
          "_shared_old384_mm", "_join_rows_mm", "_square_schedule", "_rule_depth", "mm_wrap"}
PLUMB = KERNEL | {"_w5_port_call", "_resident_gram512", "<listcomp>", "<genexpr>"}
site_acc = collections.defaultdict(lambda: [0, 0.0])      # (func:line, callable) -> [calls, s]
caller_acc = collections.defaultdict(lambda: [0, 0.0, 0])  # body caller func:line -> [products, s, bill]
stack = []; prods = []; ctx_ref = [None]

def body_caller(f):
    while f is not None:
        if f.f_code.co_filename == cand_name and f.f_code.co_name not in PLUMB:
            return f"{f.f_code.co_name}:{f.f_lineno}"
        f = f.f_back
    return "-"

orig = fb._call_numpy
def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    try:
        return orig(fn, *args, **kwargs)
    finally:
        dt = time.perf_counter() - t0
        f = sys._getframe(1); site = "-"
        while f is not None:
            if f.f_code.co_filename == cand_name:
                site = f"{f.f_code.co_name}:{f.f_lineno}"; break
            f = f.f_back
        name = getattr(fn, "__name__", None) or type(fn).__name__
        a = site_acc[(site, name)]; a[0] += 1; a[1] += dt
        if stack:
            r = stack[-1]; r["s"] += dt; r["ops"] += 1
            if name == "matmul": r["gemm_s"] += dt
for m in list(sys.modules):
    if m.startswith("flopscope") and getattr(sys.modules[m], "_call_numpy", None) is orig:
        sys.modules[m]._call_numpy = timed

def layout(views):
    S = mod._SCRATCH
    if views is getattr(mod, "_SCRATCH_PUSH6", None): return "PUSH6"
    if S is not None:
        for i, nm in ((1, "PUSH"), (2, "READ"), (3, "SQ3"), (4, "SQ4"), (5, "SQ5"), (6, "SQ6")):
            if len(S) > i and views is S[i]: return nm
    return "other"

orig_mm = mod._mm
def mm_wrap(a, b):
    rec = dict(m=int(a.shape[0]), k=int(a.shape[1]), n=int(b.shape[1]), s=0.0, gemm_s=0.0, ops=0, dec=[], ent=[], child_bill=0,
               bill0=ctx_ref[0]._flops_used, caller=body_caller(sys._getframe(1)))
    if stack: rec["parent"] = True
    stack.append(rec)
    try:
        return orig_mm(a, b)
    finally:
        stack.pop()
        rec["bill"] = ctx_ref[0]._flops_used - rec.pop("bill0")
        if stack: stack[-1]["child_bill"] += rec["bill"]; stack[-1]["tiled"] = True
        prods.append(rec)
mod._mm = mm_wrap
orig_rd = mod._rule_depth
def rd_wrap(kind, m, k, n, maximum=6):
    d = orig_rd(kind, m, k, n, maximum)
    if stack: stack[-1]["dec"].append((kind, d))
    return d
mod._rule_depth = rd_wrap
def entry(name, levels_at):
    f0 = getattr(mod, name)
    def w(*args, **kw):
        if stack:
            lv = args[levels_at] if levels_at is not None and len(args) > levels_at else None
            e = (name, lv, layout(args[3]) if len(args) > 3 else "-")
            if name == "_w5_port_product_body": e = (name, lv, "W5hit" if args[5]._leaf is not None else "W5new")
            stack[-1]["ent"].append(e)
        return f0(*args, **kw)
    setattr(mod, name, w)
entry("_packed_product_cached", 4); entry("_packed_read5", None); entry("_w5_port_product_body", 4)

def key_of(r):
    if r.get("tiled"): return ("tile-glue", r["m"], r["k"], r["n"], "-", "-")
    ent = r["ent"]
    if not ent: return ("plain", r["m"], r["k"], r["n"], 0, "matmul")
    kind = r["dec"][-1][0] if r["dec"] else "?"
    names = sorted(set(e[0].replace("_packed_product_cached", "ppc").replace("_packed_read5", "read5").replace("_w5_port_product_body", "w5") for e in ent))
    lv = sorted(set(e[1] for e in ent if e[1] is not None)); lay = sorted(set(e[2] for e in ent))
    depth = 5 if names == ["read5"] else (lv[0] if len(lv) == 1 else str(lv))
    return (kind, r["m"], r["k"], r["n"], depth, "+".join(names) + "@" + "+".join(lay) + (f"x{len(ent)}" if len(ent) > 1 else ""))

man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
est = mod.Estimator(); out = []
for rank in ranks:
    W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
    site_acc.clear(); prods.clear(); t0 = time.perf_counter(); c0 = time.process_time()
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        ctx_ref[0] = ctx
        pred = np.asarray(est.predict(MLP([W[l] for l in range(16)], f"pp-{rank}"), 2**41), dtype=np.float64)
    wall = time.perf_counter() - t0; cpu = time.process_time() - c0; bill = ctx.summary_dict()["flops_used"]
    tot = sum(v[1] for v in site_acc.values())
    part = collections.Counter(); func = collections.defaultdict(lambda: [0, 0.0])
    for (site, name), (n, s) in site_acc.items():
        fn = site.split(":")[0]; part["kernel" if fn in KERNEL else "body"] += s
        func[fn][0] += n; func[fn][1] += s
    pk = collections.defaultdict(lambda: [0, 0.0, 0.0, 0, 0])   # key -> [count, s, gemm_s, self bill, ops]
    callers = collections.defaultdict(lambda: [0, 0.0, 0])
    for r in prods:
        k = key_of(r); v = pk[k]; v[0] += 1; v[1] += r["s"]; v[2] += r["gemm_s"]; v[3] += r["bill"] - r["child_bill"]; v[4] += r["ops"]
        c = callers[r["caller"]]; c[0] += 0 if r.get("parent") else 1; c[1] += r["s"]; c[2] += r["bill"] - r["child_bill"]
    rec = dict(rank=rank, wall_s=round(wall, 2), process_cpu_s=round(cpu, 2), numpy_s=round(tot, 3), util=round(bill / 2**41, 5),
               by_part={k: round(v, 3) for k, v in part.items()},
               product_s=round(sum(v[1] for v in pk.values()), 3), product_bill_frac=round(sum(v[3] for v in pk.values()) / bill, 4),
               sites=[(s, nm, n, round(t, 4)) for (s, nm), (n, t) in sorted(site_acc.items(), key=lambda kv: -kv[1][1])[:60]],
               funcs=[(f, n, round(t, 4), "kernel" if f in KERNEL else "body") for f, (n, t) in sorted(func.items(), key=lambda kv: -kv[1][1])[:40]],
               products=[(list(k), v[0], round(v[1], 4), round(v[2], 4), int(v[3]), v[4]) for k, v in sorted(pk.items(), key=lambda kv: -kv[1][1])],
               callers=[(c, v[0], round(v[1], 4), int(v[2])) for c, v in sorted(callers.items(), key=lambda kv: -kv[1][1])])
    out.append(rec)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("sites", "funcs", "products", "callers")}), flush=True)
    print("  top products (kind m k n depth path): count, s, gemm s, bill share")
    for k, n, s, g, b, ops in rec["products"][:25]:
        print(f"   {s:7.3f} s  gemm {g:6.3f}  {n:4d}x  {b/bill*100:6.2f}%  {k}")
    print("  top body callers of products: products, s, bill share")
    for c, n, s, b in rec["callers"][:15]:
        print(f"   {s:7.3f} s  {n:4d}  {b/bill*100:6.2f}%  {c}")
    np.save(HERE / f"product_profile_{tag}_pred_r{rank}.npy", pred)
json.dump(out, open(HERE / f"product_profile_{tag}.json", "w"), indent=1)
