#!/usr/bin/env python3
"""TAG fable-engineer (E-26): TIME profile of one candidate's server-side numpy work, in-process with the SERVER's numpy/OpenBLAS
(run with astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python, 1 BLAS thread). Every numpy call flopscope makes goes
through _call_numpy; this wraps it in all 11 flopscope modules, times the call, and attributes it to the innermost candidate
function and the numpy callable. Nets run in ONE process (first cold, later warm).
usage: time_profile.py <candidate.py> <ranks e.g. 35,100> <tag>   -> fable_engineer_v1/time_profile_<tag>.json"""
import sys, json, types, time, collections
from pathlib import Path
import numpy as np
import flopscope as flops, flopscope._budget as fb
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand = Path(sys.argv[1]).resolve(); ranks = [int(x) for x in sys.argv[2].split(",")]; tag = sys.argv[3]
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
mod = types.ModuleType("time_profile_mod"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), "exec"), mod.__dict__)
cand_name = str(cand)
orig = fb._call_numpy
acc = collections.defaultdict(lambda: [0, 0.0])       # (func, callable) -> [calls, seconds]
def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    try:
        return orig(fn, *args, **kwargs)
    finally:
        dt = time.perf_counter() - t0
        f = sys._getframe(1); site = "-"
        while f is not None:
            if f.f_code.co_filename == cand_name:
                site = f.f_code.co_name; break
            f = f.f_back
        name = getattr(fn, "__name__", None) or type(fn).__name__
        a = acc[(site, name)]; a[0] += 1; a[1] += dt
for m in list(sys.modules):
    if m.startswith("flopscope") and getattr(sys.modules[m], "_call_numpy", None) is orig:
        sys.modules[m]._call_numpy = timed
CLASS = {"matmul": "gemm", "dot": "gemm", "einsum": "einsum", "tensordot": "gemm",
         "add": "add/sub", "subtract": "add/sub", "stack": "copy", "copyto": "copy", "concatenate": "copy", "copy": "copy",
         "astype": "copy", "reshape": "view", "multiply": "elementwise", "true_divide": "elementwise", "power": "elementwise",
         "sqrt": "elementwise", "exp": "elementwise", "log": "elementwise", "maximum": "elementwise", "minimum": "elementwise",
         "negative": "elementwise", "tanh": "elementwise", "clip": "elementwise", "abs": "elementwise", "sum": "reduction",
         "mean": "reduction", "cumprod": "reduction", "trace": "reduction", "max": "reduction", "argsort": "sort", "nonzero": "sort"}
KERNEL = {"_packed_product_cached", "_bh_forward", "_bh_leaf", "_bh_backward", "_read5_leaf_packets", "_packed_read5",
          "_w5_port_product_body", "_bh_o_frame", "_bh_fwd", "_bh_leafx", "_exact_view", "_a_frame", "_b_frame", "_o_frame",
          "_read5_packet_frames", "_read5_child_frame", "_read5_view", "_packed_frame", "_cp3_bind_arena_views", "_mm",
          "_square_product", "_packed_product_4", "_packed_product_5", "_packed_product_3", "_w5_port_product", "_shared_old384_mm"}
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
est = mod.Estimator(); out = []
for rank in ranks:
    W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
    acc.clear(); t0 = time.perf_counter(); c0 = time.process_time()
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        pred = np.asarray(est.predict(MLP([W[l] for l in range(16)], f"tp-{rank}"), 2**41), dtype=np.float64)
    wall = time.perf_counter() - t0; cpu = time.process_time() - c0
    tot = sum(v[1] for v in acc.values())
    by_class = collections.Counter(); by_part = collections.Counter()
    for (site, name), (n, s) in acc.items():
        by_class[CLASS.get(name, "other:" + name)] += s
        by_part["kernel" if site in KERNEL else "body"] += s
    rec = dict(rank=rank, wall_s=round(wall, 2), process_cpu_s=round(cpu, 2), numpy_s=round(tot, 2),
               by_part={k: round(v, 2) for k, v in by_part.items()},
               by_class={k: round(v, 2) for k, v in by_class.most_common()},
               top=[(site, name, n, round(s, 3)) for (site, name), (n, s) in sorted(acc.items(), key=lambda kv: -kv[1][1])[:45]],
               util=round(ctx.summary_dict()["flops_used"] / 2**41, 5))
    out.append(rec)
    print(json.dumps({k: v for k, v in rec.items() if k != "top"}), flush=True)
    for site, name, n, s in rec["top"][:30]:
        print(f"   {s:7.3f} s  {n:6d} calls  {site:28s} {name}")
    np.save(HERE / f"time_profile_{tag}_pred_r{rank}.npy", pred)
json.dump(out, open(HERE / f"time_profile_{tag}.json", "w"), indent=1)
