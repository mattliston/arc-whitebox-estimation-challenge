#!/usr/bin/env python3
"""TAG fable-wall-repair (W-H): native single-thread timing of every distinct packed-product shape of the card (from the E census) at
each depth its arena layouts support: squares (max dim <= 1024) at 0/3/4/5, reads (k > 1024) at 0/4, pushes (m = 1024, n > 1024) at
0/4/6 (6 = the batched kernel, n > 4096 as two column blocks). Warm timing (frame already bound) = median of 3; also the bill of the
kernel per (shape, depth) from the flopscope meter.  usage: bench_depths.py <candidate.py (variant F or later)>  -> bench_depths.json"""
import sys, json, time, types, importlib.util, statistics
from pathlib import Path
import flopscope as flops, flopscope.numpy as fnp
import contextlib
HERE = Path(__file__).resolve().parent
cand = Path(sys.argv[1])
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
spec = importlib.util.spec_from_file_location("bench_cand", cand); mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
rows = json.load(open(HERE / "census_variantE_contig_cached_r35.json"))
shapes = sorted({(r["m"], r["k"], r["n"]) for r in rows})
out = {}
with contextlib.nullcontext():
    with flops.BudgetContext(flop_budget=2**45, quiet=True, namespace='bench') as ctx:
        arena = fnp.empty((mod._PACKED_ELEMENTS,), dtype=fnp.float32)
        mod._SCRATCH = [arena, mod._cp3_bind_arena_views(arena, mod._PACKED_PUSH), mod._cp3_bind_arena_views(arena, mod._PACKED_READ), None, None, None, None]
        mod._p1_d6_832_bind(arena)
        for d in (3, 4, 5):
            mod._SCRATCH[d] = mod._cp3_bind_arena_views(arena, {3: mod._PACKED_SQUARE, 4: mod._PACKED_SQUARE4, 5: mod._PACKED_SQUARE5}[d])
        fnp.random.seed(0)
        for (m, k, n) in shapes:
            a = fnp.random.standard_normal((m, k)).astype(fnp.float32); b = fnp.random.standard_normal((k, n)).astype(fnp.float32)
            kind = "square" if max(m, k, n) <= 1024 else "push" if n > 1024 else "read"   # = the _mm dispatch
            depths = (0, 3, 4, 5) if kind == "square" else (0, 4) if kind == "read" else ((0, 4, 6) if (m == 1024 and m % 64 == k % 64 == n % 64 == 0) else (0, 4))
            rec = {}
            for d in depths:
                if d and (m % 2**d or k % 2**d or n % 2**d):
                    continue
                def run():
                    if d == 0:
                        return a @ b
                    o = fnp.empty((m, n), dtype=fnp.float32)
                    if kind == "push":
                        if d == 6:
                            return mod._packed_product_4(a, b, o, mod._SCRATCH[1])   # the D6 branch (batched kernel)
                        return mod._packed_product_cached(a, b, o, mod._SCRATCH[1], 4)
                    if kind == "read":
                        return mod._packed_product_cached(a, b, o, mod._SCRATCH[2], 4)
                    return mod._packed_product_cached(a, b, o, mod._SCRATCH[d], d)
                t0 = time.perf_counter(); f0 = ctx.summary_dict()["flops_used"]; r0 = run(); tb = time.perf_counter() - t0; fb = ctx.summary_dict()["flops_used"] - f0   # cold (binds the frame)
                ts = []
                for _ in range(3):
                    t0 = time.perf_counter(); f0 = ctx.summary_dict()["flops_used"]; r1 = run(); ts.append(time.perf_counter() - t0); fw = ctx.summary_dict()["flops_used"] - f0
                rec[d] = dict(warm_s=statistics.median(ts), cold_s=tb, bill=int(fw), bind_bill=int(fb - fw))
            out[f"{m},{k},{n}"] = dict(kind=kind, depths=rec)
            print(f"{kind:6s} {m:5d} {k:5d} {n:5d}: " + " | ".join(f"D{d} {v['warm_s']*1000:7.1f} ms bill {v['bill']:.3e}" for d, v in rec.items()), flush=True)
json.dump(out, open(HERE / "bench_depths.json", "w"), indent=1)
print("wrote bench_depths.json", len(out), "shapes")
