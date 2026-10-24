#!/usr/bin/env python3
"""TAG fable-engineer (E-28): in-process per-product timing at FORCED Winograd depths, for the compute-budgeted depth rule.
Loads a candidate that carries the depth-rule fragment (its _DR_FORCE hook), binds the arena exactly as _predict_binary does, and
times _mm(a, b) on random float32 operands for every (kind, m, k, n) x depth, interleaved over reps (rep-major) so load drift hits
every depth alike. Per cell: median/min wall, numpy seconds (flopscope's _call_numpy), requests, bill, max rel error vs float64.
Run with astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python and 1 BLAS thread.
usage: depth_bench.py <candidate.py> <shapes.json: [[kind,m,k,n,[depths]],...]> <reps> <tag>  -> fable_engineer_v1/depth_bench_<tag>.json"""
import sys, json, types, time, statistics
from pathlib import Path
import numpy as np
import flopscope as flops, flopscope._budget as fb, flopscope.numpy as fnp
HERE = Path(__file__).resolve().parent
cand = Path(sys.argv[1]).resolve(); shapes = json.loads(Path(sys.argv[2]).read_text()); reps = int(sys.argv[3]); tag = sys.argv[4]
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
mod = types.ModuleType("depth_bench_mod"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), "exec"), mod.__dict__)
assert hasattr(mod, "_DR_FORCE"), "candidate lacks the depth-rule fragment"
orig = fb._call_numpy; tally = [0.0, 0, 0.0]   # numpy s, requests, matmul s
def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    try:
        return orig(fn, *args, **kwargs)
    finally:
        dt = time.perf_counter() - t0; tally[0] += dt; tally[1] += 1
        if getattr(fn, "__name__", "") == "matmul": tally[2] += dt
for m in list(sys.modules):
    if m.startswith("flopscope") and getattr(sys.modules[m], "_call_numpy", None) is orig:
        sys.modules[m]._call_numpy = timed
rng = np.random.default_rng(0); out = []
with flops.BudgetContext(flop_budget=2**55, quiet=True) as ctx:
    arena = fnp.empty((mod._PACKED_ELEMENTS,), dtype=fnp.float32)
    mod._SCRATCH = [arena, mod._cp3_bind_arena_views(arena, mod._PACKED_PUSH), mod._cp3_bind_arena_views(arena, mod._PACKED_READ), None, None, None, None]
    mod._bind_push6(arena)
    for kind, m, k, n, depths in shapes:
        a64 = rng.standard_normal((m, k)).astype(np.float32); b64 = rng.standard_normal((k, n)).astype(np.float32)
        ref = a64.astype(np.float64) @ b64.astype(np.float64)
        a = fnp.asarray(a64); b = fnp.asarray(b64)
        cells = {d: dict(wall=[], np_s=[], req=[], bill=[], gemm=[]) for d in depths}; err = {}
        for rep in range(reps):
            for d in depths:
                mod._DR_FORCE.clear(); mod._DR_FORCE[(kind, m, k, n)] = d
                f0 = ctx._flops_used; s0, q0, g0 = tally[0], tally[1], tally[2]; t0 = time.perf_counter()
                r = mod._mm(a, b)
                dt = time.perf_counter() - t0
                c = cells[d]; c["wall"].append(dt); c["np_s"].append(tally[0] - s0); c["req"].append(tally[1] - q0); c["bill"].append(ctx._flops_used - f0); c["gemm"].append(tally[2] - g0)
                if rep == 0:
                    err[d] = float(np.abs(np.asarray(r, dtype=np.float64) - ref).max() / np.abs(ref).max())
                del r
        mod._DR_FORCE.clear()
        for d in depths:
            c = cells[d]; w = c["wall"][1:] if reps > 2 else c["wall"]; s = c["np_s"][1:] if reps > 2 else c["np_s"]
            rec = dict(kind=kind, m=m, k=k, n=n, depth=d, wall_med=round(statistics.median(w), 5), wall_min=round(min(w), 5),
                       np_med=round(statistics.median(s), 5), np_min=round(min(s), 5), gemm_med=round(statistics.median(c["gemm"][1:] if reps > 2 else c["gemm"]), 5), first_wall=round(c["wall"][0], 5),
                       req=c["req"][-1], bill=int(c["bill"][-1]), bill_first=int(c["bill"][0]), relerr=err[d])
            out.append(rec)
            print(json.dumps(rec), flush=True)
json.dump(out, open(HERE / f"depth_bench_{tag}.json", "w"), indent=1)
