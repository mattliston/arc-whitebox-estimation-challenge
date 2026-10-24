#!/usr/bin/env python3
"""TAG fable-engineer (E-53): client/server probe at the production boundary (numpy-free client, run with FLOPSCOPE_SERVER_URL set and
E2_SERVER_PID = the server pid). Mode 'prod': builds the candidate's arena and runs single products through _mm on typical shapes
(push packet, rebuild packet, five-level push, read, square), printing the server RSS after each call. Mode 'nets': runs full nets
and writes the raw prediction bytes to <outdir>/<tag>_r<rank>.bin (+ .json with shape and dtype) for bit-identity checks against
in-process runs, printing server RSS after each net. usage: cs_probe.py <cand.py> prod|nets <ranks> <tag> <outdir>"""
import sys, json, types, importlib.util, os, subprocess, time
from pathlib import Path
import flopscope as flops, flopscope.numpy as fnp
cand, mode, ranks, tag, outdir = Path(sys.argv[1]).resolve(), sys.argv[2], [int(x) for x in sys.argv[3].split(",")], sys.argv[4], Path(sys.argv[5])
outdir.mkdir(parents=True, exist_ok=True)
SPID = os.environ["E2_SERVER_PID"]
def rss_mb():
    return int(subprocess.run(["ps", "-o", "rss=", "-p", SPID], capture_output=True, text=True).stdout.strip() or 0) / 1024
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
spec = importlib.util.spec_from_file_location("cs_candidate", cand); mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
if mode == "prod":
    with flops.BudgetContext(flop_budget=2**50, quiet=True) as ctx:
        arena = fnp.empty((mod._PACKED_ELEMENTS,), dtype=fnp.float32)
        mod._SCRATCH = [arena, mod._cp3_bind_arena_views(arena, mod._PACKED_PUSH), mod._cp3_bind_arena_views(arena, mod._PACKED_READ), None, None, None, None]
        mod._bind_push6(arena)
        print(json.dumps(dict(tag=tag, step="arena", rss_mb=round(rss_mb()))), flush=True)
        rng = fnp.random.default_rng(0)
        for name, m, k, n, reps in (("push6 packet", 1024, 896, 4096, 3), ("push6 wide (colpack pair)", 1024, 896, 6144, 2), ("rebuild", 1024, 384, 4096, 2),
                                    ("push5", 896, 864, 4096, 3), ("read", 1024, 8192, 1024, 2), ("square", 1024, 1024, 1024, 2)):
            a = rng.standard_normal((m, k), dtype=fnp.float32); b = rng.standard_normal((k, n), dtype=fnp.float32)
            for rep in range(reps):
                f0 = int(ctx.summary_dict()["flops_used"]); t0 = time.time()
                c = mod._mm(a, b)
                s = float(fnp.sum(c)); dt = time.time() - t0
                print(json.dumps(dict(tag=tag, step=name, rep=rep, flops=int(ctx.summary_dict()["flops_used"]) - f0, wall=round(dt, 2), rss_mb=round(rss_mb()), checksum=s)), flush=True)
                del c
else:
    man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
    class MLP:
        def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
    est = mod.Estimator()
    for rank in ranks:
        with flops.BudgetContext(flop_budget=2**41, quiet=True, namespace=f"{tag}-cs-{rank}") as ctx:
            blob = fnp.load(str(STORE / Path(man[rank]["path"]).parent / "weights.npy")); weights = [blob[l] for l in range(16)]
            t0 = time.time()
            out = est.predict(MLP(weights, f"cs-{rank}"), 2**41)
            raw, shape, dtype = out._fetch_data()
            wall = time.time() - t0
        (outdir / f"{tag}_r{rank}.bin").write_bytes(bytes(raw))
        (outdir / f"{tag}_r{rank}.json").write_text(json.dumps(dict(shape=list(shape), dtype=str(dtype))))
        print(json.dumps(dict(tag=tag, rank=rank, flops=int(ctx.summary_dict()["flops_used"]), wall=round(wall, 2), rss_mb=round(rss_mb()))), flush=True)
