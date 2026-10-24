#!/usr/bin/env python3
"""TAG fable-wall-repair (W-1): native (numpy-backend) cProfile of a candidate's predict on one net, single BLAS thread, to locate the wall.
usage: profile_native.py <candidate.py> <rank> <tag>   -> <tag>_profile.txt (top functions by cumulative time), wall, flops"""
import sys, time, json, cProfile, pstats, importlib.util, types
from pathlib import Path
import numpy as np, flopscope as flops, flopscope.numpy as fnp
from threadpoolctl import threadpool_limits
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand, rank, tag = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
stub = types.ModuleType("whestbench"); 
class BaseEstimator: pass
stub.BaseEstimator = BaseEstimator; sys.modules["whestbench"] = stub
spec = importlib.util.spec_from_file_location("cand_mod", cand); mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
class MLP:
    def __init__(self, weights): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, f"prof-{rank}"
est = mod.Estimator()
with threadpool_limits(limits=1):
    pr = cProfile.Profile(); t0 = time.time()
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        pr.enable(); out = est.predict(MLP([W[l] for l in range(16)]), 2**41); pr.disable()
    wall = time.time() - t0
s = ctx.summary_dict(); print(f"{tag}: rank {rank} native wall {wall:.1f}s flops {s['flops_used']:.4e} util {s['flops_used']/2**41:.4f} residual {s.get('residual_wall_time_s', float('nan'))}")
ps = pstats.Stats(pr); ps.sort_stats("cumulative")
import io; buf = io.StringIO(); ps.stream = buf; ps.print_stats(40); (HERE / f"{tag}_profile.txt").write_text(buf.getvalue())
buf2 = io.StringIO(); ps.stream = buf2; ps.sort_stats("tottime").print_stats(25); (HERE / f"{tag}_profile_tottime.txt").write_text(buf2.getvalue())
print("profiles written")
