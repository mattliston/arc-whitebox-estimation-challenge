#!/usr/bin/env python3
"""TAG fable-wall-repair (W-2): production-boundary timing. Run in the NUMPY-FREE client python with FLOPSCOPE_SERVER_URL set to a running
OpenBLAS flopscope server (1 BLAS thread). Five persistent contexts (35,100,500,3016500,35): per-call wall, flops, residual, output sha; the
repeated rank-35 output must be byte-identical.  usage: remote_series.py <candidate.py abs> <tag>  -> <tag>.json in this dir"""
import argparse, hashlib, importlib.util, json, sys, time, types
from pathlib import Path
import flopscope as flops, flopscope.numpy as fnp
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
ap = argparse.ArgumentParser(); ap.add_argument("candidate"); ap.add_argument("tag"); args = ap.parse_args()
assert importlib.util.find_spec("numpy") is None, "must run numpy-free"
stub = types.ModuleType("whestbench")
class BaseEstimator: pass
stub.BaseEstimator = BaseEstimator; sys.modules["whestbench"] = stub
source = Path(args.candidate); spec = importlib.util.spec_from_file_location("remote_series_candidate", source)
mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
estimator = mod.Estimator(); records = []; t_all = time.time()
for index, rank in enumerate((35, 100, 500, 3016500, 35)):
    fixture = STORE / Path(man[rank]["path"]).parent / "weights.npy" if rank < 1000 else P2 / "public_g19_attribution_v1" / f"fresh_{rank}_weights.npy"
    assert fixture.is_file(), fixture
    t0 = time.time()
    with flops.BudgetContext(flop_budget=2**41, quiet=True, namespace=f"{args.tag}-{index}") as ctx:
        blob = fnp.load(str(fixture)); weights = [blob[l] for l in range(16)]
        out = estimator.predict(MLP(weights, f"remote-{rank}"), 2**41)
        raw, shape, dtype = out._fetch_data()
    wall = time.time() - t0; s = ctx.summary_dict()
    rec = dict(index=index, rank=rank, wall_s=round(wall, 3), flops=int(s["flops_used"]), util=round(int(s["flops_used"]) / 2**41, 5),
               timing={k: v for k, v in s.items() if "time" in k}, output_sha256=hashlib.sha256(raw).hexdigest(), shape=list(shape))
    records.append(rec); print(json.dumps(rec), flush=True)
ok = records[0]["output_sha256"] == records[-1]["output_sha256"]
res = dict(candidate=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), records=records, repeat_byte_identical=ok, total_wall_s=round(time.time() - t_all, 1))
(HERE / f"{args.tag}.json").write_text(json.dumps(res, indent=2) + "\n"); print("repeat byte-identical:", ok, "| max wall", max(r["wall_s"] for r in records))
