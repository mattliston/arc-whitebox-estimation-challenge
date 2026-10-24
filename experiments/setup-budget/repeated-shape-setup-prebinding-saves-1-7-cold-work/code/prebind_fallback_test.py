#!/usr/bin/env python3
"""TAG fable-engineer (E-61 fallback, lead's S17 condition): force prebind's arena-gone path once. In process: load the fired S17 file
(prebind v2b), call Estimator.setup(), then replace the setup-created arena with an object whose indexing raises (as a lost server
handle would), and predict net 35. Pass = _PB_LOG records the lazy fallback, the predict completes, and its predictions are
byte-identical to the same physics without prebinding (pair_ipt_runs/S17j05base_r35.npy = stack_S17_j05_mem.py in process).
usage: prebind_fallback_test.py <S17 fire file>"""
import sys, json, types
from pathlib import Path
import numpy as np
import flopscope as flops
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand = Path(sys.argv[1]).resolve()
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
mod = types.ModuleType("pft_mod"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), "exec"), mod.__dict__)
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, w, nm): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, w, 0, nm
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version="2", scratch_dir=None, submission_dir=None, seed=0))
print("setup log:", mod._PB_LOG, "cached entries:", sum(len(mod.__dict__[k]) for k in mod._PB_CACHES), "check armed:", mod._PB_CHECK[0], flush=True)
class Gone:
    def __getitem__(self, key):
        raise RuntimeError("simulated: setup-created arena handle not found on the server")
mod._SCRATCH[0] = Gone()
W = np.load(STORE / Path(man[35]["path"]).parent / "weights.npy")
with flops.BudgetContext(flop_budget=2**41, quiet=True, namespace="pft-35") as ctx:
    pred = np.asarray(est.predict(MLP([W[l] for l in range(16)], "pft-35"), 2**41), dtype=np.float32)
ref = np.load(HERE / "pair_ipt_runs" / "S17j05base_r35.npy")
print(json.dumps(dict(pb_log=[list(x) for x in mod._PB_LOG], bill=int(ctx.summary_dict()["flops_used"]),
                      byte_identical_to_S17_without_prebind=pred.tobytes() == ref.tobytes(),
                      cached_entries_after=sum(len(mod.__dict__[k]) for k in mod._PB_CACHES))), flush=True)
