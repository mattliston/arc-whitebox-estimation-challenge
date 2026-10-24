#!/usr/bin/env python3
"""TAG fable-refit (RF-22): accuracy impact of dropping DEAD rows (gate < 0.01, i.e. not alive) from the cube-column bank pp at each layer
(old reconstruct + young push are computed for all 1024 output units, but dead units use them only for their own k3 diagonal and
nothing downstream: p = g*pp*alive). Masking test (no FLOP saving yet): pp[dead rows] = 0 after the push; modes: 'old' (old-tier
columns only), 'all' (old + young). usage: deadrow_test.py <cand> <ranks> <modes>"""
import sys, json, types, time
from pathlib import Path
import numpy as np, flopscope as flops, flopscope.numpy as fnp
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand = Path(sys.argv[1]).resolve(); ranks = [int(x) for x in sys.argv[2].split(",")]; modes = sys.argv[3].split(",")
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
src0 = cand.read_text()
H = "            pp = _shared_old384_push(l, w32, p, previous, newborn, lam, shared_old)\n"
i = src0.rfind(H); assert i > 0
src = src0[:i] + H + "            pp = _deadrow_mask(pp, alive, shared_old)\n" + src0[i + len(H):]
src += '''

_DEAD_MODE = 'none'
def _deadrow_mask(pp, alive, state):
    if _DEAD_MODE == 'none':
        return pp
    a = np.asarray(alive.tolist() if hasattr(alive, 'tolist') else alive).astype(np.float32)
    K = int(pp.shape[1]); oc = int(state['columns'])
    colmask = np.ones(K, dtype=np.float32)
    rows = fnp.asarray(a)[:, None]
    if _DEAD_MODE == 'all':
        return pp * rows
    # old only: rows masked on the first oc columns
    keep = np.ones((1, K), dtype=np.float32); keep[0, :oc] = 0.0
    return pp * (fnp.asarray(keep) + rows * fnp.asarray(1.0 - keep))
'''
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
mod = types.ModuleType("deadrow_mod"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod; mod.np = np
exec(compile(src, str(cand), "exec"), mod.__dict__); est = mod.Estimator()
for rank in ranks:
    W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
    truth = np.load(P2 / "public_g22_operator_features_v1/truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float64)
    base = None
    for mode in ['none'] + modes:
        mod._DEAD_MODE = mode; t0 = time.time()
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            pred = np.asarray(est.predict(MLP([W[l] for l in range(16)], f"dr-{rank}"), 2**41), dtype=np.float64)
        mse = float(((pred[15] - truth[15]) ** 2).mean())
        if base is None: base = (mse, pred)
        print(json.dumps(dict(mode=mode, rank=rank, mse=mse, ratio=round(mse / base[0], 5), pred_rms_change=float(np.sqrt(np.mean((pred[15] - base[1][15]) ** 2))), wall=round(time.time() - t0, 1))), flush=True)
