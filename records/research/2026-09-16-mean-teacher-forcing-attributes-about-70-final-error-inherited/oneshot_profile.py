#!/usr/bin/env python3
"""TAG fable-wall-repair (W-L): born-vs-inherited error profile of the card. The learned-head hook is wrapped so that, at every layer,
the one-step error of the card mean (before and after the learned head) is recorded against the organizers' 1e9 truth and the state
mean is then FORCED to the truth (the card's own consistent covariance update absorbs the correction). All other state (covariance,
third-order bank, radial) stays the card's own. Prints per layer: closed-loop MSE (a normal run), one-step MSE card-only, one-step MSE
card+head, and the head's removal fraction.  usage: oneshot_profile.py <cand> <rank>"""
import sys, json, types, importlib.util
from pathlib import Path
import numpy as np, flopscope as flops, flopscope.numpy as fnp
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand, rank = Path(sys.argv[1]).resolve(), int(sys.argv[2])
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
spec = importlib.util.spec_from_file_location("oneshot_cand", cand); mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy"); truth = np.load(P2 / "public_g22_operator_features_v1/truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float64)
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
mlp = MLP([W[l] for l in range(16)], f"oneshot-{rank}")
with flops.BudgetContext(flop_budget=2**41, quiet=True):
    closed = np.asarray(mod.Estimator().predict(mlp, 2**41), dtype=np.float64)
closed_mse = ((closed - truth) ** 2).mean(axis=1)
rec = {}
orig_pb = mod._predict_binary
def forced_pb(mlp_, hook=None, **kw):
    def fhook(l, m, C, K21, aux, w):
        delta = hook(l, m, C, K21, aux, w)
        m64 = np.asarray(m, dtype=np.float64); d64 = np.asarray(delta, dtype=np.float64); t = truth[l]
        rec[l] = dict(card=float(((m64 - t) ** 2).mean()), head=float(((m64 + d64 - t) ** 2).mean()))
        return fnp.asarray(t - m64, dtype=fnp.float64)   # force the state mean to the truth
    return orig_pb(mlp_, hook=fhook, **kw)
mod._predict_binary = forced_pb
with flops.BudgetContext(flop_budget=2**41, quiet=True):
    mod.Estimator().predict(mlp, 2**41)
print(f"{cand.name} r{rank}: per-layer MSE vs 1e9 truth (closed-loop | one-step card-only | one-step card+head | head removal)")
for l in range(16):
    r = rec.get(l)
    if r: print(f"  L{l:2d}: {closed_mse[l]:.2e} | {r['card']:.2e} | {r['head']:.2e} | {100*(1-r['head']/r['card']):5.1f}%")
    else: print(f"  L{l:2d}: {closed_mse[l]:.2e} | (no hook)")
json.dump(dict(closed=closed_mse.tolist(), oneshot=rec), open(HERE / f"oneshot_{cand.stem}_r{rank}.json", "w"), indent=1)
