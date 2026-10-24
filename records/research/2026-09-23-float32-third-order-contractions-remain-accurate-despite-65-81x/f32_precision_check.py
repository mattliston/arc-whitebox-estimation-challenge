#!/usr/bin/env python3
"""TAG fable-engineer (E-29b): float32 cancellation precision of the f32 'third' patch, measured in situ. Wraps omitted_radial_source
and, on the same float32 inputs, recomputes in float64 (plain numpy, outside the budget) the diagonal sum(pp^3 lam) and the
omitted-radial contraction; logs per call the relative error of the float32 results and the cancellation ratio of each sum.
usage: f32_precision_check.py <candidate.py> <rank>"""
import sys, json, types
from pathlib import Path
import numpy as np
import flopscope as flops
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand = Path(sys.argv[1]).resolve(); rank = int(sys.argv[2])
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
mod = types.ModuleType("f32prec_mod"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), "exec"), mod.__dict__)
orig = mod.omitted_radial_source; rows = []
def wrapped(pp, lam, diagonal, omitted, alive, aux, host):
    res = orig(pp, lam, diagonal, omitted, alive, aux, host)
    P = np.asarray(pp, dtype=np.float64); lam64 = np.asarray(lam, dtype=np.float64)
    terms = P * P * P * lam64
    d_ref = terms.sum(axis=1); d32 = np.asarray(diagonal, dtype=np.float64)
    g = lambda x: np.asarray(x, dtype=np.float64)
    a = (g(aux["gate"]) - g(host) * g(aux["pdf"]) / g(aux["sigma"])) * g(omitted)
    beta = 2 * (g(aux["gaussian_mean"]) - g(host) * g(aux["gate"])) * g(alive)
    row_energy = (P * P).T @ a; column_sum = P.T @ beta
    parts = lam64 * row_energy * column_sum; self_term = d_ref * a * beta
    ref = 6 * (parts.sum() - self_term.sum())
    rows.append(dict(cols=int(P.shape[1]),
                     diag_relerr=float(np.abs(d32 - d_ref).max() / max(np.abs(d_ref).max(), 1e-300)),
                     diag_cancel=float(np.abs(terms).sum(axis=1).max() / max(np.abs(d_ref).max(), 1e-300)),
                     result_relerr=float(abs(float(res) - ref) / max(abs(ref), 1e-300)),
                     result_cancel=float((np.abs(parts).sum() + np.abs(self_term).sum()) / max(abs(parts.sum() - self_term.sum()), 1e-300))))
    return res
mod.omitted_radial_source = wrapped
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
class MLP:
    def __init__(self, weights): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, f"prec-{rank}"
with flops.BudgetContext(flop_budget=2**41, quiet=True):
    mod.Estimator().predict(MLP([W[l] for l in range(16)]), 2**41)
for i, r in enumerate(rows):
    print(f"call {i:2d}: cols {r['cols']:5d}  diagonal relerr {r['diag_relerr']:.1e} (cancellation x{r['diag_cancel']:.1f})  "
          f"contraction relerr {r['result_relerr']:.1e} (cancellation x{r['result_cancel']:.1f})")
