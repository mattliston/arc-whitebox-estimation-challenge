#!/usr/bin/env python3
"""TAG fable-engineer2 (E2-LR1, low-rank tolerance screen step 2): per-layer energy spectra of the operands the E-62 recipe would
factor, in process on a candidate (learned ON, no refit), one net. Classes (S21 census by caller, net 100):
  bank    : the bank block pushed by _push_bank_shared_newborn (p[:, :old_columns], or p) -- newborn bank push, 22.9% of the bill
  base    : base[previous, :] of the same call (the W5 newborn_common right operand) -- 3.6%
  oldtier : the old-tier reconstruction U F returned by _slr_reconstruct (rank <= 384 by construction) -- old-tier products ~23%
  star_s1 : pw = pushed o w^T, the left operand of the star's dense product s1 = pw w (second pair of _mm_batch) -- ~4.4%
Per captured matrix: shape, Frobenius norm, energy kept by the top r singular values E(r) = sum_{i<=r} s_i^2 / sum s_i^2 for r =
32, 64, 128, 256 (numpy float64 SVD, measurement only; the network runs unchanged).  usage: lr_spectra.py <cand.py> <rank> <tag>"""
import sys, json, types, time
from pathlib import Path
import numpy as np, flopscope as flops, flopscope.numpy as fnp
HERE = Path(__file__).resolve().parent; P2 = HERE.parent.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand, rank, tag = Path(sys.argv[1]).resolve(), int(sys.argv[2]), sys.argv[3]
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
mod = types.ModuleType("lr_mod"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), "exec"), mod.__dict__)
REC = []; RS = (32, 64, 128, 256)
def spec(cls, X, **extra):
    A = np.asarray(X, dtype=np.float64)
    if A.ndim != 2 or min(A.shape) == 0: return
    s = np.linalg.svd(A, compute_uv=False); e = s ** 2; tot = float(e.sum())
    REC.append(dict(cls=cls, shape=list(A.shape), fro=float(np.sqrt(tot)), E={r: (float(e[:r].sum() / tot) if tot > 0 else 1.0) for r in RS},
                    rank99=int(np.searchsorted(np.cumsum(e) / max(tot, 1e-300), 0.99) + 1), **extra))
_push = mod._push_bank_shared_newborn
def push(w, p, previous, newborn):
    if newborn is None: spec("bank", p)
    else:
        old_columns, selected, base, t, s = newborn
        spec("bank", p[:, :old_columns]); spec("base", base[previous, :])
    return _push(w, p, previous, newborn)
mod._push_bank_shared_newborn = push
_rec = mod._slr_reconstruct
def recon(U, state):
    out = _rec(U, state); spec("oldtier", out, stored_rank=int(U.shape[1])); return out
mod._slr_reconstruct = recon
_mb = mod._mm_batch
def mm_batch(pairs):
    if len(pairs) == 2 and tuple(pairs[1][0].shape) == (1024, 1024): spec("star_s1", pairs[1][0])
    return _mb(pairs)
mod._mm_batch = mm_batch
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy"); t0 = time.time()
with flops.BudgetContext(flop_budget=2**41, quiet=True):
    mod.Estimator().predict(MLP([W[l] for l in range(16)], f"lr-{rank}"), 2**41)
json.dump(dict(cand=cand.name, rank=rank, records=REC), open(HERE / f"lr_spectra_{tag}.json", "w"))
for cls in ("bank", "base", "oldtier", "star_s1"):
    rows = [r for r in REC if r["cls"] == cls]
    if not rows: print(f"{cls}: none"); continue
    print(f"{cls}: {len(rows)} captures; shapes {[tuple(r['shape']) for r in rows][:3]}...; rank for 99% energy per call {[r['rank99'] for r in rows]}")
    for r_ in RS:
        v = [r["E"][r_] for r in rows]; print(f"   E({r_}) min {min(v):.4f} median {sorted(v)[len(v)//2]:.4f} max {max(v):.4f}")
print(f"DONE {tag} wall {time.time()-t0:.0f} s")
