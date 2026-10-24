#!/usr/bin/env python3
"""TAG fable-engineer2 (E2-LR2, low-rank tolerance screen step 3): the E-62 recipe (randomized range finder with k = r + 16 columns, a
fixed seeded Gaussian, one power step with re-orthonormalization, then exact truncation of the small factor to rank r) applied to one
operand class of a candidate, in process, learned ON, no refit; final-layer MSE vs the 8M truth (parity.py's truthpost) per net, plus
the bill of the run. Classes:
  none    : the unmodified candidate (baseline arm, same harness)
  oldtier : the old tier's columns O = U F_eff (F_eff = F, or [map base | join] in the lazy layer) truncated to rank r in U's metric:
            G = R_U F_eff (U = Q_U R_U), W = top-r left subspace of G (recipe above), P = R_U^-1 W W^T R_U; BOTH consumers see P F_eff:
            _slr_reconstruct returns U (P F_eff), _cp_slr_inner_sq returns old_left (P F_eff o lambda)^T (P from the same layer's reconstruct;
            an inner call with no P for its state runs unmodified and is counted)
  star_s1 : the star's dense product s1 = pw w (second pair of _mm_batch; the batched pair still runs, so C w is bit-identical) is
            replaced by Q (B w), with Q, B the recipe's rank-r factors of pw
The linear algebra of the replacement is numpy float64 (a tolerance test, not a native build); the bill cut is priced separately.
usage: lr_e62_test.py <cand.py> <class> <r> <ranks> <tag>  -> lowrank/lr_e62_<tag>.json"""
import sys, json, types, time
from pathlib import Path
import numpy as np, flopscope as flops, flopscope.numpy as fnp
HERE = Path(__file__).resolve().parent; P2 = HERE.parent.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand, cls, R, ranks, tag = Path(sys.argv[1]).resolve(), sys.argv[2], int(sys.argv[3]), [int(x) for x in sys.argv[4].split(",")], sys.argv[5]
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
mod = types.ModuleType("lre_mod"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), "exec"), mod.__dict__)
RNG = np.random.default_rng(20260924); STATS = dict(replaced=0, missing_P=0)
def f64(x): return np.asarray(x, dtype=np.float64)
def to32(x): return fnp.asarray(np.ascontiguousarray(x, dtype=np.float32))
def range_left(G, r):
    """E-62 recipe on G (rows x cols): orthonormal basis (rows x r) of G's approximate top-r left subspace."""
    k = min(r + 16, min(G.shape))
    Om = RNG.standard_normal((G.shape[1], k))
    Q = np.linalg.qr(G @ Om)[0]
    Q = np.linalg.qr(G @ np.linalg.qr(G.T @ Q)[0])[0]          # one power step with re-orthonormalization
    Ub = np.linalg.svd(Q.T @ G, full_matrices=False)[0]          # small factor B = Q^T G, exact truncation to r
    return Q @ Ub[:, :min(r, Ub.shape[1])]
if cls == "oldtier":
    PCACHE = {}
    def f_eff(state):
        g = state['lazy']
        return f64(state['F']) if g is None else np.concatenate([f64(g['map']) @ f64(g['base']), f64(g['join'])], axis=1)
    _rec, _inn = mod._slr_reconstruct, mod._cp_slr_inner_sq
    def recon(U, state):
        F = f_eff(state); Uq, Ur = np.linalg.qr(f64(U)); W = range_left(Ur @ F, R)
        P = np.linalg.solve(Ur, W @ (W.T @ Ur)); PF = P @ F
        PCACHE[id(state)] = (int(state['columns']), PF); STATS["replaced"] += 1
        return mod._shared_old384_mm(U, to32(PF))
    def inner(old_left, lam_old, state):
        got = PCACHE.get(id(state))
        if got is None or got[0] != int(state['columns']):
            STATS["missing_P"] += 1; return _inn(old_left, lam_old, state)
        STATS["replaced"] += 1
        return mod._shared_old384_mm(old_left, to32((got[1] * f64(lam_old)[None, :]).T))
    mod._slr_reconstruct = recon; mod._cp_slr_inner_sq = inner
elif cls == "star_s1":
    _mb = mod._mm_batch
    def mm_batch(pairs):
        outs = _mb(pairs)
        if len(pairs) == 2 and tuple(pairs[1][0].shape) == (1024, 1024):
            pw, w = f64(pairs[1][0]), f64(pairs[1][1]); Q = range_left(pw, R)
            outs = [outs[0], to32(Q @ ((Q.T @ pw) @ w))]; STATS["replaced"] += 1
        return outs
    mod._mm_batch = mm_batch
elif cls != "none":
    raise SystemExit("unknown class " + cls)
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
est = mod.Estimator(); res = []
for rank in ranks:
    W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
    truth = np.load(P2 / "public_g22_operator_features_v1/truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float64)
    t0 = time.time(); STATS.update(replaced=0, missing_P=0)
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        pred = np.asarray(est.predict(MLP([W[l] for l in range(16)], f"lre-{rank}"), 2**41)).astype(np.float64)
    rec = dict(cls=cls, r=R, rank=rank, mse=float(((pred[15] - truth[15]) ** 2).mean()), bill=int(ctx.summary_dict()["flops_used"]),
               wall=round(time.time() - t0, 1), **STATS)
    res.append(rec); print(json.dumps(rec), flush=True)
    json.dump(res, open(HERE / f"lr_e62_{tag}.json", "w"))
print("DONE", tag, flush=True)
