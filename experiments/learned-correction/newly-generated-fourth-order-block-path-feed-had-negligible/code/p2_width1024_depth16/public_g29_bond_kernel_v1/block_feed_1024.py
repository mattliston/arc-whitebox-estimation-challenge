#!/usr/bin/env python3
"""TAG g29-block-feed-1024: the GEMM-class BLOCK-pattern born-word feed of the fourth-order planes at width 1024,
and the premise test for the learned-multiplier route.

Post kappa4 distinct content (T-49 census): 4-path words on a labeled path p1-p2-p3-p4 with bond multiplicities
(a, b, c), word = c1[a][p1] c1[a+b][p2] c1[b+c][p3] c1[c][p4] rho_{p1p2}^a rho_{p2p3}^b rho_{p3p4}^c / (a! b! c!),
each labeled path counted with weight 1/2 (two orderings).  Pushed by W (in, out) into the next pre-planes:
  K22[A,B] gets the pattern where the two a-vertices are the first two path positions and the two b-vertices the
  last two (aabb) and its reverse (bbaa)  -> K22_block = 1/2 (X rho^b Y + (X rho^b Y)^T)
  K31[A,B] gets the patterns with the single b-vertex at a path END (aaab: b = p4; baaa: b = p1)
     -> K31_block = 1/2 ( [(X rho^b) ∘ (c1[b+c] W)^T] (rho^c diag(c1[c]) W) + [(Y^T rho^b) ∘ (c1[a+b] W)^T] (rho^a diag(c1[a]) W) )
  with X = (W^T diag(c1[a]) rho^a) ∘ (W^T diag(c1[a+b]))   (out x in),  Y = (diag(c1[b+c]) W) ∘ (rho^c diag(c1[c]) W)  (in x out),
  rho^a = Hadamard power, c1[t] = Hermite coefficient columns of relu(h), h ~ N(m, v) (k3gen.hermite_cols).
Certificate: dense sum over all distinct quadruples at n = 24 vs the GEMM form (this file, --cert).
Premise test (default): with TRUTH pre m, v, C at layer l, compare the block feed at l+1 with the push residual
(truth pre-plane at l+1 minus the plane-supported push of the truth post planes: push_residual_rank0035_L{l}.npz):
ratio, corr, and the residual after the best single scale.  Usage: block_feed_1024.py [--cert] [NB]"""
import itertools
import json
import sys
from math import factorial
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
G19 = HERE.parent / "public_g19_attribution_v1"
G23 = HERE.parent / "public_g23_k3_supply_v1"
sys.path.insert(0, str(G23)); sys.path.insert(0, str(G19))
from k3gen import hermite_cols

NB = int([a for a in sys.argv[1:] if not a.startswith("--")][0]) if [a for a in sys.argv[1:] if not a.startswith("--")] else 3


def words(nb):
    out = []
    for a in range(1, nb + 1):
        for b in range(1, nb + 2 - a):
            for c in range(1, nb + 3 - a - b):
                if a + b + c <= nb:
                    out.append((a, b, c))
    return out


def block_feed(m, v, C, W, nb=NB):
    """GEMM-class block-pattern feed -> (K22_block, K31_block) at the next layer (out x out)."""
    c1 = hermite_cols(m, v)[0]
    sd = np.sqrt(v); rho = C / np.outer(sd, sd); np.fill_diagonal(rho, 0.0)
    rp = {}
    def rpow(t):
        if t not in rp:
            rp[t] = rho ** t
        return rp[t]
    n_out = W.shape[1]
    K22 = np.zeros((n_out, n_out)); K31 = np.zeros((n_out, n_out))
    for (a, b, c) in words(nb):
        coef = 0.5 / (factorial(a) * factorial(b) * factorial(c))
        X = (W.T * c1[a][None, :]) @ rpow(a)                       # (out, in): sum_{p1} W_{p1 A} c1[a]_{p1} rho^a_{p1 p2}
        X = X * (W.T * c1[a + b][None, :])                          # ∘ c1[a+b]_{p2} W_{p2 A}
        Y = (W * c1[b + c][:, None]) * (rpow(c) @ (W * c1[c][:, None]))   # (in, out): c1[b+c]_{p3} W_{p3 B} sum_{p4} rho^c_{p3 p4} c1[c]_{p4} W_{p4 B}
        M = (X @ rpow(b)) @ Y                                       # aabb
        K22 += coef * (M + M.T)
        Z = (X @ rpow(b)) * (W * c1[b + c][:, None]).T             # (out, in) at p3: a-vertices p1,p2,p3
        K31 += coef * (Z @ (rpow(c) @ (W * c1[c][:, None])))       # b = p4
        Z2 = (Y.T @ rpow(b)) * (W * c1[a + b][:, None]).T          # (out, in) at p2: a-vertices p4,p3,p2 (reversed path)
        K31 += coef * (Z2 @ (rpow(a) @ (W * c1[a][:, None])))      # b = p1
    np.fill_diagonal(K22, 0.0); np.fill_diagonal(K31, 0.0)
    return K22, K31


def brute(m, v, C, W, nb=NB):
    """Dense reference: exact sums over all distinct labeled quadruples (n small), block patterns only."""
    n = len(m); c1 = hermite_cols(m, v)[0]
    sd = np.sqrt(v); rho = C / np.outer(sd, sd); np.fill_diagonal(rho, 0.0)
    K22 = np.zeros((W.shape[1], W.shape[1])); K31 = np.zeros_like(K22)
    # all tuples with adjacent indices distinct (rho's zero diagonal kills p1=p2, p2=p3, p3=p4); non-adjacent
    # coincidences (p1=p3, p2=p4, p1=p4) are KEPT: they are the (2,1,1)-class born entries the plane push lacks
    for quad in itertools.product(range(n), repeat=4):
        p = quad
        if p[0] == p[1] or p[1] == p[2] or p[2] == p[3]:
            continue
        wsum = 0.0
        for (a, b, c) in words(nb):
            wsum += (c1[a][p[0]] * c1[a + b][p[1]] * c1[b + c][p[2]] * c1[c][p[3]]
                     * rho[p[0], p[1]] ** a * rho[p[1], p[2]] ** b * rho[p[2], p[3]] ** c) / (factorial(a) * factorial(b) * factorial(c))
        wsum *= 0.5
        if wsum == 0.0:
            continue
        Wa = W[p[0]] * W[p[1]]; Wb = W[p[2]] * W[p[3]]
        K22 += wsum * np.outer(Wa, Wb); K22 += wsum * np.outer(Wb, Wa)
        K31 += wsum * np.outer(W[p[0]] * W[p[1]] * W[p[2]], W[p[3]])
        K31 += wsum * np.outer(W[p[1]] * W[p[2]] * W[p[3]], W[p[0]])
    np.fill_diagonal(K22, 0.0); np.fill_diagonal(K31, 0.0)
    return K22, K31


def main():
    if "--cert" in sys.argv:
        rng = np.random.default_rng(5); n = 24
        A = rng.normal(size=(n, n)) / np.sqrt(n); A += 0.4 * np.ones((n, n)) / np.sqrt(n); C = A @ A.T
        m = rng.normal(0, 0.3, n); v = np.diag(C).copy(); W = rng.normal(size=(n, n)) / np.sqrt(n)
        g22, g31 = block_feed(m, v, C, W); b22, b31 = brute(m, v, C, W)
        off = ~np.eye(n, dtype=bool)
        for nm, g, b in (("K22", g22, b22), ("K31", g31, b31)):
            print(f"cert {nm}: rel err {np.linalg.norm(g[off] - b[off]) / np.linalg.norm(b[off]):.2e} (|ref| {np.linalg.norm(b[off]):.3e})")
        return
    rank = 35
    T = np.load(G19 / f"mps_truth_rank{rank:04d}_P32M.npz"); C8 = np.load(G19 / f"mps_cov2_rank{rank:04d}.npz")
    STORE = G19.parent / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
    man = json.loads((STORE / "manifest.json").read_text()); entry = man["roles"]["observer_fit"]["row_manifests"][rank]
    weights = np.load((STORE / entry["path"]).parent / "weights.npy", mmap_mode="r")
    off = ~np.eye(1024, dtype=bool)
    out = {}
    for l in (8, 12):
        m = T["mc_pre_m"][l].astype(np.float64); v = T["mc_pre_v"][l].astype(np.float64); C = C8["mc_pre_cov"][l].astype(np.float64)
        W = np.asarray(weights[l + 1], dtype=np.float64)
        K22b, K31b = block_feed(m, v, C, W)
        z = np.load(HERE / f"push_residual_rank{rank:04d}_L{l}.npz")
        rec = {}
        for nm, f in (("K22", K22b), ("K31", K31b)):
            miss = z[f"{nm}_missing"].astype(np.float64); truth_next = miss + z[f"{nm}_pushed"].astype(np.float64)
            e, t = f[off], miss[off]
            beta = float(e @ t / (e @ e)); corr = float(e @ t / np.sqrt((e @ e) * (t @ t)))
            rec[nm] = {"ratio_feed_over_missing": float(np.sqrt(np.mean(e ** 2) / np.mean(t ** 2))), "corr_with_missing": corr,
                       "best_scale": beta, "missing_left_after_scaled_feed": float(np.linalg.norm(t - beta * e) / np.linalg.norm(t)),
                       "missing_over_truth_next": float(np.linalg.norm(t) / np.linalg.norm(truth_next[off]))}
            print(f"L{l}->{l+1} {nm}: feed/missing ratio {rec[nm]['ratio_feed_over_missing']:.3f}, corr {corr:+.3f}, best scale {beta:.2f}, "
                  f"missing left after the scaled feed {rec[nm]['missing_left_after_scaled_feed']:.3f} (missing = {rec[nm]['missing_over_truth_next']:.2f} of the truth plane)", flush=True)
        out[str(l)] = rec
    (HERE / "BLOCK_FEED_PREMISE_rank0035.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
