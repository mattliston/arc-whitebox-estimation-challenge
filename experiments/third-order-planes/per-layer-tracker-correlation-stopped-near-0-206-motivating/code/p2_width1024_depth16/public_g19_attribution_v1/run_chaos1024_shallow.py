#!/usr/bin/env python3
"""TAG chaos1024-shallow: the truncation ladder on the REAL 1024 net.

Identification-free shallow chaos frame (biasless net, x ~ N(0,I)):
z0 = W0 x exactly order-1; relu(z0) decomposes per unit into 1D
Hermites of s_i = a_i.x/|a_i| with exact GH coefficients. Arms build
z1 from that decomposition truncated at k<=2 / k<=4 / k<=12 (~exact),
then push relu(z1_arm) -> z2 EXACTLY on shared samples. Grades full
C and K21 matrices at pre-L1 and pre-L2 for each arm AND the card's
carried planes (host_cov), all against banked MC truth
(mps_cov2 mc_pre_cov) + in-run sampled truth for K21 (not banked).
This is the 1024 mechanism gate: does chaos-order-4 supply beat the
card's plane recursion on the real net at the first depths where the
card starts approximating? Usage: run_chaos1024_shallow.py [rank]
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RANK = int(sys.argv[1]) if len(sys.argv) > 1 else 35
N, CHUNK, KMAX = 400_000, 50_000, 12
ARMS = [2, 4, KMAX]
SEED = 20260831


def hermite_relu_coefs(kmax):
    """alpha_k for relu(s), s~N(0,1), orthonormal He basis (trapz)."""
    x = np.linspace(-12.0, 12.0, 2_000_001)
    w = np.exp(-x * x / 2) / np.sqrt(2 * np.pi)
    He = [np.ones_like(x), x]
    for k in range(2, kmax + 1):
        He.append(x * He[-1] - (k - 1) * He[-2])
    ph = np.maximum(x, 0.0)
    from math import factorial, sqrt
    return np.array([np.trapz(w * ph * He[k], x) / sqrt(factorial(k))
                     for k in range(kmax + 1)])


def load_weights(rank):
    P2 = HERE.parent
    STORE = (P2 / "learned_residual_observer_design/v4_feature_store_v3/"
             "sealed_target_blind_feature_store_v3")
    man = json.loads((STORE / "manifest.json").read_text())
    entry = man["roles"]["observer_fit"]["row_manifests"][rank]
    return np.load((STORE / entry["path"]).parent / "weights.npy",
                   mmap_mode="r")


class Acc:
    """Streaming raw-moment accumulators for C and K21 of one matrix."""

    def __init__(self, n):
        self.s1 = np.zeros(n)
        self.s2 = np.zeros((n, n))
        self.s3 = np.zeros((n, n))       # sum z_c^2 z_d
        self.sq = np.zeros(n)            # sum z^2 (diag helper)
        self.N = 0

    def add(self, Z):
        Zd = Z.astype(np.float64)
        self.s1 += Zd.sum(0)
        self.s2 += Zd.T @ Zd
        self.s3 += (Zd * Zd).T @ Zd
        self.sq += (Zd * Zd).sum(0)
        self.N += len(Zd)

    def finish(self):
        m = self.s1 / self.N
        E2 = self.s2 / self.N
        C = E2 - np.outer(m, m)
        E21 = self.s3 / self.N            # E[z_c^2 z_d]
        Ezz = E2
        Ez2 = self.sq / self.N
        # central E[(zc-mc)^2 (zd-md)]
        K21 = (E21 - 2 * m[:, None] * Ezz - m[None, :] * Ez2[:, None]
               + 2 * (m ** 2)[:, None] * m[None, :])
        return m, C, K21


def rel(a, b):
    return round(float(np.linalg.norm(a - b)
                       / max(np.linalg.norm(b), 1e-30)), 4)


def main():
    W = load_weights(RANK)
    W0 = np.asarray(W[0], dtype=np.float64)
    W1 = np.asarray(W[1], dtype=np.float32)
    W2 = np.asarray(W[2], dtype=np.float32)
    truth = np.load(HERE / f"mps_cov2_rank{RANK:04d}.npz")
    v0 = truth["mc_pre_v"][0].astype(np.float64)
    v1 = truth["mc_pre_v"][1].astype(np.float64)
    # orientation: z0 = W0 x (rows) or W0^T x (cols)?
    r_row = np.corrcoef(np.sum(W0 ** 2, 1), v0)[0, 1]
    r_col = np.corrcoef(np.sum(W0 ** 2, 0), v0)[0, 1]
    if r_col > r_row:
        W0 = W0.T
    sig = np.linalg.norm(W0, axis=1)
    ahat = (W0 / sig[:, None]).astype(np.float32)
    al = hermite_relu_coefs(KMAX)
    print(f"relu He coefs {np.round(al[:8], 5)}  W0 orient "
          f"row_corr {r_row:.5f} col_corr {r_col:.5f}", flush=True)
    # push orientation probe: exact post-cov diag @ L0 via alpha^2,
    # v1 = diag(W1 K W1^T) (rows) vs diag(W1^T K W1) (cols)
    varp = np.zeros(1024)
    for c in range(1, KMAX + 1):
        varp += al[c] ** 2
    varp = varp * sig ** 2                   # exact per-unit Var(phi0)
    va = (W1 ** 2) @ varp                    # rows-convention approx
    vb = (W1 ** 2).T @ varp                  # cols-convention approx
    ra = np.corrcoef(va, v1)[0, 1]
    rb = np.corrcoef(vb, v1)[0, 1]
    ROWS = ra >= rb
    print(f"push orient: rows_corr {ra:.5f} cols_corr {rb:.5f} "
          f"-> {'z=W@phi' if ROWS else 'z=W.T@phi'}", flush=True)
    if not ROWS:
        W1 = np.ascontiguousarray(W1.T)
        W2 = np.ascontiguousarray(W2.T)
    from math import factorial, sqrt
    rng = np.random.default_rng(SEED)
    accs = {("z1", k): Acc(1024) for k in ARMS}
    accs.update({("z2", k): Acc(1024) for k in ARMS})
    half = {("z1", 0): Acc(1024), ("z1", 1): Acc(1024),
            ("z2", 0): Acc(1024), ("z2", 1): Acc(1024)}
    for s0 in range(0, N, CHUNK):
        x = rng.standard_normal((CHUNK, 1024)).astype(np.float32)
        s = x @ ahat.T                      # unit-variance per unit
        del x
        He_km1 = np.ones_like(s)
        He_k = s.copy()
        # phi0 truncated reconstructions, all arms in one recursion
        rec = {k: np.zeros_like(s) for k in ARMS}
        for k in ARMS:
            rec[k] += al[0] * He_km1
        term = al[1] * He_k                 # k=1 term (He_1/sqrt(1!))
        for k in ARMS:
            rec[k] += term
        for kk in range(2, KMAX + 1):
            He_km1, He_k = He_k, s * He_k - (kk - 1) * He_km1
            term = (al[kk] / sqrt(factorial(kk))) * He_k
            for k in ARMS:
                if kk <= k:
                    rec[k] += term
        del He_km1, He_k, term, s
        for k in ARMS:
            ph0 = sig.astype(np.float32)[None, :] * rec[k]
            z1 = ph0 @ W1.T
            accs[("z1", k)].add(z1)
            z2 = np.maximum(z1, 0.0) @ W2.T
            accs[("z2", k)].add(z2)
            if k == KMAX:
                p = (s0 // CHUNK) % 2
                half[("z1", p)].add(z1)
                half[("z2", p)].add(z2)
            del z1, z2, ph0
        del rec
        print(f"chunk {s0 + CHUNK}/{N}", flush=True)

    truth = np.load(HERE / f"mps_cov2_rank{RANK:04d}.npz")
    host = np.load(HERE / f"host_cov_rank{RANK:04d}_base.npz")
    out = {}
    for li, lname in [(1, "preL1"), (2, "preL2")]:
        tC = truth["mc_pre_cov"][li].astype(np.float64)
        # truth K21 from the k=KMAX arm (exact-to-tail chain)
        _, Cx, Kx = accs[(f"z{li}", KMAX)].finish()
        # align card row by diag correlation
        best, bcorr = -1, -2.0
        for i in range(host["cov"].shape[0]):
            c = np.corrcoef(np.diag(host["cov"][i]), np.diag(tC))[0, 1]
            if c > bcorr:
                best, bcorr = i, c
        hC = host["cov"][best].astype(np.float64)
        hK = host["k21"][best].astype(np.float64)
        _, Ca, Ka = half[(f"z{li}", 0)].finish()
        _, Cb, Kb = half[(f"z{li}", 1)].finish()
        rec = {"card_row": best, "card_row_diagcorr": round(bcorr, 5),
               "card_C_rel": rel(hC, tC), "card_K21_rel_vsK12": rel(hK, Kx),
               "card_K21T_rel_vsK12": rel(hK.T, Kx),
               "K21_noise_floor_halfdiff": rel(Ka, Kb),
               "C_noise_floor_halfdiff": rel(Ca, Cb),
               "k12_C_rel_vs_banked": rel(Cx, tC)}
        for k in ARMS:
            m, C, K = accs[(f"z{li}", k)].finish()
            rec[f"k{k}_C_rel"] = rel(C, tC)
            rec[f"k{k}_K21_rel_vsK12"] = rel(K, Kx)
        np.savez_compressed(HERE / f"chaos1024_planes_L{li}_r{RANK}.npz",
                            C=Cx.astype(np.float32),
                            K21=Kx.astype(np.float32))
        out[lname] = rec
        print(lname, json.dumps(rec), flush=True)
    (HERE / f"CHAOS1024_SHALLOW_rank{RANK:04d}.json").write_text(
        json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
