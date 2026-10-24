#!/usr/bin/env python3
"""TAG word-closure: is the CARRIED H3/H4 state word-expressible?

Reads the k=4 arm's saved post-phi projection coefficients
(ceil_coef_seed{S}_L{l}.npy from run_chaos_ceiling.py --save),
builds each layer's PRE-activation state coefficients (coef @ W),
extracts per-unit (a, M, T, Q4) from the normalized-He rows, and
fits T and Q4 per unit on word dictionaries built from that unit's
own (a, M). Birth word-sparsity was R^2 ~0.979 (cell 3a); the
question here is whether MIXING preserves it (the carried state is
word-closed) -> 1024 state compresses to ~20 scalars/unit; or not ->
dense-T carry + H4 readout route. Usage: run_word_closure.py [seed]
"""
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 10
n, r, DEPTH = 32, 8, 8


def multi_indices(r, kmax):
    out = []
    for k in range(kmax + 1):
        for c in itertools.combinations_with_replacement(range(r), k):
            a = np.zeros(r, dtype=np.int64)
            for d in c:
                a[d] += 1
            out.append(a)
    return np.array(out)


MI = multi_indices(r, 4)
ORD = MI.sum(1)
FACT = np.array([math.sqrt(np.prod([math.factorial(int(o))
                                    for o in a])) for a in MI])
IDX3 = np.where(ORD == 3)[0]
IDX4 = np.where(ORD == 4)[0]
# multiplicity of a sym tensor entry class = #distinct permutations
MULT = np.array([math.factorial(int(k)) / np.prod(
    [math.factorial(int(o)) for o in a])
    for a, k in zip(MI, ORD)])


def tens_to_coef(T, idx):
    """Dense symmetric tensor -> normalized-He coefficient rows.

    For z_part = sum_{indices} T[indices] prod g: the pure-chaos
    coefficient on normalized feature He_alpha/sqrt(alpha!) is
    MULT(alpha) * FACT(alpha) * T[alpha]. Derivation checked on
    classes (1,1,1): 6T; (2,1): 3*sqrt(2)*T; (3): sqrt(6)*T; and
    consistent with the k=2 M-convention (coef_ab = M_ab,
    coef_aa = M_aa/sqrt(2) for dense T2 = M/2)."""
    out = np.empty(len(idx))
    for j, f in enumerate(idx):
        a = MI[f]
        ids = []
        for d in range(r):
            ids += [d] * int(a[d])
        out[j] = T[tuple(ids)] * MULT[f] * FACT[f]
    return out


def sym3(*Ts):
    S = np.zeros((r, r, r))
    for T in Ts:
        for p in itertools.permutations((0, 1, 2)):
            S += np.transpose(T, p)
    return S / (6 * len(Ts))


def sym4(T):
    S = np.zeros((r, r, r, r))
    for p in itertools.permutations((0, 1, 2, 3)):
        S += np.transpose(T, p)
    return S / 24


def words3(a, M):
    Ma = M @ a
    M2 = M @ M
    M2a = M2 @ a
    ws = []
    for u, v, w in [(a, a, a), (a, a, Ma), (a, Ma, Ma), (Ma, Ma, Ma),
                    (a, a, M2a), (a, Ma, M2a)]:
        ws.append(sym3(np.einsum("a,b,c->abc", u, v, w)))
    for u, S in [(a, M), (Ma, M), (M2a, M), (a, M2), (Ma, M2)]:
        ws.append(sym3(np.einsum("a,bc->abc", u, S)))
    ws.append(sym3(np.einsum("ad,bd,cd->abc", M, M, M)))
    return ws


def words4(a, M):
    Ma = M @ a
    M2 = M @ M
    ws = []
    for S1, S2 in [(M, M), (M, M2), (M2, M2)]:
        ws.append(sym4(np.einsum("ab,cd->abcd", S1, S2)))
    for u, v, S in [(a, a, M), (a, Ma, M), (Ma, Ma, M), (a, a, M2)]:
        ws.append(sym4(np.einsum("a,b,cd->abcd", u, v, S)))
    for u, v, w, x in [(a, a, a, a), (a, a, a, Ma), (a, a, Ma, Ma)]:
        ws.append(sym4(np.einsum("a,b,c,d->abcd", u, v, w, x)))
    ws.append(sym4(np.einsum("ae,be,ce,de->abcd", M, M, M, M)))
    ws.append(sym4(np.einsum("ab,ce,de->abcd", M, M, M)))  # M x MM^T
    return ws


def fit_r2(y, X):
    if np.linalg.norm(y) < 1e-14:
        return 1.0
    beta, *_ = np.linalg.lstsq(X, y, rcond=1e-10)
    res = y - X @ beta
    return 1.0 - float((res ** 2).sum() / (y ** 2).sum())


def main():
    rng = np.random.default_rng(SEED)
    rng.standard_normal((n, r))
    rng.standard_normal((n, r, r))
    rng.standard_normal(n)
    Ws = [rng.standard_normal((n, n)) * np.sqrt(2.0 / n)
          for _ in range(DEPTH)]
    out = {}
    for L in range(1, DEPTH):
        cf = np.load(HERE / f"ceil_coef_seed{SEED}_L{L}.npy")  # F x n
        pre = cf.astype(np.float64) @ Ws[L - 1]                # F x n
        r2_3, r2_4, sh3, sh4 = [], [], [], []
        for i in range(n):
            a = pre[1:1 + r, i]
            Mv = pre[1 + r:1 + r + r * (r + 1) // 2, i]
            M = np.zeros((r, r))
            pos = 0
            for f in range(1 + r, 1 + r + r * (r + 1) // 2):
                ids = np.where(MI[f] > 0)[0]
                if len(ids) == 1:
                    M[ids[0], ids[0]] = Mv[pos] * math.sqrt(2)
                else:
                    M[ids[0], ids[1]] = M[ids[1], ids[0]] = Mv[pos]
                pos += 1
            y3 = pre[IDX3, i]
            y4 = pre[IDX4, i]
            X3 = np.stack([tens_to_coef(w, IDX3)
                           for w in words3(a, M)], axis=1)
            X4 = np.stack([tens_to_coef(w, IDX4)
                           for w in words4(a, M)], axis=1)
            r2_3.append(fit_r2(y3, X3))
            r2_4.append(fit_r2(y4, X4))
            tot = (pre[1:, i] ** 2).sum()
            sh3.append(float((y3 ** 2).sum() / max(tot, 1e-30)))
            sh4.append(float((y4 ** 2).sum() / max(tot, 1e-30)))
        rec = {"r2_T_med": round(float(np.median(r2_3)), 4),
               "r2_T_p10": round(float(np.percentile(r2_3, 10)), 4),
               "r2_Q4_med": round(float(np.median(r2_4)), 4),
               "r2_Q4_p10": round(float(np.percentile(r2_4, 10)), 4),
               "H3_energy_share": round(float(np.mean(sh3)), 4),
               "H4_energy_share": round(float(np.mean(sh4)), 4)}
        out[L + 1] = rec
        print(f"preL{L + 1}", json.dumps(rec), flush=True)
    (HERE / f"WORD_CLOSURE_seed{SEED}.json").write_text(
        json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
