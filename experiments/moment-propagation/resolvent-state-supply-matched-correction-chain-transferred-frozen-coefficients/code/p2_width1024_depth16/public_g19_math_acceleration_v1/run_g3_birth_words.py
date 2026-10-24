#!/usr/bin/env python3
"""TAG g3-birth-words: analytic H3 birth emission vs exact (tiny world).

The supply the residual field R needs is the G3-sector Gram (three
independent confirmations). Cell G3-1: the BIRTH T-emission model —
third Stein ladder of phi(z) for z = m + a.g + Q_M(g):
  T = E[phi'''] a3 + 3E[phi''''] sym(a,a,Ma) + 3E[phi''] sym(M,a)
      + 3E[phi'''] sym(M,Ma) + 3E[phi'''] sym(M2,a)
with CF-exact scalars E[phi^(k+2)] = (-1)^k p^(k)(0). Verified per
unit against the EXACT order-3 coefficients banked by the ceiling
harness (ceil_coef_seed10_L1.npy = pure birth at the exact L1 state).
Gate: R^2 med >= 0.95 at L1; report the per-word ablation.
Usage: run_g3_birth_words.py [seed] [L]
"""
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_word_closure import (MI, IDX3, tens_to_coef, sym3, r, n)

_args = [a for a in sys.argv[1:] if a.lstrip("-").isdigit()
         and not a.startswith("-")]
SEED = int(_args[0]) if _args else 10
L = int(_args[1]) if len(_args) > 1 else 1
DEPTH = 8


def cf_scalars(m, lam, at):
    """p(0), p'(0), p''(0), P(z>0) via CF-only integrals (certified)."""
    ts = np.concatenate([np.linspace(1e-6, 1.0, 4000, endpoint=False),
                         np.linspace(1.0, 300.0, 24000)])
    den = 1.0 - 1j * ts[:, None] * lam[None, :]
    lcf = (1j * ts * m - 1j * ts * lam.sum() / 2
           - 0.5 * ts ** 2 * ((at[None, :] ** 2) / den).sum(1)
           - 0.5 * np.log(den).sum(1))
    cf = np.exp(lcf)
    pz = 0.5 + (1 / np.pi) * np.trapz(cf.imag / ts, ts)
    p0 = (1 / np.pi) * np.trapz(cf.real, ts)
    p1 = (1 / np.pi) * np.trapz(ts * cf.imag, ts)
    p2 = -(1 / np.pi) * np.trapz(ts * ts * cf.real, ts)
    return pz, p0, p1, p2


def birth_words(a, M):
    Ma = M @ a
    M2 = M @ M
    M2a = M2 @ a
    words = {
        "aaa": sym3(np.einsum("i,j,k->ijk", a, a, a)),
        "aaMa": 3 * sym3(np.einsum("i,j,k->ijk", a, a, Ma)),
        "Ma_sym": 3 * sym3(np.einsum("ij,k->ijk", M, a)),
        "MMa": 3 * sym3(np.einsum("ij,k->ijk", M, Ma)),
        "M2a": 3 * sym3(np.einsum("ij,k->ijk", M2, a)),
        "aaM2a": 3 * sym3(np.einsum("i,j,k->ijk", a, a, M2a)),
    }
    return words


def main():
    import os
    rng = np.random.default_rng(SEED)
    A0 = rng.standard_normal((n, r)) * float(os.environ.get("SA", "0.3"))
    M0 = rng.standard_normal((n, r, r)) * float(os.environ.get("SM", "0.25"))
    M0 = 0.5 * (M0 + np.transpose(M0, (0, 2, 1)))
    m0 = rng.standard_normal(n) * float(os.environ.get("SMU", "0.1"))
    Ws = [rng.standard_normal((n, n)) * np.sqrt(2.0 / n)
          for _ in range(DEPTH)]
    cf = np.load(HERE / f"ceil_coef_seed{SEED}_L{L}.npy").astype(
        np.float64)
    if L == 1:
        mv, A, Mt = m0, A0, M0
    else:
        pre = np.load(HERE / f"ceil_coef_seed{SEED}_L{L - 1}.npy"
                      ).astype(np.float64) @ Ws[L - 2]
        mv = pre[0, :]
        A = pre[1:1 + r, :].T
        Mt = np.zeros((n, r, r))
        pos = 1 + r
        for f in range(1 + r, 1 + r + r * (r + 1) // 2):
            ids = np.where(MI[f] > 0)[0]
            row = pre[f, :]
            if len(ids) == 1:
                Mt[:, ids[0], ids[0]] = row * math.sqrt(2)
            else:
                Mt[:, ids[0], ids[1]] = row
                Mt[:, ids[1], ids[0]] = row
            pos += 1
    y_exact = cf[IDX3, :]                     # (120, n) exact T coefs
    r2s, betas = [], []
    ab_r2 = {k: [] for k in
             ["aaa", "aaMa", "Ma_sym", "MMa", "M2a", "aaM2a"]}
    for c in range(n):
        a = A[c]
        M = Mt[c]
        lam, U = np.linalg.eigh(M)
        at = U.T @ a
        pz, p0, p1, p2 = cf_scalars(float(mv[c]), lam, at)
        E3, E4, E2 = -p1, p2, p0
        words = birth_words(a, M)
        Tm = (E3 * words["aaa"] + E4 * words["aaMa"]
              + E2 * words["Ma_sym"] + E3 * words["MMa"]
              + E3 * words["M2a"] + E4 * 0.0 * words["aaM2a"])
        ym = tens_to_coef(Tm / 6.0, IDX3)   # E[d^3 phi] = 3! T_dense
        y = y_exact[:, c]
        nY = float((y ** 2).sum())
        if nY < 1e-18:
            continue
        res = y - ym
        r2s.append(1 - float((res ** 2).sum()) / nY)
        betas.append(float(np.dot(ym, y) / max(np.dot(ym, ym), 1e-30)))
        # per-word solo fits (diagnostic)
        for k, w in words.items():
            yw = tens_to_coef(w, IDX3)
            bw = float(np.dot(yw, y) / max(np.dot(yw, yw), 1e-30))
            ab_r2[k].append(1 - float(((y - bw * yw) ** 2).sum()) / nY)
    out = {
        "L": L,
        "R2_med": round(float(np.median(r2s)), 4),
        "R2_p10": round(float(np.percentile(r2s, 10)), 4),
        "beta_med": round(float(np.median(betas)), 4),
        "solo_R2_med": {k: round(float(np.median(v)), 3)
                        for k, v in ab_r2.items()},
    }
    print(json.dumps(out, indent=1), flush=True)
    (HERE / f"G3_BIRTH_seed{SEED}_L{L}.json").write_text(
        json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
