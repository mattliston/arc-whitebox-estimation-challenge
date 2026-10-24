#!/usr/bin/env python3
"""TAG g3-exact-emission: resolvent-exact H3 emission (no ladder).

For z = m + a.g + Q_M(g): the tilted measure e^{itz} makes g complex-
Gaussian with mean mu_t = i at t/(1-i lam t), var 1/(1-i lam t) per
eigenmode, so E[d (x)3 e^{itz}] = CF(t)[m_t^3 + 3 sym(m_t (x) C_t)]
with m_t = (I-itM)^-1 a (resolvent), C_t = M^2 (I-itM)^-1. Then
  T_exact = E[phi''' d3] + 3 sym(E[phi'' d] (x) M)
  E[phi''' d3] = (1/2pi) int it CF [m_t^3 + 3sym(m_t C_t)] dt
  E[phi'' d]   = (1/2pi) int CF m_t dt
(real-part symmetrization over t>0). Carried sectors: pz*T
passthrough + exact-alpha' Q4 contraction. Certified against the
tiny-world exact emissions. Usage: SA=.. run_g3_exact_emission.py 21
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_word_closure import MI, IDX3, IDX4, FACT, MULT, tens_to_coef, r, n
from run_g3_transport import state_at, coef_to_dense
from run_g3_birth_words import cf_scalars

_args = [a for a in sys.argv[1:] if a.isdigit()]
SEED = int(_args[0]) if _args else 21
LAYERS = [int(x) for x in _args[1:]] or [1, 2, 4, 6]


def exact_T(m, a, M):
    lam, U = np.linalg.eigh(M)
    at = U.T @ a
    ts = np.concatenate([np.linspace(1e-6, 1.0, 2000, endpoint=False),
                         np.linspace(1.0, 240.0, 12000)])
    den = 1.0 - 1j * ts[:, None] * lam[None, :]          # nt x r
    lcf = (1j * ts * m - 1j * ts * lam.sum() / 2
           - 0.5 * ts ** 2 * ((at[None, :] ** 2) / den).sum(1)
           - 0.5 * np.log(den).sum(1))
    cf = np.exp(lcf)
    mt = at[None, :] / den                                # nt x r
    Ct = (lam[None, :] ** 2) / den                        # diag, nt x r
    w = np.gradient(ts)
    # E[phi''' d3] eigenframe: (1/pi) int Re{ it cf [...] } dt
    itcf = 1j * ts * cf
    T3 = np.einsum("t,tm,tn,tp->mnp", itcf * w, mt, mt, mt)
    # 3 sym(m_t x C_t): C_t diagonal -> term_mnp = m_m C_n delta_np sym
    S = np.einsum("t,tm,tn->mn", itcf * w, mt, Ct)        # m x (diag n)
    T3b = np.zeros((r, r, r), dtype=complex)
    for nu in range(r):
        T3b[:, nu, nu] += S[:, nu]
        T3b[nu, :, nu] += S[:, nu]
        T3b[nu, nu, :] += S[:, nu]
    Te = (T3 + T3b).real / np.pi
    # E[phi'' d] eigenframe
    md = np.einsum("t,tm->m", cf * w, mt).real / np.pi
    # back to g-frame
    Te_g = np.einsum("am,bn,cp,mnp->abc", U, U, U, Te, optimize=True)
    d1 = U @ md
    Mx = M
    T2 = np.zeros((r, r, r))
    for i in range(r):
        T2 += 0.0
    T2 = (np.einsum("i,jk->ijk", d1, Mx)
          + np.einsum("j,ik->ijk", d1, Mx)
          + np.einsum("k,ij->ijk", d1, Mx))
    return Te_g + T2, d1


def q4_dense(coef4):
    import itertools as it
    T = np.zeros((r, r, r, r))
    for j, f in enumerate(IDX4):
        al = MI[f]
        ids = []
        for d in range(r):
            ids += [d] * int(al[d])
        val = coef4[j] / (MULT[f] * FACT[f])
        for pm in set(it.permutations(ids)):
            T[pm] = val
    return T


def main():
    rng = np.random.default_rng(SEED)
    A0 = rng.standard_normal((n, r)) * float(os.environ.get("SA", "0.3"))
    M0 = rng.standard_normal((n, r, r)) * float(os.environ.get("SM", "0.25"))
    M0 = 0.5 * (M0 + np.transpose(M0, (0, 2, 1)))
    m0 = rng.standard_normal(n) * float(os.environ.get("SMU", "0.1"))
    Ws = [rng.standard_normal((n, n)) * np.sqrt(2.0 / n) for _ in range(8)]
    out = {}
    for L in LAYERS:
        if L == 1:
            mv, A, Mt = m0, A0, M0
            Tcar = np.zeros((n, len(IDX3)))
            Q4rows = np.zeros((n, len(IDX4)))
        else:
            pre = np.load(HERE / f"ceil_coef_seed{SEED}_L{L - 1}.npy"
                          ).astype(np.float64) @ Ws[L - 2]
            mv, A, Mt = state_at(pre)
            Tcar = pre[IDX3, :].T
            Q4rows = pre[IDX4, :].T
        ye = np.load(HERE / f"ceil_coef_seed{SEED}_L{L}.npy"
                     ).astype(np.float64)[IDX3, :].T
        r2s, betas = [], []
        for c in range(n):
            Tex, d1 = exact_T(float(mv[c]), A[c], Mt[c])
            lam, U = np.linalg.eigh(Mt[c])
            at = U.T @ A[c]
            pz, p0, p1, p2 = cf_scalars(float(mv[c]), lam, at)
            model = pz * Tcar[c] + tens_to_coef(Tex / 6.0, IDX3)
            if Q4rows[c].any():
                Q4 = q4_dense(Q4rows[c])
                cand = 4.0 * np.einsum("m,mjkl->jkl", d1, Q4)
                model = model + tens_to_coef(cand / 6.0, IDX3)
            y = ye[c]
            nY = float((y ** 2).sum())
            if nY < 1e-18:
                continue
            r2s.append(1 - float(((y - model) ** 2).sum()) / nY)
            betas.append(float(np.dot(model, y)
                               / max(np.dot(model, model), 1e-30)))
        out[L] = {"R2_med": round(float(np.median(r2s)), 4),
                  "R2_p10": round(float(np.percentile(r2s, 10)), 4),
                  "beta_med": round(float(np.median(betas)), 4)}
        print(f"L{L}", json.dumps(out[L]), flush=True)
    (HERE / f"G3_EXACT_seed{SEED}.json").write_text(
        json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
