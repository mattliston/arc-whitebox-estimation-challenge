#!/usr/bin/env python3
"""TAG g3-dv1024: the first 1024 G3-supply value signal.

From fitted cores (a_i, M_i) at layer l (shared r-dim latent,
ncqc_corefit): per-unit RESOLVENT-EXACT post T-emission (certified
machinery), coefficient-space Gram G3hat_post, then the dv-hat
candidate dv_G3 = diag(W_{l+1}^T [G3hat - copula_t3] W_{l+1})
(+ raw un-subtracted variant), graded against the true Delta-v
target (tv - hv at l+1) alongside the chain's dv-hat — route-(iii)
consumption grading (correlation is what the beta-fit consumes).
Usage: run_g3_dv1024.py <rank> <layer> [corefile]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "public_g19_math_acceleration_v1"))
from run_chaos1024_shallow import load_weights
from run_tensor_recursion_v4 import columns

RANK = int(sys.argv[1]) if len(sys.argv) > 1 else 35
L = int(sys.argv[2]) if len(sys.argv) > 2 else 5
CORE = sys.argv[3] if len(sys.argv) > 3 else \
    f"ncqc_corefit_rank{RANK:04d}_L{L}_r24.npz"


def exact_T_r(m, a, M):
    """Resolvent-exact E[grad^3 phi] at latent dim r (dense)."""
    rr = len(a)
    lam, U = np.linalg.eigh(M)
    at = U.T @ a
    ts = np.concatenate([np.linspace(1e-6, 1.0, 1500, endpoint=False),
                         np.linspace(1.0, 240.0, 9000)])
    den = 1.0 - 1j * ts[:, None] * lam[None, :]
    lcf = (1j * ts * m - 1j * ts * lam.sum() / 2
           - 0.5 * ts ** 2 * ((at[None, :] ** 2) / den).sum(1)
           - 0.5 * np.log(den).sum(1))
    cf = np.exp(lcf)
    mt = at[None, :] / den
    Ct = (lam[None, :] ** 2) / den
    w = np.gradient(ts)
    itcfw = 1j * ts * cf * w
    T3 = np.einsum("t,tm,tn,tp->mnp", itcfw, mt, mt, mt)
    S = np.einsum("t,tm,tn->mn", itcfw, mt, Ct)
    T3b = np.zeros((rr, rr, rr), dtype=complex)
    for nu in range(rr):
        T3b[:, nu, nu] += S[:, nu]
        T3b[nu, :, nu] += S[:, nu]
        T3b[nu, nu, :] += S[:, nu]
    Te = (T3 + T3b).real / np.pi
    md = np.einsum("t,tm->m", cf * w, mt).real / np.pi
    Te_g = np.einsum("am,bn,cp,mnp->abc", U, U, U, Te, optimize=True)
    d1 = U @ md
    T2 = (np.einsum("i,jk->ijk", d1, M)
          + np.einsum("j,ik->ijk", d1, M)
          + np.einsum("k,ij->ijk", d1, M))
    return Te_g + T2


def sym_coef_vec(T):
    """Dense sym3 tensor -> orthonormal-He coefficient vector."""
    rr = T.shape[0]
    out = []
    for i in range(rr):
        for j in range(i, rr):
            for k in range(j, rr):
                cnt = np.array([i, j, k])
                mult = math.factorial(3)
                fac = 1.0
                for v in set(cnt.tolist()):
                    c = int((cnt == v).sum())
                    mult //= math.factorial(c)
                    fac *= math.factorial(c)
                out.append(T[i, j, k] * mult * math.sqrt(fac))
    return np.array(out)


def main():
    W = load_weights(RANK)
    st = np.load(HERE / f"states_rank{RANK:04d}.npz")
    tr = np.load(HERE / f"mps_cov2_rank{RANK:04d}.npz")
    hc = np.load(HERE / f"host_cov_rank{RANK:04d}_base.npz")
    core = np.load(HERE / CORE)
    a_all = core["a"].astype(np.float64)
    M_all = core["M"].astype(np.float64)
    sel = core["sel"]
    nfit = len(sel)
    m_ = st["host_m"][L - 1].astype(np.float64)
    coefs = []
    for i in range(nfit):
        T = exact_T_r(float(m_[sel[i]]), a_all[i], M_all[i]) / 6.0
        coefs.append(sym_coef_vec(T))
        if (i + 1) % 256 == 0:
            print(f"emission {i + 1}/{nfit}", flush=True)
    X = np.stack(coefs)
    G3 = X @ X.T
    # copula t3 model on the same sub-block (card state)
    v_ = st["host_v"][L - 1].astype(np.float64)
    C = hc["cov"][L - 1].astype(np.float64)
    c1, c2, sd, pdf, cdf = columns(m_, v_)
    rho = np.clip(C / np.outer(sd, sd), -0.9999, 0.9999)
    T3meh = (rho ** 3 / 6.0) * np.outer(c1[3], c1[3])
    T3s = T3meh[np.ix_(sel, sel)]
    # push: dv on the fitted sub-block only (weights sub-columns)
    Wp = np.asarray(W[L + 1], dtype=np.float64)[sel, :]
    tv = tr["mc_pre_v"][L + 1].astype(np.float64)
    hv = st["host_v"][L].astype(np.float64)
    dv_true = tv - hv
    out = {}
    for tag, Gx in (("raw", G3), ("swap", G3 - T3s)):
        dv = np.einsum("ac,ab,bc->c", Wp, Gx, Wp, optimize=True)
        cc = float(np.corrcoef(dv, dv_true)[0, 1])
        bb = float(np.dot(dv, dv_true) / max(np.dot(dv, dv), 1e-30))
        out[tag] = {"corr": round(cc, 4), "beta": round(bb, 4)}
    # chain reference at the same layer
    try:
        dvh = np.load(HERE / f"dvhat_rank{RANK:04d}.npz")[f"L{L + 1}"]
        cc = float(np.corrcoef(dvh, dv_true)[0, 1])
        out["chain_ref"] = {"corr": round(cc, 4)}
    except Exception:
        pass
    print(RANK, f"L{L}", json.dumps(out), flush=True)
    (HERE / f"G3_DV1024_rank{RANK:04d}_L{L}.json").write_text(
        json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
