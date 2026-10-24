#!/usr/bin/env python3
"""TAG g3-engine-delta: the difference-first G3 supply candidate.

dP = [resolvent-exact K21post from identified cores]
   - [engine emission_K21 on core-implied stats]   (matched inputs)
Resolvent post-state, all exact (no Stein ladder):
  alpha = E[g phi] = a/2 + (1/pi) int Im[CF m_t]/t dt
  B = E[(gg^T - I)phi] = (1/pi) Re int CF [m_t m_t^T + C_t] dt + pz M
  T = exact_T (certified)
K21post assembly with the certified Wick coefficients (1,1,1,1/2,1).
Value grade: push-diag(dP) vs the chain's UNSUPPLIED residual
(dv_true - beta*dvhat_chain) -> the stacking signal.
Usage: run_g3_engine_delta.py <rank> <layer> [corefile]
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "public_g19_math_acceleration_v1"))
from run_chaos1024_shallow import load_weights
from run_tensor_recursion_v4 import emission_K21

RANK = int(sys.argv[1]) if len(sys.argv) > 1 else 35
L = int(sys.argv[2]) if len(sys.argv) > 2 else 5
CORE = sys.argv[3] if len(sys.argv) > 3 else \
    f"ncqc_corefit_rank{RANK:04d}_L{L}_r24.npz"


def post_state(m, a, M):
    """Exact resolvent post-state (alpha, B, T, pz) for one unit."""
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
    pz = 0.5 + (1 / np.pi) * float(np.sum(w * cf.imag / ts))
    # alpha (eigenframe): at/2 + (1/pi) int Im[cf mt]/t
    al_e = at / 2 + (1 / np.pi) * np.einsum(
        "t,tm->m", w / ts, (cf[:, None] * mt).imag)
    cfw = cf * w
    B_e = (1 / np.pi) * (np.einsum("t,tm,tn->mn", cfw, mt, mt)
                         + np.diag(np.einsum("t,tm->m", cfw, Ct))).real
    B_e = B_e + pz * np.diag(lam)
    # T eigenframe (resolvent-exact, same integrals)
    itcfw = 1j * ts * cfw
    T3 = np.einsum("t,tm,tn,tp->mnp", itcfw, mt, mt, mt)
    S = np.einsum("t,tm,tn->mn", itcfw, mt, Ct)
    T3b = np.zeros((rr, rr, rr), dtype=complex)
    for nu in range(rr):
        T3b[:, nu, nu] += S[:, nu]
        T3b[nu, :, nu] += S[:, nu]
        T3b[nu, nu, :] += S[:, nu]
    md = np.einsum("t,tm->m", cfw, mt).real / np.pi
    Te = (T3 + T3b).real / np.pi
    # back to shared frame
    al_g = U @ al_e
    B_g = U @ B_e @ U.T
    Te_g = np.einsum("am,bn,cp,mnp->abc", U, U, U, Te, optimize=True)
    d1 = U @ md
    T2 = (np.einsum("i,jk->ijk", d1, M)
          + np.einsum("j,ik->ijk", d1, M)
          + np.einsum("k,ij->ijk", d1, M))
    return al_g, B_g, (Te_g + T2), pz


def main():
    W = load_weights(RANK)
    st = np.load(HERE / f"states_rank{RANK:04d}.npz")
    tr = np.load(HERE / f"mps_cov2_rank{RANK:04d}.npz")
    hc = np.load(HERE / f"host_cov_rank{RANK:04d}_base.npz")
    core = np.load(HERE / CORE)
    a_all = core["a"].astype(np.float64)
    M_all = core["M"].astype(np.float64)
    sel = core["sel"]
    nf = len(sel)
    m_ = st["host_m"][L - 1].astype(np.float64)[sel]
    AL = np.zeros((nf, a_all.shape[1]))
    BM = np.zeros((nf,) + M_all.shape[1:])
    TT = np.zeros((nf,) + (a_all.shape[1],) * 3)
    for i in range(nf):
        al_g, B_g, T_g, pz = post_state(float(m_[i]), a_all[i], M_all[i])
        AL[i] = al_g
        BM[i] = B_g
        TT[i] = T_g
        if (i + 1) % 256 == 0:
            print(f"post-state {i + 1}/{nf}", flush=True)
    # K21post assembly (certified coefficients)
    B2 = np.einsum("iab,ibc->iac", BM, BM)
    K21r = (np.einsum("iab,jab->ij", B2, BM)
            + np.einsum("ia,jab,ib->ij", AL, BM, AL)
            + 2 * np.einsum("ia,iab,jb->ij", AL, BM, AL))
    K21r += np.einsum("jabc,ia,ibc->ij", TT, AL, BM, optimize=True)
    K21r += np.einsum("iabc,ia,jbc->ij", TT, AL, BM, optimize=True)
    K21r += np.einsum("iabc,iab,jc->ij", TT, BM, AL, optimize=True)
    K21r += 0.5 * np.einsum("iabc,idbc,jad->ij", TT, TT, BM,
                            optimize=True)
    K21r += np.einsum("iabc,jdbc,iad->ij", TT, TT, BM, optimize=True)
    # engine side on core-implied stats (matched)
    C_impl = (a_all @ a_all.T
              + 0.5 * np.einsum("iab,jab->ij", M_all, M_all))
    M2p = np.einsum("iab,ibc->iac", M_all, M_all)
    K21_impl = (np.einsum("iab,jab->ij", M2p, M_all)
                + np.einsum("ia,jab,ib->ij", a_all, M_all, a_all)
                + 2 * np.einsum("ia,iab,jb->ij", a_all, M_all, a_all))
    v_impl = np.diag(C_impl).copy()
    k3_impl = np.diag(K21_impl).copy()
    lam_all = np.linalg.eigvalsh(M_all)
    k4_impl = 3 * (lam_all ** 4).sum(1) \
        + 12 * np.einsum("ia,iab,ibc,ic->i", a_all, M_all, M_all, a_all)
    K22c = hc["k22"][L - 1].astype(np.float64)[np.ix_(sel, sel)]
    K31c = hc["k31"][L - 1].astype(np.float64)[np.ix_(sel, sel)]
    K21e = emission_K21(m_, v_impl, k3_impl, k4_impl,
                        C_impl, K21_impl, K22c, K31c)
    dP = K21r - K21e
    # value grade
    Wp = np.asarray(W[L + 1], dtype=np.float64)[sel, :]
    tv = tr["mc_pre_v"][L + 1].astype(np.float64)
    hv = st["host_v"][L].astype(np.float64)
    dv_true = tv - hv
    dvh = np.load(HERE / f"dvhat_rank{RANK:04d}.npz")[f"L{L + 1}"]
    beta = float(np.dot(dvh, dv_true) / np.dot(dvh, dvh))
    resid = dv_true - beta * dvh
    # dP is a K21-plane object; its dv-image: the chain consumes dK21
    # via variance-map sensitivities. Leading dv-image of a K21
    # correction: through the k3-diagonal push. Use the pushed diag as
    # the shape candidate (beta-fit absorbs scale):
    dv_c = np.einsum("ac,ab,bc->c", Wp, dP, Wp, optimize=True)
    out = {
        "corr_vs_dvtrue": round(float(np.corrcoef(dv_c, dv_true)[0, 1]), 4),
        "corr_vs_resid": round(float(np.corrcoef(dv_c, resid)[0, 1]), 4),
        "chain_corr": round(float(np.corrcoef(dvh, dv_true)[0, 1]), 4),
        "dP_rel_vs_K21e": round(float(np.linalg.norm(dP)
                                      / np.linalg.norm(K21e)), 4),
    }
    tp = np.load(HERE / f"tensor_planes_rank{RANK:04d}.npz")
    if f"post_{L}" in tp.files:
        refs = tp[f"post_{L}"].astype(np.float64)[np.ix_(sel, sel)]
        out["K21r_vs_ref"] = round(float(
            np.linalg.norm(K21r - refs) / np.linalg.norm(refs)), 4)
        out["K21e_vs_ref"] = round(float(
            np.linalg.norm(K21e - refs) / np.linalg.norm(refs)), 4)
    np.savez_compressed(HERE / f"g3delta_planes_L{L}.npz",
                        K21r=K21r.astype(np.float32),
                        K21e=K21e.astype(np.float32))
    print(RANK, f"L{L}", json.dumps(out), flush=True)
    (HERE / f"G3_ENGINE_DELTA_rank{RANK:04d}_L{L}.json").write_text(
        json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
