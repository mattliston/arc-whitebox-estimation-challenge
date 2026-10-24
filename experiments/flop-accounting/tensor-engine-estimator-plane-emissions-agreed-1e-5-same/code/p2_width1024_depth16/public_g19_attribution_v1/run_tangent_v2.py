#!/usr/bin/env python3
"""TAG g19b-tangent-v2: covariance-error chain with ANALYTIC Mehler kernels.
eps_{l+1} = W^T [ D o eps_l + Inj_l ] W
  D_ij = dM11/dcov = (1/(si sj)) sum_{t>=1} t rho^{t-1}/t! e_i[t] e_j[t],
         e_u[t] = c1_u[t]/sd_u^t  (standardized Mehler factors)
  Inj_l = sum_channels G_ab o (supply plane delta):
         k22_add/4 * G(2,2) + [k31_add/6 * G(3,1) + transpose] +
         k4_add: (k4/24)(G(4,0)+G(0,4)) diag-vector form
Deltas from the banked supply formulas applied to the tapped K (K_LOG).
Output: dv_hat per layer; scored vs true dv and at the endpoint via the
universal-beta consumption. Usage: run_tangent_v2.py <rank>"""
import json, os, sys
from math import factorial, erf
from pathlib import Path
import numpy as np
sys.path.insert(0, '.')
from run_attribution import edg2_relu_mean
_erf = np.vectorize(erf)
HERE = Path('.').resolve()

def c1_std_factors(m, v, tmax=12):
    v = np.maximum(v, 1e-30); sd = np.sqrt(v); a = m/sd
    pdf = np.exp(-0.5*a*a)/np.sqrt(2*np.pi)
    cdf = 0.5*(1+_erf(a/np.sqrt(2)))
    ma = -a
    He = [np.ones_like(a), ma]
    for k in range(2, tmax+6):
        He.append(ma*He[-1] - (k-1)*He[-2])
    c1 = [sd*pdf + m*cdf, sd*cdf]
    for q in range(2, tmax+6):
        c1.append(sd*He[q-2]*pdf)
    c1 = np.stack(c1)
    e = np.stack([c1[t]/sd**t for t in range(tmax+1)])   # e[t] standardized
    return c1, e, sd

DVSAVE = {}


def varmap_sens(m, v, k3, k4):
    """dVar_post/dk3 and dVar_post/dk4 of the ReLU GC variance map
    (E[f^2]-E[f]^2), closed forms verified vs numeric quadrature."""
    from run_attribution import edg2_relu_mean
    v = np.maximum(v, 1e-30)
    s = np.sqrt(v)
    al = m / s
    ph = np.exp(-0.5 * al * al) / np.sqrt(2 * np.pi)
    ma = -al
    He = [np.ones_like(al), ma]
    for k in range(2, 6):
        He.append(ma * He[-1] - (k - 1) * He[-2])
    dF2_dg3 = (1 / 6.) * (m * m * He[2] + 2 * m * s * (He[3] + 3 * He[1])
                          + s * s * (He[4] + 7 * He[2] + 6.0)) * ph
    dF2_dg4 = (1 / 24.) * (m * m * He[3] + 2 * m * s * (He[4] + 4 * He[2])
                           + s * s * (He[5] + 9 * He[3] + 12 * He[1])) * ph
    c1_3 = s * He[1] * ph
    c1_4 = s * He[2] * ph
    c1_6 = s * He[4] * ph
    g3 = k3 / s ** 3
    Ef = edg2_relu_mean(m, v, k3, k4)
    dEf_dg3 = c1_3 / 6.0 + (g3 / 36.0) * c1_6
    dEf_dg4 = c1_4 / 24.0
    S3 = (dF2_dg3 - 2.0 * Ef * dEf_dg3) / s ** 3
    S4 = (dF2_dg4 - 2.0 * Ef * dEf_dg4) / s ** 4
    return S3, S4


def main():
    rank = int(sys.argv[1])
    from pathlib import Path as _P
    d = (np.load(f"states_rank{rank:04d}_lite.npz")
         if _P(f"states_rank{rank:04d}_lite.npz").exists()
         and not _P(f"states_rank{rank:04d}.npz").exists()
         else np.load(f"states_rank{rank:04d}.npz"))
    z = np.load(f"host_cov_rank{rank:04d}_base.npz")
    try:
        K = np.load(f"sidecar_K_rank{rank:04d}.npz")["K"].astype(
            np.float64)
    except FileNotFoundError:
        K = np.zeros((0, 1024, 1024))      # only used by supplyK mode
    s = np.load(f"sidecar_states_rank{rank:04d}.npz")
    P2 = HERE.parent
    STORE = P2/"learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
    manifest = json.loads((STORE/"manifest.json").read_text())
    entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
    W = np.load((STORE/entry["path"]).parent/"weights.npy", mmap_mode="r")
    T = np.load(f"mps_cov2_rank{rank:04d}.npz")
    Ct = T["mc_pre_cov"].astype(np.float64)

    eps = np.zeros((1024, 1024))
    out = {"rank": rank, "layers": {}}
    for l in range(1, 15):
        i = l - 1
        m_, v_ = d["host_m"][i], d["host_v"][i]
        hk3, hk4 = d["host_k3"][i], d["host_k4"][i]
        C = z["cov"][i].astype(np.float64)
        c1, e, sd = c1_std_factors(m_, v_)
        rho = np.clip(C/np.outer(sd, sd), -0.9999, 0.9999)
        # transport kernel D = dM11/dcov = (dM11/drho)/(si sj)
        D = np.zeros_like(rho)
        rp = np.ones_like(rho)
        for t in range(1, 9):
            if t > 1: rp = rp * rho
            D += t * rp / factorial(t) * np.outer(c1[t], c1[t])
        D = D / np.outer(sd, sd)
        # supply plane deltas at this reset (from the banked formulas)
        if i < K.shape[0] and i < 14:
            Wl = np.asarray(W[l], dtype=np.float64)  # push weights at this reset? K is defined at reset i, plane deltas act on NEXT assemble
        Kk = K[i] if i < K.shape[0] else np.zeros((1024, 1024))
        # plane-delta injection through the pair-map channel kernels at THIS
        # layer's consumption (K is the k22-residual supplied at this reset;
        # its k22-channel effect on M11):
        def Gker(a, b, tmax=8):
            acc = np.zeros_like(rho); rp2 = np.ones_like(rho)
            for t in range(tmax+1):
                if t > 0: rp2 = rp2 * rho
                acc += rp2/factorial(t) * np.outer(c1[a+t]/sd**a, c1[b+t]/sd**b)
            return acc
        mode = sys.argv[2] if len(sys.argv) > 2 else "supplyK"
        # DK21_FILTER: split every plane delta into its collective
        # rank-1 projection (host-native mean direction; the only
        # frame available at deploy) vs the remainder.
        #   coll  -> keep only proj onto B_ab = outer(w^a, w^b)
        #   resid -> keep only the remainder
        filt = os.environ.get("DK21_FILTER", "")
        wcv = m_ / max(np.linalg.norm(m_), 1e-30)

        rk_env = int(os.environ.get("DK21_RANK", "0"))
        perm_seed = int(os.environ.get("DK21_PERM", "-1"))
        if perm_seed >= 0 and not hasattr(main, "_perm"):
            main._perm = np.random.Generator(
                np.random.PCG64(perm_seed)).permutation(1024)

        blk_env = int(os.environ.get("DK21_BLOCK", "0"))

        def _filter_plane(dK, a, b):
            if perm_seed >= 0:
                p_ = main._perm
                dK = dK[np.ix_(p_, p_)]
            if blk_env > 0:
                nb = 1024 // blk_env
                dKb = dK.reshape(nb, blk_env, nb, blk_env).mean((1, 3))
                dK = np.repeat(np.repeat(dKb, blk_env, 0), blk_env, 1)
            if rk_env > 0:
                U, S, Vt = np.linalg.svd(dK, full_matrices=False)
                dK = (U[:, :rk_env] * S[:rk_env]) @ Vt[:rk_env]
            if not filt:
                return dK
            B = np.outer(wcv ** a, wcv ** b)
            c = float(np.sum(dK * B)) / max(float(np.sum(B * B)), 1e-30)
            return c * B if filt == "coll" else dK - c * B
        if mode in ("tensorK21", "hubK21"):
            src = "tensor_planes" if mode == "tensorK21" else "hub_planes"
            tsx = os.environ.get("TP_SUFFIX", "")
            tp = np.load(HERE / f"{src}_rank{rank:04d}{tsx}.npz")
            if f"pre_{l}" in tp.files:
                dK21 = (tp[f"pre_{l}"].astype(np.float64)
                        - z["k21"][i].astype(np.float64))
                dK21 = _filter_plane(dK21, 2, 1)
                inj = (dK21/2.0)*Gker(2, 1) + (dK21.T/2.0)*Gker(1, 2)
            else:
                inj = np.zeros_like(rho)
        elif mode == "trueK21" or mode.startswith("rk:"):
            Pl = np.load(HERE / f"mps_planes_rank{rank:04d}.npz")
            if l in Pl["layers"]:
                dK21 = (Pl[f"K21_{l}"].astype(np.float64)
                        - z["k21"][i].astype(np.float64))
                if mode.startswith("rk:"):
                    kk = int(mode.split(":")[1])
                    U, S, Vt = np.linalg.svd(dK21)
                    dK21 = (U[:, :kk] * S[:kk]) @ Vt[:kk]
                inj = (dK21/2.0)*Gker(2, 1) + (dK21.T/2.0)*Gker(1, 2)
            else:
                inj = np.zeros_like(rho)
        else:
            inj = (Kk/4.0) * Gker(2, 2)
        if os.environ.get("TANGENT_PAIRS") == "1":
            dK22 = dK31 = None
            if mode == "trueK21":
                if f"K22_{l}" in Pl.files and i < z["k22"].shape[0]:
                    dK22 = (Pl[f"K22_{l}"].astype(np.float64)
                            - z["k22"][i].astype(np.float64))
                    dK31 = (Pl[f"K31_{l}"].astype(np.float64)
                            - z["k31"][i].astype(np.float64))
            elif mode in ("tensorK21", "hubK21"):
                if f"K22_{l}" in tp.files and i < z["k22"].shape[0]:
                    dK22 = (tp[f"K22_{l}"].astype(np.float64)
                            - z["k22"][i].astype(np.float64))
                    dK31 = (tp[f"K31_{l}"].astype(np.float64)
                            - z["k31"][i].astype(np.float64))
            if dK22 is not None:
                np.fill_diagonal(dK22, 0.0)
                np.fill_diagonal(dK31, 0.0)
                dK22 = _filter_plane(dK22, 2, 2)
                dK31 = _filter_plane(dK31, 3, 1)
                inj = inj + ((dK22 / 4.0) * Gker(2, 2)
                             + (dK31 / 6.0) * Gker(3, 1)
                             + (dK31.T / 6.0) * Gker(1, 3))
        if os.environ.get("TANGENT_COLL") == "1" and mode == "tensorK21":
            # collective channel: route the collective part of dK21
            # around the pairwise Gker (which under-converts it) and
            # apply the exact 1D-ridge response instead.
            mw = T["mc_pre_m"][l].astype(np.float64)
            wcol = mw / max(np.linalg.norm(mw), 1e-30)
            w2n = wcol * wcol
            base_img = np.outer(w2n, wcol)
            bnorm2 = float(np.sum(base_img ** 2))
            if "dK21" in dir() and bnorm2 > 0:
                dk3s = float(np.sum(dK21 * base_img)) / bnorm2
                # subtract the collective component from the pair path
                inj = inj - ((dk3s * base_img / 2.0) * Gker(2, 1)
                             + (dk3s * base_img.T / 2.0) * Gker(1, 2))
                # exact 1D response: dVar(s') = (dk3s_scalar/6) *
                #   (<He3 F^2> - 2<F><He3 F>) with F = the ridge
                mn2 = T["mc_pre_m"][l + 1].astype(np.float64)
                wn2 = mn2 / max(np.linalg.norm(mn2), 1e-30)
                Wd_l = np.asarray(W[l + 1], dtype=np.float64)
                uvec = Wd_l @ wn2
                Cl_ = z["cov"][i].astype(np.float64)
                sig2 = max(wcol @ Cl_ @ wcol, 1e-30)
                sigc = np.sqrt(sig2)
                bv = (Cl_ @ wcol) / sigc
                vpp = np.maximum(np.diag(Cl_) - bv * bv, 1e-30)
                sdp2 = np.sqrt(vpp)
                xq, wq = np.polynomial.hermite_e.hermegauss(60)
                wq = wq / np.sqrt(2 * np.pi)
                Fv = np.zeros(len(xq))
                for a2, t2 in enumerate(xq):
                    mu2 = m_ + t2 * bv
                    al2 = mu2 / sdp2
                    pd2 = np.exp(-0.5 * al2 * al2) / np.sqrt(2 * np.pi)
                    cd2 = 0.5 * (1 + _erf(al2 / np.sqrt(2)))
                    Fv[a2] = uvec @ (sdp2 * pd2 + mu2 * cd2)
                He3 = xq ** 3 - 3 * xq
                # plane image of delta*(w x w x w) is delta*base_img and
                # kappa3(s)-change = delta (w unit), so dk3_scalar = dk3s
                g3s = dk3s / sig2 ** 1.5
                dVar = (g3s / 6.0) * (float(wq @ (He3 * Fv ** 2))
                                      - 2 * float(wq @ Fv)
                                      * float(wq @ (He3 * Fv)))
                # delta-C = c u-hat u-hat^T with u^T dC u = dVar
                # => c = dVar/||u||^2
                un = uvec / max(np.linalg.norm(uvec), 1e-30)
                inj = inj + (dVar / max(float(uvec @ uvec), 1e-30)
                             ) * np.outer(un, un)
        # INJ_LAYERS="lo-hi": keep injections only for l in [lo,hi]
        # (transport D always active). Linear in inj => keep-only
        # windows partition dv_hat exactly.
        wnd = os.environ.get("INJ_LAYERS", "")
        if wnd:
            lo_, hi_ = (int(x) for x in wnd.split("-"))
            if not (lo_ <= l <= hi_):
                inj = np.zeros_like(rho)
        uv = os.environ.get("TANGENT_UV", "off")
        if uv != "off":
            hk3n = d["host_k3"][i].astype(np.float64)
            hk4n = d["host_k4"][i].astype(np.float64)
            if uv == "oracle":
                dk3 = T["mc_pre_k3"][l].astype(np.float64) - hk3n
                dk4 = T["mc_pre_k4"][l].astype(np.float64) - hk4n
            else:
                tpu = np.load(HERE / f"tensor_planes_rank{rank:04d}"
                              f"{os.environ.get('TP_SUFFIX', '')}.npz")
                dk3 = (tpu[f"k3u_{l}"].astype(np.float64) - hk3n) \
                    if f"k3u_{l}" in tpu.files else np.zeros_like(hk3n)
                dk4 = (tpu[f"k4u_{l}"].astype(np.float64) - hk4n) \
                    if f"k4u_{l}" in tpu.files else np.zeros_like(hk4n)
            S3v, S4v = varmap_sens(m_, v_, hk3n, hk4n)
            inj = inj + np.diag(S3v * dk3 + S4v * dk4)
        core = D * eps + inj
        Wd = np.asarray(W[l+1], dtype=np.float64)
        eps = Wd.T @ core @ Wd
        dv_hat = np.diag(eps)
        dv_true = T["mc_pre_v"][l+1] - d["host_v"][l]
        cc = float(np.sum(dv_hat*dv_true)
                   / np.sqrt(np.sum(dv_hat**2)*np.sum(dv_true**2) + 1e-300))
        beta = float(np.sum(dv_hat*dv_true)/(np.sum(dv_hat**2)+1e-300))
        eps_true = Ct[l+1] - z["cov"][i+1].astype(np.float64) if i+1 < z["cov"].shape[0] else None
        ec = None
        if eps_true is not None:
            off_h = eps - np.diag(np.diag(eps))
            off_t = eps_true - np.diag(np.diag(eps_true))
            ec = float(np.sum(off_h*off_t)
                       / np.sqrt(np.sum(off_h**2)*np.sum(off_t**2) + 1e-300))
        if mode in ("tensorK21", "hubK21", "trueK21"):
            DVSAVE[l + 1] = dv_hat.copy()
        out["layers"][str(l+1)] = {"eps_hat_off_rms": float(np.sqrt(np.mean((eps-np.diag(np.diag(eps)))**2))),
                                   "inj_off_rms": float(np.sqrt(np.mean((inj-np.diag(np.diag(inj)))**2))),
                                   "eps_true_off_rms": (float(np.sqrt(np.mean((eps_true-np.diag(np.diag(eps_true)))**2))) if eps_true is not None else None),
                                   "dv_corr": round(cc, 3),
                                   "dv_beta": round(beta, 3),
                                   "eps_offdiag_corr": (round(ec, 3) if ec is not None else None)}
    (HERE / f"TANGENT_V2_rank{rank:04d}_{sys.argv[2] if len(sys.argv)>2 else 'supplyK'}.json").write_text(json.dumps(out, indent=1) + "\n")
    if DVSAVE:
        md = sys.argv[2] if len(sys.argv) > 2 else ""
        sfx = ({"hubK21": "_hub", "trueK21": "_true"}.get(md, "")
               + os.environ.get("TP_SUFFIX", ""))
        if os.environ.get("DK21_FILTER"):
            sfx += "_" + os.environ["DK21_FILTER"]
        if os.environ.get("INJ_LAYERS"):
            sfx += "_w" + os.environ["INJ_LAYERS"].replace("-", "")
        if os.environ.get("DK21_RANK"):
            sfx += "_rk" + os.environ["DK21_RANK"]
        if os.environ.get("DK21_BLOCK"):
            sfx += "_blk" + os.environ["DK21_BLOCK"]
        if os.environ.get("DK21_PERM"):
            sfx += "_perm" + os.environ["DK21_PERM"]
        np.savez(HERE / f"dvhat_rank{rank:04d}{sfx}.npz",
                 **{f"L{k}": v for k, v in DVSAVE.items()})
    for l in (5, 9, 12, 14, 15):
        print(rank, "L"+str(l), json.dumps(out["layers"].get(str(l))))

if __name__ == "__main__":
    main()
