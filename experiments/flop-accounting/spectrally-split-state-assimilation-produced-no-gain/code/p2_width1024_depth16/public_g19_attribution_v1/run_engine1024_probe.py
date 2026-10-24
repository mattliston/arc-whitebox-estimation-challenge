#!/usr/bin/env python3
"""TAG engine1024-probe: corrected CF engines at REAL 1024 spectra.

Probe units of z1 (H<=2 state, exact from the identification-free
shallow frame): per unit build (m, a, M) in x-chaos, eigh(M), then:
  - trace-formula marginals v, k3, k4
  - direct-CF Emean (corrected CF, log-space products, tail-corrected)
  - leave-one-mode F_rest engine couplings for the top-R modes
    (linear E[h phi] and quad-diag E[(h^2-1) phi]); rest-CF exact.
Graded against shared-draw sampled truth of the SAME H<=2 state
(same rng stream as run_chaos1024_shallow). Gates: Emean ratio
1 +/- 0.5%; coupling corr >= 0.995, beta 1 +/- 3%; marginals within
MC noise. This certifies the engine layer at real-net spectra — the
last verify-first gate before the streaming-tracker build.
Usage: run_engine1024_probe.py [rank]
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_chaos1024_shallow import hermite_relu_coefs, load_weights

RANK = int(sys.argv[1]) if len(sys.argv) > 1 else 35
N, CHUNK = 400_000, 50_000
SEED = 20260831
NPROBE, RTOP = 64, 8
NT, TMAX = 4096, 80.0
NX, XW = 2048, 12.0


def unit_state(w, sig, ahat, al):
    m = al[0] * float(np.dot(w, sig))
    a = ahat.T @ (w * sig) * al[1]
    u = w * sig * np.sqrt(2.0) * al[2]
    M = (ahat.T * u[None, :]) @ ahat
    return m, a, M


def log_cf(ts, m, lam, at):
    den = 1.0 - 1j * ts[:, None] * lam[None, :]
    return (1j * ts * m - 1j * ts * lam.sum() / 2
            - 0.5 * ts ** 2 * ((at[None, :] ** 2) / den).sum(1)
            - 0.5 * np.log(den).sum(1))


def emean_direct(m, lam, at):
    ts = np.concatenate([np.linspace(1e-6, 1.0, 4000, endpoint=False),
                         np.linspace(1.0, 300.0, 24000)])
    cf = np.exp(log_cf(ts, m, lam, at))
    eabs = (2 / np.pi) * (np.trapz((1 - cf.real) / ts ** 2, ts)
                          + 1.0 / 300.0)
    return 0.5 * (m + eabs)


def couplings(m, lam, at, top):
    """Linear and quad-diag couplings for modes in `top` via F_rest."""
    ts = np.linspace(1e-6, TMAX, NT)
    lcf = log_cf(ts, m, lam, at)
    den = 1.0 - 1j * ts[:, None] * lam[top][None, :]
    lfac = (-1j * ts[:, None] * lam[top][None, :] / 2
            - 0.5 * ts[:, None] ** 2 * (at[top][None, :] ** 2) / den
            - 0.5 * np.log(den))
    lrest = lcf[:, None] - lfac                  # nt x R
    sd = np.sqrt((at ** 2).sum() + (lam ** 2).sum() / 2)
    xs = np.linspace(m - XW * sd, m + XW * sd, NX)
    dt = ts[1] - ts[0]
    ker = np.exp(-1j * np.outer(ts, xs))
    er = np.exp(lrest).T
    raw = er @ ker - 0.5 * (np.outer(er[:, 0], ker[0])
                            + np.outer(er[:, -1], ker[-1]))
    dens = np.maximum(raw.real * dt / np.pi, 0.0)
    dens /= np.maximum(np.trapz(dens, xs, axis=1), 1e-300)[:, None]
    dx = xs[1] - xs[0]
    P_up = np.cumsum(dens[:, ::-1], axis=1)[:, ::-1] * dx
    XP_up = np.cumsum((dens * xs[None, :])[:, ::-1],
                      axis=1)[:, ::-1] * dx
    gx, gw = np.polynomial.hermite_e.hermegauss(64)
    gw = gw / gw.sum()
    lin = np.zeros(len(top))
    quad = np.zeros(len(top))
    for j, mu in enumerate(top):
        c = at[mu] * gx + lam[mu] * (gx ** 2 - 1) / 2
        idx = np.clip(np.searchsorted(xs, -c), 0, NX - 1)
        F = c * P_up[j, idx] + XP_up[j, idx]
        lin[j] = np.sum(gw * gx * F)
        quad[j] = np.sum(gw * (gx ** 2 - 1) * F)
    return lin, quad


def main():
    W = load_weights(RANK)
    W0 = np.asarray(W[0], dtype=np.float64)
    truth = np.load(HERE / f"mps_cov2_rank{RANK:04d}.npz")
    v0 = truth["mc_pre_v"][0].astype(np.float64)
    if np.corrcoef(np.sum(W0 ** 2, 0), v0)[0, 1] > \
            np.corrcoef(np.sum(W0 ** 2, 1), v0)[0, 1]:
        W0 = W0.T
    W1 = np.asarray(W[1], dtype=np.float32).T   # cols convention
    sig = np.linalg.norm(W0, axis=1)
    ahat = W0 / sig[:, None]
    al = hermite_relu_coefs(4)
    probes = np.linspace(0, 1023, NPROBE).astype(int)
    states, engines = [], []
    for c in probes:
        m, a, M = unit_state(W1[c].astype(np.float64), sig, ahat, al)
        lam, U = np.linalg.eigh(M)
        at = U.T @ a
        imp = at ** 2 + lam ** 2 / 2
        top = np.argsort(imp)[::-1][:RTOP]
        v = (at ** 2).sum() + (lam ** 2).sum() / 2
        k3 = (lam ** 3).sum() + 3 * float(at @ (lam * at))
        k4 = 3 * (lam ** 4).sum() + 12 * float((at * lam) @ (lam * at))
        em = emean_direct(m, lam, at)
        lin, quad = couplings(m, lam, at, top)
        states.append((m, U[:, top], top))
        engines.append({"m": m, "v": v, "k3": k3, "k4": k4,
                        "emean": em, "lin": lin, "quad": quad})
        if len(engines) % 16 == 0:
            print(f"engine {len(engines)}/{NPROBE}", flush=True)
    # ---- sampled truth of the SAME H<=2 state, shared rng stream
    rng = np.random.default_rng(SEED)
    ahat32 = ahat.astype(np.float32)
    sig32 = sig.astype(np.float32)
    Uall = np.stack([np.asarray(s[1], dtype=np.float32)
                     for s in states])          # P x 1024 x R
    Umat = Uall.transpose(0, 2, 1).reshape(NPROBE * RTOP, 1024).T
    acc = {"n": 0, "s1": 0.0, "s2": 0.0, "s3": 0.0, "s4": 0.0,
           "ph": 0.0, "hph": 0.0, "h2ph": 0.0}
    for s0 in range(0, N, CHUNK):
        x = rng.standard_normal((CHUNK, 1024)).astype(np.float32)
        s = x @ ahat32.T
        rec = (al[0] + al[1] * s
               + (al[2] / np.sqrt(2.0)) * (s * s - 1.0))
        z1 = (sig32[None, :] * rec) @ W1.T.astype(np.float32)
        zp = z1[:, probes].astype(np.float64)
        h = (x @ Umat).reshape(CHUNK, NPROBE, RTOP).astype(np.float64)
        ph = np.maximum(zp, 0.0)
        acc["n"] += CHUNK
        acc["s1"] += zp.sum(0)
        acc["s2"] += (zp ** 2).sum(0)
        acc["s3"] += (zp ** 3).sum(0)
        acc["s4"] += (zp ** 4).sum(0)
        acc["ph"] += ph.sum(0)
        acc["hph"] += np.einsum("npr,np->pr", h, ph)
        acc["h2ph"] += np.einsum("npr,np->pr", h * h - 1.0, ph)
        del x, s, rec, z1, zp, h, ph
        print(f"sample chunk {s0 + CHUNK}/{N}", flush=True)
    n = acc["n"]
    m_s = acc["s1"] / n
    v_s = acc["s2"] / n - m_s ** 2
    m3 = acc["s3"] / n
    m2 = acc["s2"] / n
    k3_s = m3 - 3 * m_s * m2 + 2 * m_s ** 3
    m4 = acc["s4"] / n
    k4_s = (m4 - 4 * m_s * m3 + 6 * m_s ** 2 * m2 - 3 * m_s ** 4
            - 3 * v_s ** 2)
    em_s = acc["ph"] / n
    lin_s = acc["hph"] / n
    quad_s = acc["h2ph"] / n
    em_e = np.array([e["emean"] for e in engines])
    ratios = em_e / em_s
    lin_e = np.stack([e["lin"] for e in engines]).ravel()
    quad_e = np.stack([e["quad"] for e in engines]).ravel()
    lin_sv = lin_s.ravel()
    quad_sv = quad_s.ravel()

    def beta_corr(a, b):
        c = float(np.corrcoef(a, b)[0, 1])
        return c, float(np.dot(a, b) / np.dot(b, b))
    lc, lb = beta_corr(lin_e, lin_sv)
    qc, qb = beta_corr(quad_e, quad_sv)
    out = {
        "emean_ratio_med": round(float(np.median(ratios)), 5),
        "emean_ratio_min": round(float(ratios.min()), 5),
        "emean_ratio_max": round(float(ratios.max()), 5),
        "v_ratio_med": round(float(np.median(
            [e["v"] for e in engines] / v_s)), 5),
        "k3_ratio_med": round(float(np.nanmedian(
            np.array([e["k3"] for e in engines])
            / np.where(np.abs(k3_s) < 1e-12, np.nan, k3_s))), 5),
        "k4_ratio_med": round(float(np.nanmedian(
            np.array([e["k4"] for e in engines])
            / np.where(np.abs(k4_s) < 1e-12, np.nan, k4_s))), 5),
        "lin_corr": round(lc, 6), "lin_beta": round(lb, 5),
        "quad_corr": round(qc, 6), "quad_beta": round(qb, 5),
    }
    print(json.dumps(out, indent=1), flush=True)
    (HERE / f"ENGINE1024_PROBE_rank{RANK:04d}.json").write_text(
        json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
