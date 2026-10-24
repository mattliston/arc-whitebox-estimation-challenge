#!/usr/bin/env python3
"""TAG vresid-field: the supervised residual field for the v-channel.

R_l = mc_post_cov^truth[l] - mu11^engine(card state at l): the exact
pair-level field the dv-hat supply chain must produce (truth is
shared-draw, gradable at 1e-5). mu11 uses the engine's own Mehler
pair machinery (series + dM, SECOND_ORDER off = certified-chain
default) on the card's carried (m, v, k3, k4, C, K21, K22, K31).
Banks R_l per layer (f32 npz) + summary metrics: relative Frobenius,
diag share, and the pushed-diagonal check against the card's known
next-v error. Usage: run_vresid_field.py <rank>
"""
import json
import sys
from math import factorial
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_tensor_recursion_v4 import columns
from run_attribution import edg2_relu_mean
from run_chaos1024_shallow import load_weights

RANK = int(sys.argv[1]) if len(sys.argv) > 1 else 35


def engine_post_cov(m, v, k3, k4, C, K21c, K22c, K31c):
    c1, c2, sd, pdf, cdf = columns(m, v)
    rho = np.clip(C / np.outer(sd, sd), -0.9999, 0.9999)

    def series(colsA, colsB, a, b, tmax=None):
        import os
        tmax = int(os.environ.get("TMAX", "20")) if tmax is None else tmax
        acc = np.zeros_like(rho)
        rp = np.ones_like(rho)
        for t in range(tmax + 1):
            if t > 0:
                rp = rp * rho
            acc += rp / factorial(t) * np.outer(colsA[a + t] / sd ** a,
                                                colsB[b + t] / sd ** b)
        return acc

    def dM(colsA, colsB):
        return ((k3[:, None] / 6.0) * series(colsA, colsB, 3, 0)
                + (k3[None, :] / 6.0) * series(colsA, colsB, 0, 3)
                + (K21c / 2.0) * series(colsA, colsB, 2, 1)
                + (K21c.T / 2.0) * series(colsA, colsB, 1, 2)
                + (k4[:, None] / 24.0) * series(colsA, colsB, 4, 0)
                + (k4[None, :] / 24.0) * series(colsA, colsB, 0, 4)
                + (K22c / 4.0) * series(colsA, colsB, 2, 2)
                + (K31c / 6.0) * series(colsA, colsB, 3, 1)
                + (K31c.T / 6.0) * series(colsA, colsB, 1, 3))

    def dM2(colsA, colsB):
        k3r = k3[:, None]
        k3c = k3[None, :]
        Kt = K21c.T
        return (
            (k3r * k3r / 72.0) * series(colsA, colsB, 6, 0)
            + (k3c * k3c / 72.0) * series(colsA, colsB, 0, 6)
            + (k3r * k3c / 36.0) * series(colsA, colsB, 3, 3)
            + (k3r * K21c / 12.0) * series(colsA, colsB, 5, 1)
            + (k3r * Kt / 12.0) * series(colsA, colsB, 4, 2)
            + (k3c * Kt / 12.0) * series(colsA, colsB, 1, 5)
            + (k3c * K21c / 12.0) * series(colsA, colsB, 2, 4)
            + (K21c * K21c / 8.0) * series(colsA, colsB, 4, 2)
            + (Kt * Kt / 8.0) * series(colsA, colsB, 2, 4)
            + (K21c * Kt / 4.0) * series(colsA, colsB, 3, 3))

    import os
    M11 = series(c1, c1, 0, 0) + dM(c1, c1) \
        + (dM2(c1, c1) if os.environ.get("SECOND_ORDER") == "1" else 0.0)
    r1 = edg2_relu_mean(m, v, k3, k4)
    return M11 - np.outer(r1, r1)


def main():
    W = load_weights(RANK)
    st = np.load(HERE / f"states_rank{RANK:04d}.npz")
    hc = np.load(HERE / f"host_cov_rank{RANK:04d}_base.npz")
    tr = np.load(HERE / f"mps_cov2_rank{RANK:04d}.npz")
    out = {}
    fields = {}
    for l in range(1, 14):
        i = l - 1
        if i >= hc["cov"].shape[0]:
            break
        m_ = st["host_m"][i].astype(np.float64)
        v_ = st["host_v"][i].astype(np.float64)
        k3 = st["host_k3"][i].astype(np.float64)
        k4 = st["host_k4"][i].astype(np.float64)
        C = hc["cov"][i].astype(np.float64)
        K21 = hc["k21"][i].astype(np.float64)
        K22 = hc["k22"][i].astype(np.float64) \
            if i < hc["k22"].shape[0] else np.zeros_like(C)
        K31 = hc["k31"][i].astype(np.float64) \
            if i < hc["k31"].shape[0] else np.zeros_like(C)
        mu11 = engine_post_cov(m_, v_, k3, k4, C, K21, K22, K31)
        Tp = tr["mc_post_cov"][l].astype(np.float64)
        R = Tp - mu11
        nT = np.linalg.norm(Tp)
        rec = {"R_rel": round(float(np.linalg.norm(R) / nT), 6),
               "R_diag_rel": round(float(
                   np.linalg.norm(np.diag(R))
                   / np.linalg.norm(np.diag(Tp))), 6)}
        # pushed diagonal vs known next-v error
        wi = l + 1
        if wi < W.shape[0]:
            Wp = np.asarray(W[wi], dtype=np.float64)
            dv = np.einsum("ac,ab,bc->c", Wp, R, Wp, optimize=True)
            tv = tr["mc_pre_v"][l + 1].astype(np.float64)
            hv = st["host_v"][l].astype(np.float64)
            rec["pushedR_v_rel"] = round(float(
                np.sqrt(np.mean(dv ** 2)) / np.sqrt(np.mean(tv ** 2))), 6)
            rec["card_next_v_rel"] = round(float(np.sqrt(np.mean(
                (hv - tv) ** 2)) / np.sqrt(np.mean(tv ** 2))), 6)
        fields[f"R_{l}"] = R.astype(np.float32)
        out[l] = rec
        print(RANK, f"L{l}", json.dumps(rec), flush=True)
    import os
    sfx = "_so2" if os.environ.get("SECOND_ORDER") == "1" else ""
    np.savez_compressed(HERE / f"vresid_field_rank{RANK:04d}{sfx}.npz",
                        **fields)
    (HERE / f"VRESID_FIELD_rank{RANK:04d}{sfx}.json").write_text(
        json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
