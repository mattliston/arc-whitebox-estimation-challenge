#!/usr/bin/env python3
"""TAG gram-verdict: does exact H3+H4 supply close the v-residual?

Joins the sampled chaos-order Grams (run_gram_split) with the
closure model's own order-3/4 content (gated engine_post_cov evals:
M3 = M(full) - M(k3=K21=0); M4 = M(k3=K21=0) - M(all kappa inputs
zero)) and the banked residual field R_l. Verdict per layer:
  D = (G3 - M3) + (G4 - M4)   [the tracker's exact would-be supply]
  frob_closure = 1 - |R - D|/|R|
  corr(R, D); pushed-v closure = 1 - rms(push(R-D))/rms(push(R))
Plus the tail check G5+G6 (is higher-order material?).
Usage: run_gram_verdict.py <rank>
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_vresid_field import engine_post_cov
from run_chaos1024_shallow import load_weights

RANK = int(sys.argv[1]) if len(sys.argv) > 1 else 35
LAYERS = [5, 8, 11, 13]


def main():
    W = load_weights(RANK)
    st = np.load(HERE / f"states_rank{RANK:04d}.npz")
    hc = np.load(HERE / f"host_cov_rank{RANK:04d}_base.npz")
    RF = np.load(HERE / f"vresid_field_rank{RANK:04d}.npz")
    out = {}
    for l in LAYERS:
        i = l - 1
        m_ = st["host_m"][i].astype(np.float64)
        v_ = st["host_v"][i].astype(np.float64)
        k3 = st["host_k3"][i].astype(np.float64)
        k4 = st["host_k4"][i].astype(np.float64)
        C = hc["cov"][i].astype(np.float64)
        K21 = hc["k21"][i].astype(np.float64)
        K22 = hc["k22"][i].astype(np.float64)
        K31 = hc["k31"][i].astype(np.float64)
        z = np.zeros_like
        Mfull = engine_post_cov(m_, v_, k3, k4, C, K21, K22, K31)
        Mno3 = engine_post_cov(m_, v_, z(k3), k4, C, z(K21), K22, K31)
        Mno34 = engine_post_cov(m_, v_, z(k3), z(k4), C,
                                z(K21), z(K22), z(K31))
        M3 = Mfull - Mno3
        M4 = Mno3 - Mno34
        G = np.load(HERE / f"gram_split_rank{RANK:04d}_L{l}.npz")
        G3 = G["G3"].astype(np.float64)
        G4 = G["G4"].astype(np.float64)
        G56 = G["G5"].astype(np.float64) + G["G6"].astype(np.float64)
        R = RF[f"R_{l}"].astype(np.float64)
        D = (G3 - M3) + (G4 - M4)
        nR = np.linalg.norm(R)
        Wp = np.asarray(W[l + 1], dtype=np.float64)

        def push(M):
            return np.einsum("ac,ab,bc->c", Wp, M, Wp, optimize=True)
        pR = push(R)
        rec = {
            "R_rel_norm": round(float(nR), 5),
            "corr_R_D": round(float(
                np.dot(R.ravel(), D.ravel())
                / max(nR * np.linalg.norm(D), 1e-30)), 4),
            "frob_closure": round(1 - float(
                np.linalg.norm(R - D) / nR), 4),
            "pushedv_closure": round(1 - float(
                np.sqrt(np.mean(push(R - D) ** 2))
                / np.sqrt(np.mean(pR ** 2))), 4),
            "G3_vs_M3_ratio": round(float(
                np.linalg.norm(G3) / max(np.linalg.norm(M3), 1e-30)), 3),
            "G4_vs_M4_ratio": round(float(
                np.linalg.norm(G4) / max(np.linalg.norm(M4), 1e-30)), 3),
            "tail_G56_vs_R": round(float(np.linalg.norm(G56) / nR), 3),
            "pushedv_closure_G3only": round(1 - float(
                np.sqrt(np.mean(push(R - (G3 - M3)) ** 2))
                / np.sqrt(np.mean(pR ** 2))), 4),
        }
        out[l] = rec
        print(RANK, f"L{l}", json.dumps(rec), flush=True)
    (HERE / f"GRAM_VERDICT_rank{RANK:04d}.json").write_text(
        json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
