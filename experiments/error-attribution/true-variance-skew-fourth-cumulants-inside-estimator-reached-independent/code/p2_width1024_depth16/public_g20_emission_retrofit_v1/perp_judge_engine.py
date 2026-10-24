"""Perpendicular judge (teammate D's REDUNDANCY_DIAGNOSTIC recipe, public_g22 LANE §10) for the engine in-card arms.

Frame: final-layer (layer 16) per-unit vectors, independent-4M truth (mps_truth_rank*_indep4M.npz mc_post[15]).
  host      = original deployed host, F2J off, no tables (retro_out_base_rank*.npy row 0)
  deployed  = deployed full card (deployed_rank*.npy[15])
  c_F2J     = deployed - host                          (the operator's correction)
  two-table host = en_base (panel k4 + k3 scale tables, F2J off)
  for each arm: c_total = arm host - host (tables + injection);  c_own = arm host - two-table host (the injection alone)
  metrics: corr(c, c_F2J), slope of c_F2J on c, fraction of |c|^2 inside c_F2J's span, corr(c, truth-host),
           corr(c_perp, truth-deployed) with c_perp = c - proj_{c_F2J}(c); own-half judged against truth - two-table host.
  F2J-on MSE of each arm from row 1 of the same retro_out file (deployed payload on the corrected card's tape).
Writes PERP_JUDGE_ENGINE.json next to this file and prints a markdown table.
"""
import glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
G19 = os.path.join(os.path.dirname(HERE), "public_g19_attribution_v1")
ARMS = ["en_dv_b1", "en_dv_b04", "en_dk3", "en_both_b04", "en_both_b1"]


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def judge(c, cF, e_ref, e_dep):
    """c: effect vector; cF: operator correction; e_ref: truth - the host c was added to; e_dep: truth - deployed."""
    proj = (c @ cF) / (cF @ cF) * cF
    cperp = c - proj
    return {
        "rms": float(np.sqrt(np.mean(c ** 2))),
        "corr_cF2J": corr(c, cF),
        "slope_cF2J_on_c": float((c @ cF) / (c @ c)),
        "frac_c_inside_cF2J_span": float((proj @ proj) / (c @ c)),
        "corr_c_residual": corr(c, e_ref),
        "corr_cperp_truthminusdeployed": corr(cperp, e_dep),
        # ceiling: the deployed residual energy c_perp could remove at its OPTIMAL scale (= corr^2, centred)
        "perp_best_removal_pct": float(100.0 * corr(cperp, e_dep) ** 2),
        "energy_removed_from_residual_pct": float(100.0 * (1.0 - np.mean((e_ref - c) ** 2) / np.mean(e_ref ** 2))),
    }


out = {}
for R, tag in (("0035", "r35"), ("0100", "r100")):
    _tz = np.load(os.path.join(G19, f"mps_truth_rank{R}_indep4M.npz"))
    truth = _tz["mc_post"][15]
    noise = float(np.mean(_tz["mc_se"][15] ** 2))  # MC noise floor of the 4M truth; MSEs below are noise-corrected (the harness's mse_nc frame)
    dep = np.load(os.path.join(G19, f"deployed_rank{R}.npy"))[15]
    host = np.load(os.path.join(HERE, f"retro_out_base_rank{R}.npy"))[0]
    fb = os.path.join(HERE, f"retro_out_en_base_{tag}_rank{R}.npy")
    if not os.path.exists(fb):
        print(f"[{R}] no en_base yet"); continue
    en_base = np.load(fb)
    tt_host = en_base[0]
    cF = dep - host
    e_host, e_dep, e_tt = truth - host, truth - dep, truth - tt_host
    rec = {
        "noise_floor": noise,
        "host_mse": float(np.mean(e_host ** 2)) - noise, "deployed_mse": float(np.mean(e_dep ** 2)) - noise,
        "two_table_host_mse": float(np.mean(e_tt ** 2)) - noise, "two_table_f2j_on_mse": float(np.mean((truth - en_base[1]) ** 2)) - noise,
        "cF2J_rms": float(np.sqrt(np.mean(cF ** 2))), "corr_cF2J_ehost": corr(cF, e_host),
        "arms": {},
    }
    rec["arms"]["en_base (two tables)"] = {"total": judge(tt_host - host, cF, e_host, e_dep), "f2j_on_mse": rec["two_table_f2j_on_mse"], "host_mse": rec["two_table_host_mse"]}
    for a in ARMS:
        f = os.path.join(HERE, f"retro_out_{a}_{tag}_rank{R}.npy")
        if not os.path.exists(f):
            continue
        o = np.load(f)
        rec["arms"][a] = {
            "host_mse": float(np.mean((truth - o[0]) ** 2)) - noise,
            "f2j_on_mse": float(np.mean((truth - o[1]) ** 2)) - noise,
            "total": judge(o[0] - host, cF, e_host, e_dep),
            "own": judge(o[0] - tt_host, cF, e_tt, e_dep),
        }
    out[R] = rec
    print(f"\n### rank {R}: host {rec['host_mse']:.3e}, deployed {rec['deployed_mse']:.3e}, two-table host {rec['two_table_host_mse']:.3e} (F2J-on {rec['two_table_f2j_on_mse']:.3e}); corr(c_F2J, truth-host) {rec['corr_cF2J_ehost']:+.2f}")
    print("| arm | host MSE (nc) | F2J-on MSE (nc) | c_total: inside span | corr(c,truth-host) | corr(c⊥,truth-dep) | c_own rms | own inside span | corr(c_own, truth-2table host) | corr(c_own⊥, truth-dep) | ⊥ ceiling (% of deployed MSE) | energy removed from 2-table residual |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a, r in rec["arms"].items():
        t = r["total"]; ow = r.get("own")
        own = f"{ow['rms']:.2e} | {ow['frac_c_inside_cF2J_span']:.2f} | {ow['corr_c_residual']:+.2f} | {ow['corr_cperp_truthminusdeployed']:+.2f} | {ow['perp_best_removal_pct']:.1f}% | {ow['energy_removed_from_residual_pct']:+.1f}%" if ow else "— | — | — | — | — | —"
        print(f"| {a} | {r['host_mse']:.3e} | {r['f2j_on_mse']:.3e} | {t['frac_c_inside_cF2J_span']:.2f} | {t['corr_c_residual']:+.2f} | {t['corr_cperp_truthminusdeployed']:+.2f} | {own} |")

with open(os.path.join(HERE, "PERP_JUDGE_ENGINE.json"), "w") as fh:
    json.dump(out, fh, indent=1)
