#!/usr/bin/env python3
"""Brute-verify the F-form decomposition and the kappa211 plane routes.

TAG=missing-routes-verify  Tolerance 1e-12 relative, small n:
  V1 (ARM 1): explicit tensor push of K (entries K_ij/2 at the 6
      arrangements) vs the deployed channel spellings: k4 diagonal and k31
      must MATCH the deployed forms exactly (nothing missing), and the k22
      channel minus the deployed (W2^T K W2) must equal exactly 2F.
  V2 (ARM 2): per canonical group AND all groups combined, the explicit
      tensor push of kappa211 (entries /4 at the 12 arrangements, values
      from the raw FORMULA terms) vs (a) the banked diagonal S(a) spelling
      3 sum W2 W W kappa211 and (b) kappa211_plane_supplies' k31/k22
      (diagonals compared before zeroing is not needed - the tensor
      channels are read off-diagonal).
Writes VERIFY_MISSING_ROUTES.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from missing_routes_core import (ALL_GROUPS, K22_P1_COEF, K22_P2_COEF,
                                 K31_ROUTE_COEF, f_form_k22_missing,
                                 group_terms, k211_values,
                                 kappa211_plane_supplies, pair_pass,
                                 push_tensor, tensor_from_K,
                                 tensor_from_k211)


def rel_err(fast, brute):
    scale = max(float(np.abs(brute).max()), 1e-300)
    return float(np.abs(np.asarray(fast) - np.asarray(brute)).max() / scale)


def offdiag(M):
    out = np.array(M, dtype=np.float64)
    np.fill_diagonal(out, 0.0)
    return out


def random_case(rng, n, na):
    W = rng.standard_normal((n, na))
    raw = rng.standard_normal((n, n)) / np.sqrt(n)
    cov = raw @ raw.T + 0.5 * np.eye(n)
    d = 1.0 / np.sqrt(np.diag(cov))
    rho = cov * d[:, None] * d[None, :]
    c1 = rng.standard_normal((5, n))
    c2 = rng.standard_normal((5, n))
    return W, rho, c1, c2


def verify_arm1(rng, n, na):
    W = rng.standard_normal((n, na))
    raw = rng.standard_normal((n, n))
    K = raw + raw.T
    np.fill_diagonal(K, 0.0)
    W2 = W * W
    kappa = push_tensor(tensor_from_K(K), W)
    a_idx = np.arange(na)
    carried = W2.T @ K @ W2
    e_k4 = rel_err(kappa[a_idx, a_idx, a_idx, a_idx],
                   3.0 * np.diag(carried))
    deployed_k31 = 3.0 * (((W2.T @ K) * W.T) @ W)
    tensor_k31 = kappa[a_idx[:, None], a_idx[:, None], a_idx[:, None], a_idx[None, :]]
    e_k31 = rel_err(offdiag(tensor_k31), offdiag(deployed_k31))
    tensor_k22 = kappa[a_idx[:, None], a_idx[:, None], a_idx[None, :], a_idx[None, :]]
    missing = offdiag(tensor_k22) - offdiag(carried)
    e_k22 = rel_err(missing, f_form_k22_missing(W, K))
    e_pair = rel_err(pair_pass(K, W, W, W, W),
                     np.einsum("ij,ia,ib,ja,jb->ab", K, W, W, W, W))
    return {"k4diag_vs_deployed": e_k4, "k31_vs_deployed": e_k31,
            "k22_missing_vs_2F": e_k22, "pair_pass_vs_einsum": e_pair}


def verify_arm2(rng, n, na, groups):
    W, rho, c1, c2 = random_case(rng, n, na)
    W2 = W * W
    R = rho - np.eye(n)
    terms = [t for tag in groups for t in group_terms(tag)]
    values = k211_values(R, c1, c2, terms, n)
    kappa = push_tensor(tensor_from_k211(values, n), W)
    a_idx = np.arange(na)
    diag_banked = np.zeros(na)
    for (i, j, k), value in values.items():
        diag_banked += 3.0 * W2[i] * W[j] * W[k] * value
    e_diag = rel_err(kappa[a_idx, a_idx, a_idx, a_idx], diag_banked)
    k31_fast, k22_fast = kappa211_plane_supplies(W, rho, c1, c2,
                                                 groups=groups)
    tensor_k31 = kappa[a_idx[:, None], a_idx[:, None], a_idx[:, None], a_idx[None, :]]
    tensor_k22 = kappa[a_idx[:, None], a_idx[:, None], a_idx[None, :], a_idx[None, :]]
    e_k31 = rel_err(offdiag(tensor_k31), k31_fast)
    e_k22 = rel_err(offdiag(tensor_k22), k22_fast)
    return {"diag_vs_banked_S": e_diag, "k31_planes": e_k31,
            "k22_planes": e_k22}


def main():
    checks = {"arm1": [], "arm2_per_group": {}, "arm2_combined": []}
    worst = 0.0
    for seed in (1, 2, 3):
        rng = np.random.default_rng(20260826 * 10 + seed)
        row = verify_arm1(rng, n=7, na=5)
        checks["arm1"].append(row)
        worst = max(worst, *row.values())
        print(json.dumps({"arm1_seed": seed, **{k: f"{v:.2e}" for k, v in row.items()}}), flush=True)

    rng = np.random.default_rng(20260826)
    for tag in ALL_GROUPS:
        row = verify_arm2(rng, n=6, na=4, groups=(tag,))
        checks["arm2_per_group"][tag] = row
        worst = max(worst, *row.values())
        print(json.dumps({"group": tag, **{k: f"{v:.2e}" for k, v in row.items()}}), flush=True)

    for seed in (11, 12):
        rng = np.random.default_rng(20260826 * 100 + seed)
        row = verify_arm2(rng, n=7, na=5, groups=ALL_GROUPS)
        checks["arm2_combined"].append(row)
        worst = max(worst, *row.values())
        print(json.dumps({"combined_seed": seed, **{k: f"{v:.2e}" for k, v in row.items()}}), flush=True)

    passed = worst <= 1e-12
    result = {"schema": "missing-routes-verify-v1", "tolerance_rel": 1e-12,
              "worst_rel_err": worst, "pass": passed, "checks": checks}
    (HERE / "VERIFY_MISSING_ROUTES.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"worst": worst, "pass": passed}))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
