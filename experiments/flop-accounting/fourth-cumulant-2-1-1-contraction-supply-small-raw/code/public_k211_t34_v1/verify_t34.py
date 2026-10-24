#!/usr/bin/env python3
"""Brute-verify the T3+T4 kappa211 GEMM chains at small n.

TAG=k211-t34-verify  Three checks, tolerance 1e-12 relative:
  1. fast T3+T4 vs triple-loop brute over the raw FORMULA term list,
     n in {24, 32, 40}, three seeds each, random W / correlation-like rho /
     arbitrary functional channels (the identity is algebraic, so arbitrary
     c-vectors are the strongest test);
  2. the order-2 component (fast full<=4 minus fast T3+T4) vs the banked
     supply_core.kappa211_supply, proving the class machinery reproduces
     the audited T2 implementation exactly;
  3. full <=4 fast (include_t2) vs brute orders=(2,3,4).
Writes VERIFY_T34.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(P2 / "public_k211_supply_v1"))
from t34_supply_core import (T2_GROUPS, T34_CHAINSTAR_GROUPS,
                             kappa211_t34_supply, kappa211_supply_brute)
from supply_core import kappa211_supply as banked_t2_supply


def brute_subset(W, rho, c1, c2, orders, keep):
    """Brute restricted to terms passing keep(p, q, t)."""
    import t34_supply_core as core
    saved = core.FORMULA_TERMS
    core.FORMULA_TERMS = tuple(t for t in saved if keep(t[0], t[1], t[2]))
    try:
        return kappa211_supply_brute(W, rho, c1, c2, orders=orders)
    finally:
        core.FORMULA_TERMS = saved


def random_case(rng, n, na):
    W = rng.standard_normal((n, na))
    raw = rng.standard_normal((n, n)) / np.sqrt(n)
    cov = raw @ raw.T + 0.5 * np.eye(n)
    d = 1.0 / np.sqrt(np.diag(cov))
    rho = cov * d[:, None] * d[None, :]
    c1 = rng.standard_normal((5, n))
    c2 = rng.standard_normal((5, n))
    return W, rho, c1, c2


def rel_err(fast, brute):
    scale = max(float(np.abs(brute).max()), 1e-300)
    return float(np.abs(fast - brute).max() / scale)


def main():
    checks = []
    worst = {"t34": 0.0, "t2_banked": 0.0, "full_le4": 0.0}
    for n in (24, 32, 40):
        for seed in (1, 2, 3):
            rng = np.random.default_rng(20260825 * 100 + n * 10 + seed)
            W, rho, c1, c2 = random_case(rng, n, na=7)
            started = time.perf_counter()

            fast34 = kappa211_t34_supply(W, rho, c1, c2, include_t2=False)
            brute34 = kappa211_supply_brute(W, rho, c1, c2, orders=(3, 4))
            e34 = rel_err(fast34, brute34)

            fast_full = kappa211_t34_supply(W, rho, c1, c2, include_t2=True)
            brute_full = kappa211_supply_brute(W, rho, c1, c2, orders=(2, 3, 4))
            efull = rel_err(fast_full, brute_full)

            fast_t2 = fast_full - fast34
            banked = banked_t2_supply(W, rho, c1[0], c1[1], c1[2], c2[1], c2[2])
            et2 = rel_err(fast_t2, banked)

            fast_cs = kappa211_t34_supply(
                W, rho, c1, c2, groups=T2_GROUPS + T34_CHAINSTAR_GROUPS)
            brute_cs = brute_subset(W, rho, c1, c2, (2, 3, 4),
                                    lambda p, q, t: min(p, q, t) == 0)
            ecs = rel_err(fast_cs, brute_cs)

            fast_tri = kappa211_t34_supply(W, rho, c1, c2, groups=("TRI",))
            brute_tri = brute_subset(W, rho, c1, c2, (3, 4),
                                     lambda p, q, t: min(p, q, t) >= 1)
            etri = rel_err(fast_tri, brute_tri)

            worst["t34"] = max(worst["t34"], e34)
            worst["full_le4"] = max(worst["full_le4"], efull)
            worst["t2_banked"] = max(worst["t2_banked"], et2)
            worst["chainstar_subset"] = max(worst.get("chainstar_subset", 0.0), ecs)
            worst["triangle_subset"] = max(worst.get("triangle_subset", 0.0), etri)
            checks.append({
                "n": n, "seed": seed,
                "rel_err_t34_vs_brute": e34,
                "rel_err_full_le4_vs_brute": efull,
                "rel_err_t2_vs_banked_supply_core": et2,
                "rel_err_chainstar_subset_vs_brute": ecs,
                "rel_err_triangle_subset_vs_brute": etri,
                "seconds": time.perf_counter() - started,
            })
            print(json.dumps(checks[-1]), flush=True)

    passed = all(v <= 1e-12 for v in worst.values())
    result = {
        "schema": "k211-t34-verify-v1",
        "tolerance_rel": 1e-12,
        "worst": worst,
        "pass": passed,
        "checks": checks,
    }
    (HERE / "VERIFY_T34.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"worst": worst, "pass": passed}, indent=1))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
