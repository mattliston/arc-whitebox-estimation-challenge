#!/usr/bin/env python3
"""P2R0F3A design probe: exact synthetic billing of the K22-residual carrier.

TAG=k22-carrier-billing  width=1024  synthetic-only (targets_read=0)

Measures, with the canonical flopscope meter at production shape n=1024, the
billed FLOPs of every component of the proposed K22-residual second-star
carrier (rank-factored form Q ~= V diag(lam) V^T):

  emission (per reset):  3-sweep subspace iteration on a dense symmetric
                         residual, small-r eigh finish
  transport (per layer): fixed channels    - factored W2^T Q W2, S1/S2
                         per-rank channel  - G_r = W^T diag(v_r) W (both a
                         plain-GEMM spelling and an einsum spelling, to
                         capture any symmetric orbit-quotient discount)
  consumption:           plane adds into the existing (k4,k31,k22) planes

Reports totals for r in {2,4,8,16} over 14 pushed layers + 14 resets and
compares each against the three budget envelopes of record: raw headroom
66.5e9, + conditional-repair dividend 275e9, + star-compression dividend
~200e9. Pure cost measurement: no truth data, no host modification.
"""
import json
import hashlib
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np

N = 1024
LAYERS = 14
RANKS = (2, 4, 8, 16)
HEADROOM_RAW = 66.5e9
DIVIDEND_REPAIR = 275.0e9
DIVIDEND_STAR_COMPRESSION = 200.0e9

rng = np.random.default_rng(11)


def fresh(*shape):
    return fnp.asarray(rng.standard_normal(shape).astype(np.float32),
                       dtype=fnp.float32)


def metered(fn):
    with flops.BudgetContext(10**14) as budget:
        fn()
        return int(budget.flops_used)


def bill_emission(r):
    """Dense symmetric residual -> rank-r factors, no eigh on n."""
    R = fresh(N, N)
    R = 0.5 * (R + fnp.transpose(R))
    probe = fresh(N, r)

    def run():
        y = R @ probe
        for _ in range(2):
            b = fnp.transpose(y) @ y                      # r x r gram
            fnp.linalg.cholesky(b + fnp.eye(r, dtype=fnp.float32))
            y = R @ y
        small = fnp.transpose(y) @ (R @ y)                # r x r projected
        fnp.linalg.eigh(0.5 * (small + fnp.transpose(small)))
    return metered(run)


def bill_transport_fixed(r):
    """Per-layer fixed channels with Q in factored form V lam V^T."""
    W = fresh(N, N)
    W2 = W * W
    V = fresh(N, r)
    lam = fresh(r)

    def run():
        vw2 = fnp.transpose(W2) @ V                       # n^2 r
        (vw2 * lam[None, :]) @ fnp.transpose(vw2)         # k22' family part
        qv = V @ (lam[:, None] * fnp.transpose(V))        # dense Q, n^2 r
        s_one = (fnp.transpose(W2) @ qv * fnp.transpose(W)) @ W
        s_two = (fnp.transpose(W) * fnp.transpose(qv @ W2)) @ W
        s_one + fnp.transpose(s_two)
    return metered(run)


def bill_transport_per_rank_gemm():
    """G = W^T diag(v) W, plain GEMM spelling."""
    W = fresh(N, N)
    v = fresh(N)

    def run():
        fnp.transpose(W) @ (v[:, None] * W)
    return metered(run)


def bill_transport_per_rank_einsum_smallgrid():
    """Symmetric einsum spelling for the orbit-quotient discount.

    The 0.12.0 client raises SymmetryError(max_deviation=inf) for this
    einsum at n>=128 (works at n=64), so the discount is measured at 64
    and recorded as evidence only; planning uses the GEMM price.
    """
    n = 64
    W = fresh(n, n)
    v = fresh(n)

    def run_einsum():
        fnp.einsum("ia,i,ib->ab", W, v, W)

    def run_gemm():
        fnp.transpose(W) @ (v[:, None] * W)
    return metered(run_einsum), metered(run_gemm)


def bill_pairing_consumption():
    """G-dependent consumed pieces per rank: diag vector + G o G add."""
    G = fresh(N, N)
    gd = fresh(N)

    def run():
        gd * gd                                            # k4 channel
        G * G                                              # k22 pairing form
        G * gd[:, None]                                    # k31 channel
    return metered(run)


def bill_plane_adds():
    a, b = fresh(N, N), fresh(N, N)
    v1, v2 = fresh(N), fresh(N)

    def run():
        a + b
        a + b
        v1 + v2
    return metered(run)


def main():
    per_rank_gemm = bill_transport_per_rank_gemm()
    einsum64, gemm64 = bill_transport_per_rank_einsum_smallgrid()
    per_rank = per_rank_gemm
    pairing_consume = bill_pairing_consumption()
    plane_adds = bill_plane_adds()

    report = {
        "per_layer_components": {
            "per_rank_G_gemm_spelling_n1024": per_rank_gemm,
            "per_rank_G_planning_price": per_rank,
            "per_rank_pairing_consumption": pairing_consume,
            "plane_adds": plane_adds,
        },
        "symmetric_einsum_orbit_discount": {
            "einsum_n64": einsum64,
            "gemm_n64": gemm64,
            "ratio": einsum64 / gemm64,
            "client_bug": (
                "fnp.einsum('ia,i,ib->ab', W, v, W) raises "
                "SymmetryError(max_deviation=inf) at n>=128 in flopscope "
                "0.12.0 local backend (works at n=64; the GEMM spelling of "
                "the same map is symmetric to 3e-8 at every size). If the "
                "einsum spelling worked at n=1024 the orbit quotient would "
                "roughly halve the per-rank transport price. Report to "
                "arc-whestbench@aicrowd.com per the Phase-2 rules "
                "invitation; do not work around the accounting."
            ),
        },
        "per_rank_totals": {},
    }
    for r in RANKS:
        emission = bill_emission(r)
        fixed = bill_transport_fixed(r)
        per_layer = fixed + r * (per_rank + pairing_consume) + plane_adds
        total = LAYERS * (per_layer + emission)
        report["per_rank_totals"][f"r{r}"] = {
            "emission_per_reset": emission,
            "transport_fixed_per_layer": fixed,
            "per_layer_total": per_layer,
            "campaign_total_14_layers": total,
            "fits_raw_headroom_66.5e9": total <= HEADROOM_RAW,
            "fits_with_repair_dividend_341.5e9": total <= HEADROOM_RAW + DIVIDEND_REPAIR,
            "fits_with_both_dividends_541.5e9": total
            <= HEADROOM_RAW + DIVIDEND_REPAIR + DIVIDEND_STAR_COMPRESSION,
        }

    payload = {
        "schema": "p2r0f3a-k22-carrier-billing-v1",
        "node": "P2R0F3A",
        "status": "COMPLETE_SYNTHETIC_NO_DATA",
        "width": N,
        "layers_billed": LAYERS,
        "flopscope_version": flops.__version__,
        "envelopes": {
            "raw_headroom": HEADROOM_RAW,
            "conditional_repair_dividend": DIVIDEND_REPAIR,
            "star_compression_dividend_contingent": DIVIDEND_STAR_COMPRESSION,
        },
        "report": report,
        "targets_read": 0,
    }
    out = Path(__file__).resolve().parent / "F3A_K22_CARRIER_BILLING.json"
    assert not out.exists(), "append-only"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.rename(out)
    print(json.dumps(report, indent=1))
    print("receipt_sha256:", hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
