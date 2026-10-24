# Direct third-cumulant products confirmed rank-cap error was physical; float16 offered no billing saving

2026-09-04 · experiment · informational · measured

BILLING: flopscope bills f16 matmuls exactly as f32 (1.0 x 2N^3; f64 2.0x) - no precision lever below f32. SYNTHESIS (15:20, for Matt): under adjusted = raw x util, the phase-2 blocks compose to ~1.45e-8 (chain card: in-card fourth-cumulant two-pair plane (K22) chain replacing fourth-cumulant canonical-polyadic carrier (CP4) + quartic star, f32 covariance, ~0.50 util, raw ~2.9-3.1e-8) = -20% vs the bank; 4e-9 at 0.50 needs raw 8e-9 (the half-truth per-unit state class), and a 0.2-util card needs a third-order representation 4x cheaper than the third-cumulant canonical-polyadic carrier (CP3) carrier, which the corpus does not contain. Packaging the chain card next: fnp port of k22_chain into the estimator_cap.py lineage.

Recorded measurement:

    - T-91 THE CAP EXPLOSION IS PHYSICS, NOT THE ARENA (RETRO_PLAIN_CP3=1: plain GEMM push/read in place of the sealed Winograd arena; r35, independent 4M): cap 16384 plain reproduces the deployed card exactly (F2J on 2.7038e-8, off 8.622e-8) at util 0.771 (the arena saves 0.17); lean card (CP4 cap 256, f32 covariance) at cap 4096 -> host 3.30e-7 (util 0.447), 2048 -> 8.90e-7 (0.379), 1024 -> 1.64e-6 (0.335); with the chain 9.3e-7 / 1.65e-6. Dropping carrier terms breaks the third-order cancellation structure (T-51's partial-carry law); the carrier's full rank is load-bearing, so the CP3 push (0.27 arena / 0.47 plain) is this architecture's cost floor. A competitor-class 0.2-util card needs a different third-order representation.
    
    - T-91b BILLING: flopscope bills f16 matmuls exactly as f32 (1.0 x 2N^3; f64 2.0x) - no precision lever below f32. SYNTHESIS (15:20, for Matt): under adjusted = raw x util, the phase-2 blocks compose to ~1.45e-8 (chain card: in-card K22 chain replacing CP4 + quartic star, f32 covariance, ~0.50 util, raw ~2.9-3.1e-8) = -20% vs the bank; 4e-9 at 0.50 needs raw 8e-9 (the half-truth per-unit state class), and a 0.2-util card needs a third-order representation 4x cheaper than the CP3 carrier, which the corpus does not contain. Packaging the chain card next: fnp port of k22_chain into the estimator_cap.py lineage.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_d11_util_halving_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/estimator_cap.py](code/estimator_cap.py)

## Evidence

- [LANE_g3_tracker.md#L762](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L762)
- [LANE_g3_tracker.md#L763](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L763)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-direct-third-cumulant-products-confirmed-rank-cap-error-physical) · [Copy provenance](../../../PROVENANCE.json)

