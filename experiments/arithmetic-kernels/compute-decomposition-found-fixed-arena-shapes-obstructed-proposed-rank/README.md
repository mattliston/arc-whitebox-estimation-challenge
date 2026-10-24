# Compute decomposition found fixed arena shapes obstructed proposed rank-cap schedules

2026-09-01 · experiment · informational · measured

CAP LADDER BLOCKED BY THE ARENA: regenerated arenas at 8192/7680/4096/3840 (read = push-1024, -2048, or the sealed 15:14 block ratio) all crash on block-shape mismatches in _cp3_winograd_arena_depth4; only the sealed 15360/14336 and the T-51 30720/28672 layouts run. Decision (14:50): package the ~0.50-util chain card (in-card fourth-cumulant two-pair plane (K22) chain replacing fourth-cumulant canonical-polyadic carrier (CP4) + the quartic star; f32 covariance; refit operator or fitted transport-energy residual correction (F2J) off), expected adjusted 1.35-1.55e-8 (-13..-24% vs the bank); the competitor's 4e-9 class is out of reach of this architecture's third order.

Recorded measurement:

    - T-90 COST CENSUS AND THE CAP DIAL ([instruction removed]). The deployed card's 0.600 (CENSUS_RAW.json, r35): CP3 Winograd arena push 0.27 (grows with depth: 1.6e10 at L3 -> 7.4e10 at L14), CP4 read + push 0.106, quartic star push 0.082, covariance push f64 0.059, cubic star push 0.055, pair-series einsum 0.018, operator ~0. Cutting the CP3 cap is NOT a dial in phase 2: cap 4096 explodes the raw 10x (old card 3.36e-7 F2J off, chain host 3.32e-7) at util 0.456; caps 2048/1024/512 0.41/0.39/0.39 and 9e-7..2e-6; a matching regenerated arena (RETRO_ARENA_PUSH/READ) crashes on read-shape mismatches at caps 8192/4096. A ~0.20-util card = covariance (f32) + in-card K22 chain (~0.12) + a PLANE-ONLY third order (no CP3/CP4/quartic star) = the competitor's compute class; its raw is what run_plane_recursion.py graded. Looking that up.
    
    - T-90b PLANE-ONLY THIRD ORDER IS DEAD (tracker line 18 / 332: the standalone plane recursion is 14x worse; the pushed plane is a cancellation between plane-slice terms and distinct entries, removing the distinct content breaks it). So the CP3 carrier is load-bearing and its cost (0.27 at cap 16384) sets our architecture's floor: with the chain replacing CP4 + the quartic star (-0.19 + 0.10) and f32 covariance (-0.03), the card is ~0.50 at cap 16384 (raw ~3.1e-8 F2J off -> adjusted ~1.5e-8, -13% vs the bank; with the refit operator ~1.35e-8, -24%). A cap-4096 CP3 with a CORRECT drop schedule (not the sealed arena's) would give ~0.30; testing whether the cap explosion (T-90) is the arena schedule (relaxed arena, read = push - 2048).
    
    - T-90c CAP LADDER BLOCKED BY THE ARENA: regenerated arenas at 8192/7680/4096/3840 (read = push-1024, -2048, or the sealed 15:14 block ratio) all crash on block-shape mismatches in _cp3_winograd_arena_depth4; only the sealed 15360/14336 and the T-51 30720/28672 layouts run. Decision (14:50): package the ~0.50-util chain card (in-card K22 chain replacing CP4 + the quartic star; f32 covariance; refit operator or F2J off), expected adjusted 1.35-1.55e-8 (-13..-24% vs the bank); the competitor's 4e-9 class is out of reach of this architecture's third order.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/p2_width1024_depth16/public_g19_attribution_v1/run_plane_recursion.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_plane_recursion.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_attribution.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_attribution.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_chaos1024_shallow.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_chaos1024_shallow.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_vresid_field.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_vresid_field.py)
- [code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py)
- [code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py](code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py)

## Evidence

- [LANE_g3_tracker.md#L759](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L759)
- [LANE_g3_tracker.md#L760](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L760)
- [LANE_g3_tracker.md#L761](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L761)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-01-compute-decomposition-found-fixed-arena-shapes-obstructed-proposed-rank) · [Copy provenance](../../../PROVENANCE.json)

