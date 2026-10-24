# Learned mean-shift covariance coefficient x1.2 / x0.8: +0.082% / +0.010% adjusted

2026-10-20T16:14:50Z · experiment · rejected · measured

The global coefficient controlling the learned mean shift rank-two covariance correction was at a flat optimum on 32 networks. Moving it by 20% in either direction did not improve adjusted score.

Recorded measurement:

    - [fable-lead] 2026-10-20T16:14:50Z kc_p (kappa_C fraction x1.2 at every layer) DEAD on 32 ranks: adjusted +0.082% (se 0.082, lower 14/32), raw +0.083%, bill 0; heldout25 +0.078%. kc_m runs next on the single lane after the early-layer covariance probes.
    - [fable-lead] 2026-10-20T16:38:00Z kc_m (kappa_C x0.8) FLAT on 32 ranks: +0.010% (se 0.082, lower 13/32); the global KC knob is at its optimum (x1.2 +0.08%). POST-COVARIANCE PROBES (probe_cpost_dump_v1.py, nets 35/100): the common-mode deficit m^T dC m / m^T Ct m is -0.13% already at L1 (post of the exact Gaussian layer: the pair map's own output), -0.16% L2-3, -0.23..-0.35% L4-7 on net 35 then -0.25..-0.34% through L13; on net 100 it peaks -0.22% at L6 and shrinks to -0.06% by L13 (the late corrections compensate on 100, not on 35: net-specific). Agent B8 (IDEAS_v2): along the common mode the Gaussian pair series is exact; the NON-GAUSSIAN transfer (cumulant-carried: b-carrier, roster K21 cross terms, k22) runs 0.5-1.4% short at L1-8 and matches truth from L9 (where the edge/GN knobs were fitted); budget of the early injection: cumulant transfer 40-60%, GNN rank-2 (KC) 20-40%, GN dcov 15-20%. Prediction: a per-layer GSK split reads gsk_early_p NEGATIVE, gsk_late_p ~0. BUILT: _GSK per layer via a _gq wrapper keyed on _PHYS_LAYER (tails gsk_early_p 0.99 / gsk_early_m 0.81 at L1-7, gsk_late_p 0.99 at L8-14; base _GSK = 0.9, so the earlier global gsk arms were rebuilt: p 1.0 / m 0.8). Queue now: kc_late_m (running), kc_late_p, kc_early_p, gsk_early_p, iso_early_m, kc_early_m, gsk_early_m, gsk_late_p, gsk_p, gsk_m, rc_m, rc_p, then the remaining early arms; the early CMODE profiles (full/half) run in the gap after kc_late_m.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_kc_m_sci_v1.py](code/candidate_kc_m_sci_v1.py)
- [code/candidate_kc_p_sci_v1.py](code/candidate_kc_p_sci_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6200](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6200)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6204](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6204)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-learned-mean-shift-covariance-coefficient-x1-2-x0-8) · [Copy provenance](../../../PROVENANCE.json)

