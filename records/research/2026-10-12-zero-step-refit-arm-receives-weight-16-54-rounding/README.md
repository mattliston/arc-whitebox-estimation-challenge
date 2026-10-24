# Zero-step refit arm receives weight 16.54 from rounding noise; zero-arm guard drops it

2026-10-12T18:26:49Z · incident · failed · measured

candidate generation 51 (S51)'s freeze gave the arm escal (the learned-edge scalars e.cov_factor + e.third_scale) a weight of 16.54, but its parameter file is bit-identical to the start point (gradient t +2.26 / -2.32 under the |t| > 2.5 keep rule, so the trial step was 0) and the column it multiplied was pure thread rounding (base evals at 4 or 8 threads, escal evals at 2, max prediction difference 4.77e-7; bit-identical on the all-2-thread set); 16.54 x 0 = 0, nothing was exported, and a re-solve without escal changed the other weights by at most 0.001 (leave-one-out (LOO) -0.186% both ways). From candidate generation 52 (S52) a zero-arm guard compares each arm's parameter file with the round's start point (28 arrays, dials, per-layer shifts, head) and drops bit-identical arms before sizing; escal was dropped by rule.

Recorded measurement:

    escal weight 16.54; arm_escal.pt = base max |diff| 0; A39 prediction difference 4.77e-7 (threads 4 / 8 vs 2), A41 bit-identical; escal alone LOO -0.000%; re-solve without escal LOO -0.186% (original -0.186%), model -0.217% (original -0.219%), weights changed <= 0.001; the full freeze: LOO -0.186%, model -0.219%, weights arrp 0.76, hb3 0.43, h17 0.29, diso_11_14 0.36, dk21n_7_10 1.02, drsy_11_14 0.28

## Programs

- [zero_arms_v1.py](zero_arms_v1.py)
- [fix_make_zero_v2.py](fix_make_zero_v2.py)
- [../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [../../../experiments/learned-correction/continued-training-lowers-held-out-error-only-0-240/code/otr_common.py](../../../experiments/learned-correction/continued-training-lowers-held-out-error-only-0-240/code/otr_common.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1612](../../TEAM_LOG_2026-10-01_to_10-22.md#L1612)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1614](../../TEAM_LOG_2026-10-01_to_10-22.md#L1614)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1617](../../TEAM_LOG_2026-10-01_to_10-22.md#L1617)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1631](../../TEAM_LOG_2026-10-01_to_10-22.md#L1631)

<details>
<summary>All result files and records (2)</summary>

- [FROZEN_POINT_v1.json](FROZEN_POINT_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-12-zero-step-refit-arm-receives-weight-16-54-rounding) · [Copy provenance](../../../PROVENANCE.json)

