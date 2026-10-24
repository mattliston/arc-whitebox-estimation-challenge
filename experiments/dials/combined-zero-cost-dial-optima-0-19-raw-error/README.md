# Combined zero-cost dial optima: -0.19% raw error on 28 networks

2026-10-19T21:30:00Z · experiment · adopted · measured

Each flat dial's parabola optimum is worth 0.01-0.1% alone and would never pass the fire bar, but the mechanisms are independent, so every optimum was stacked (k4 bands x0.9911/x1.0071/x0.9933, pair-law k3 by-layer x0.943, K31R row factor x1.04, fourth-order ages 1-2 coefficient x0.933, pair-law strengths x0.974, birth skew 0.434): raw -0.19% on 28 ranks, matching the predicted sum of -0.20%; as a release form on the fired physics -0.13% adjusted on 32 ranks with the same request profile.

Recorded measurement:

    mstack1 (28 ranks vs stk7 physics): raw -0.19% (lower 17/28), held-out-21 raw -0.33% (15/21), source7 worse; adjusted -0.10% all / -0.24% held-out; release form rel_mstack1 vs the fired l1314x physics (32 ranks): raw -0.11%, bill -0.02%, adjusted -0.13% (16/32), held-out-25 -0.19%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_mstack1_sci_v1.py](code/candidate_mstack1_sci_v1.py)
- [code/candidate_rel_mstack1_sci_v1.py](code/candidate_rel_mstack1_sci_v1.py)
- [code/candidate_rel_mstack1.py](code/candidate_rel_mstack1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6031](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6031)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6037](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6037)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6039](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6039)

<details>
<summary>All result files and records (4)</summary>

- [mstack1_dials_v1.txt](../../../instruments/dial-finite-difference-toolchain-arm-builder-parabola-optimum-per/mstack1_dials_v1.txt) · config
- [RESULT32_v1.json](RESULT32_v1.json) · result
- [RESULT32_v1-b5ac3946.json](RESULT32_v1-b5ac3946.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- shipped in: [Submission #337212: combined zero-cost dial optima on #337142, graded 1.4296259e-9](../../../submissions/337212/README.md)

[Research graph](../../../graph/index.html#node=2026-10-19-combined-zero-cost-dial-optima-0-19-raw-error) · [Copy provenance](../../../PROVENANCE.json)

