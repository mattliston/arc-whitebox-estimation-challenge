# Innovation-target and joint propagated-feature ridge fits fail final-objective cross-validation

2026-10-15T17:49:17Z · experiment · rejected · measured

Two fixed saved-data fits failed their final objective: the innovation-target correction had too few negative folds, while the joint 760-feature correction increased error and projected score.

Recorded measurement:

    innovation-target final -.012422%,3/8folds; joint760 +.045916%raw,projected+.053920%net;6.38s675MB

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_joint_propagated_feature_ridge_v1/fit_joint_propagated_HELD_v1.py](code/astra_joint_propagated_feature_ridge_v1/fit_joint_propagated_HELD_v1.py)
- [code/astra_native_intermediate_innovation_refit_v1/fit_innovation_blocked_HELD_v1.py](code/astra_native_intermediate_innovation_refit_v1/fit_innovation_blocked_HELD_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3451](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3451)

<details>
<summary>All result files and records (12)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [JOINT_PROPAGATED_BLOCK_CV_RESULT_v1.json](JOINT_PROPAGATED_BLOCK_CV_RESULT_v1.json) · result
- [ACTUAL_FIT_EXIT_v1.json](ACTUAL_FIT_EXIT_v1.json) · result
- [FIT_GUARD_v1.json](FIT_GUARD_v1.json) · result
- [FIT_PLAN_v1.json](FIT_PLAN_v1.json) · result
- [FIT_RULE_FROZEN_v1.json](FIT_RULE_FROZEN_v1.json) · result
- [OUTPUT_ONLY_DEPLOYMENT_AMEND_v1.json](OUTPUT_ONLY_DEPLOYMENT_AMEND_v1.json) · result
- [INNOVATION_BLOCK_CV_RESULT_v1.json](INNOVATION_BLOCK_CV_RESULT_v1.json) · result
- [ACTUAL_FIT_EXIT_v1-2837486a.json](ACTUAL_FIT_EXIT_v1-2837486a.json) · result
- [FIT_GUARD_v1-d8fc898e.json](FIT_GUARD_v1-d8fc898e.json) · result
- [FIT_PLAN_v1-09e9acd8.json](FIT_PLAN_v1-09e9acd8.json) · result
- [FIT_RULE_FROZEN_v1-38a94a98.json](FIT_RULE_FROZEN_v1-38a94a98.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-15-innovation-target-joint-propagated-feature-ridge-fits-fail-final) · [Copy provenance](../../../PROVENANCE.json)

