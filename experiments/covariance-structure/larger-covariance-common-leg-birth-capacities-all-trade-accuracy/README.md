# Larger covariance, common-leg and birth capacities all trade accuracy below one-to-one with bill

2026-10-16T06:05:04Z · experiment · rejected · measured

Every fixed eight-network capacity increase missed the incremental adjusted-score criterion. Several exposed knobs were dead or already saturated in the live call chain.

Recorded measurement:

    star8MSE-.285%,bill+.619%,rate.46;common320-.459%/+1.542%,rate.30;earlybirth.0075-.322%/+1.226%,rate.26;latebirth.005-.177%/+1.327%,rate.13

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_grow_v1/grade_grow_v1.py](code/opus_grow_v1/grade_grow_v1.py)
- [code/opus_grow_v1/build_variants_v1.py](code/opus_grow_v1/build_variants_v1.py)
- [code/opus_grow_v1/build_variants_v2.py](code/opus_grow_v1/build_variants_v2.py)
- [code/opus_grow_v1/cand_s2tr_grow_bfe75_v1.py](code/opus_grow_v1/cand_s2tr_grow_bfe75_v1.py)
- [code/opus_grow_v1/cand_s2tr_grow_bfe_v1.py](code/opus_grow_v1/cand_s2tr_grow_bfe_v1.py)
- [code/opus_grow_v1/cand_s2tr_grow_bfl_v1.py](code/opus_grow_v1/cand_s2tr_grow_bfl_v1.py)
- [code/opus_grow_v1/cand_s2tr_grow_f4c3_v1.py](code/opus_grow_v1/cand_s2tr_grow_f4c3_v1.py)
- [code/opus_grow_v1/cand_s2tr_grow_jord_v1.py](code/opus_grow_v1/cand_s2tr_grow_jord_v1.py)
- [code/opus_grow_v1/cand_s2tr_grow_k4l3f_v1.py](code/opus_grow_v1/cand_s2tr_grow_k4l3f_v1.py)
- [code/opus_grow_v1/cand_s2tr_grow_lcr320_v1.py](code/opus_grow_v1/cand_s2tr_grow_lcr320_v1.py)
- [code/opus_s2tr_v1/candidate_S2tr_v1.py](code/opus_s2tr_v1/candidate_S2tr_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3941](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3941)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3917](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3917)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3933](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3933)

<details>
<summary>All result files and records (3)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [GROW_GRADE_latest.json](../../dials/increasing-learned-covariance-higher-order-strengths-worsens-current-native/GROW_GRADE_latest.json) · result
- [GROW_RESULT_v1.json](../../dials/increasing-learned-covariance-higher-order-strengths-worsens-current-native/GROW_RESULT_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-larger-covariance-common-leg-birth-capacities-all-trade-accuracy) · [Copy provenance](../../../PROVENANCE.json)

