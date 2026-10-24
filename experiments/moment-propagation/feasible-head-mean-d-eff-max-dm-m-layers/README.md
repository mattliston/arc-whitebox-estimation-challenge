# Feasible head mean d_eff = max(dm, -m) at layers 1-14 before covariance compensation: fresh -0.076%, OOS -0.075%

2026-10-14T15:02:32Z · experiment · adopted · measured

The feasible head mean (FHM) clips the learned head's mean displacement so the post-ReLU mean stays feasible, one fixed unit d_eff = max(dm, -m) at layers 1 to 14 before the unchanged RC/KC covariance compensation, with no tuned band or coefficient. It priced at zero bill and passed the strictly-negative fresh, out-of-sample and all-32 gates, becoming the accuracy part of the #336075 stack.

Recorded measurement:

    price MAX log bill 0.000044446%; fresh 16 pairs net -0.075739265% (SE 0.063045338%), PASS; OOS 16 log net -0.07465737478358218%; all 32 -0.07519831979670727%; combined FHM+dense fixed 6 incremental bill +0.000043736%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)
- [code/engineer/a62_modules/s49/s56ubq1tw_x/candidate.py](code/engineer/a62_modules/s49/s56ubq1tw_x/candidate.py)
- [code/release/guard_footprint_v3.py](code/release/guard_footprint_v3.py)
- [code/release/rusage.py](code/release/rusage.py)
- [code/scout_card/feasible_head_mean_held_astra_v1/build_feasible_mean_native_astra_v1.py](code/scout_card/feasible_head_mean_held_astra_v1/build_feasible_mean_native_astra_v1.py)
- [code/scout_card/feasible_head_mean_held_astra_v1/current_OFF_UNVALIDATED_astra_v1.py](code/scout_card/feasible_head_mean_held_astra_v1/current_OFF_UNVALIDATED_astra_v1.py)
- [code/scout_card/feasible_head_mean_held_astra_v1/current_ON_UNVALIDATED_astra_v1.py](code/scout_card/feasible_head_mean_held_astra_v1/current_ON_UNVALIDATED_astra_v1.py)
- [code/scout_card/feasible_head_mean_held_astra_v1/current_fragment_UNVALIDATED_astra_v1.py](code/scout_card/feasible_head_mean_held_astra_v1/current_fragment_UNVALIDATED_astra_v1.py)
- [code/scout_card/feasible_head_mean_held_astra_v1/finite_fhm_qualification_UNVALIDATED_v3.py](code/scout_card/feasible_head_mean_held_astra_v1/finite_fhm_qualification_UNVALIDATED_v3.py)
- [code/scout_card/feasible_head_mean_held_astra_v1/finite_fhm_science_HELD_v1.py](code/scout_card/feasible_head_mean_held_astra_v1/finite_fhm_science_HELD_v1.py)
- [code/scout_card/feasible_head_mean_held_astra_v1/native_fhm_qualification_judge_v1.py](code/scout_card/feasible_head_mean_held_astra_v1/native_fhm_qualification_judge_v1.py)
- [code/scout_card/feasible_head_mean_held_astra_v1/native_fhm_science_judge_HELD_v1.py](code/scout_card/feasible_head_mean_held_astra_v1/native_fhm_science_judge_HELD_v1.py)

<details>
<summary>Additional programs and dependencies (4)</summary>

- [code/scout_card/feasible_head_mean_held_astra_v1/owned_native_runner_v1.py](code/scout_card/feasible_head_mean_held_astra_v1/owned_native_runner_v1.py)
- [code/scout_native/certify_astra_bank_copy_v2.py](code/scout_native/certify_astra_bank_copy_v2.py)
- [code/scout_native/k4liab_s51k2tw/candidate.py](code/scout_native/k4liab_s51k2tw/candidate.py)
- [code/scout_native/run_cert_astra_bank_copy_v2.py](code/scout_native/run_cert_astra_bank_copy_v2.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2805](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2805)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2820](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2820)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2830](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2830)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2870](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2870)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2885](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2885)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2897](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2897)

<details>
<summary>All result files and records (3)</summary>

- [SCIENCE_PROTOCOL_HELD_v2.json](SCIENCE_PROTOCOL_HELD_v2.json) · record
- [UNUSED32_CLASSIFIED_PROPOSAL_v1.json](UNUSED32_CLASSIFIED_PROPOSAL_v1.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- shipped in: [Submission #336075: feasible head mean, dense covariance reuse and Gaussian T22 reuse on #335920, 1.51317e-9](../../../submissions/336075/README.md)
- ← shipped in: [Submission #336075: feasible head mean, dense covariance reuse and Gaussian T22 reuse on #335920, 1.51317e-9](../../../submissions/336075/README.md)

[Research graph](../../../graph/index.html#node=2026-10-14-feasible-head-mean-d-eff-max-dm-m-layers) · [Copy provenance](../../../PROVENANCE.json)

