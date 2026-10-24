# Exact-term mixture improves Torch error by 0.48% but worsens native error by 0.18%

2026-10-02T08:53Z · experiment · rejected · measured

A linear mixture of the rule-A column shift with the exact hub coefficients S1 and S2 (pair constants fixed at birth) beat the shift plus kurtosis term by 0.46-0.51% on two sets of 24 held-out ranks (the mixture model predicted the real runs within 0.07%). The native port has parity with the torch module on four networks and costs +0.13% bill and about 850 requests, but at the physicist's literal strengths on the S38 dials it gave +0.18% on ten fresh networks: candidate 2's dials had taken part of what the exact terms gave. The module entered S39 only through the joint dial solve, with the exact S2 term at 0.444 / 0.171 and no exact S1.

Recorded measurement:

    24 ranks: F4 0.58 + K4L 0.73 -5.58% [-6.37,-4.78]; mixture + K4L 0.71 -6.06% [-6.82,-5.26]; vs F4+K4L -0.51% [-1.02,+0.06] (17 of 24); ranks 25-48: -0.46% [-0.92,+0.04]; 48 ranks -0.48% [-0.83,-0.12] (34 of 48, mean -0.49%, se 0.18); native parity 474 / 648 cosine 0.88-0.94, MBK vs FK -0.44% / -0.36% native vs +0.13% / -0.80% torch; 35 / 651: -2.46% / -1.58% native vs -2.66% / -1.89% torch; bill +0.130% / +0.127%, logged calls +856 / +846; preview on S38 stack, ten fresh: MSE +0.18% [-0.40,+0.91], bill +0.133%, adjusted +0.33%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_reexam_20261001_v1/physicist/f4b.py](code/opus_reexam_20261001_v1/physicist/f4b.py)
- [code/opus_reexam_20261001_v1/physicist/t9_mix.py](code/opus_reexam_20261001_v1/physicist/t9_mix.py)
- [code/opus_reexam_20261001_v1/physicist/t4_fr48.py](code/opus_reexam_20261001_v1/physicist/t4_fr48.py)
- [code/opus_reexam_20261001_v1/physicist/f4c_native_ref.py](code/opus_reexam_20261001_v1/physicist/f4c_native_ref.py)
- [code/opus_reexam_20261001_v1/engineer/a27_f4c/fragment_f4c_v1.py](code/opus_reexam_20261001_v1/engineer/a27_f4c/fragment_f4c_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a27_f4c/unit_f4c.py](code/opus_reexam_20261001_v1/engineer/a27_f4c/unit_f4c.py)
- [code/opus_reexam_20261001_v1/engineer/a27_f4c/run_parity.sh](code/opus_reexam_20261001_v1/engineer/a27_f4c/run_parity.sh)
- [code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)

<details>
<summary>Additional programs and dependencies (19)</summary>

- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/native_parameters.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/native_parameters.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/train_segment.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/train_segment.py)
- [code/fable_analyst_v1/s2card.py](code/fable_analyst_v1/s2card.py)
- [code/fable_analyst_v1/s6bcard.py](code/fable_analyst_v1/s6bcard.py)
- [code/fable_covhunter_v1/covstep.py](code/fable_covhunter_v1/covstep.py)
- [code/fable_covhunter_v1/covstep_s2.py](code/fable_covhunter_v1/covstep_s2.py)
- [code/fable_covhunter_v1/gn3_knobs.py](code/fable_covhunter_v1/gn3_knobs.py)
- [code/fable_covhunter_v1/s6bcard_joint2.py](code/fable_covhunter_v1/s6bcard_joint2.py)
- [code/fable_covhunter_v1/s6bcard_joint2t.py](code/fable_covhunter_v1/s6bcard_joint2t.py)
- [code/fable_engineer_v1/gn_linear_hook.py](code/fable_engineer_v1/gn_linear_hook.py)
- [code/fable_engineer_v1/join_nystrom_torch.py](code/fable_engineer_v1/join_nystrom_torch.py)
- [code/fable_engineer_v1/pair_series_torch.py](code/fable_engineer_v1/pair_series_torch.py)
- [code/fable_physicist_v1/refit6b/lrstar_s1zero_hook.py](code/fable_physicist_v1/refit6b/lrstar_s1zero_hook.py)
- [code/fable_physicist_v1/refit6b/s6bcard_loop5js.py](code/fable_physicist_v1/refit6b/s6bcard_loop5js.py)
- [code/fable_porter_v1/star_nys_test.py](code/fable_porter_v1/star_nys_test.py)
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_bfix.py](code/fable_refit_v1/model_bank_bfix.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)
- [code/opus_reexam_20261001_v1/physicist/pcard.py](code/opus_reexam_20261001_v1/physicist/pcard.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L122](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L122)
- [TEAM_LOG_2026-10-01_to_10-22.md#L154](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L154)
- [TEAM_LOG_2026-10-01_to_10-22.md#L170](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L170)
- [TEAM_LOG_2026-10-01_to_10-22.md#L181](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L181)
- [TEAM_LOG_2026-10-01_to_10-22.md#L184](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L184)
- [TEAM_LOG_2026-10-01_to_10-22.md#L189](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L189)
- [TEAM_LOG_2026-10-01_to_10-22.md#L193](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L193)
- [TEAM_LOG_2026-10-01_to_10-22.md#L214](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L214)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L457](../../../records/SESSION_NOTES_2026-10-01_to_10-03.md#L457)
- [RESULT.md#L1344](../../../records/research/2026-10-01-kink-row-third-order-terms-account-6374-one-step/RESULT.md#L1344)
- [RESULT.md#L1359](../../../records/research/2026-10-01-kink-row-third-order-terms-account-6374-one-step/RESULT.md#L1359)
- [RESULT.md#L992](../../arithmetic-kernels/half-zero-strip-kernel-odd-32-dimensions-lowers-charged/RESULT.md#L992)

<details>
<summary>All result files and records (5)</summary>

- [quick_s38_f4c_preview_fresh_a.json](quick_s38_f4c_preview_fresh_a.json) · result
- [RESULT.md](../../arithmetic-kernels/half-zero-strip-kernel-odd-32-dimensions-lowers-charged/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [SESSION_NOTES_2026-10-01_to_10-03.md](../../../records/SESSION_NOTES_2026-10-01_to_10-03.md) · record
- [RESULT.md](../../../records/research/2026-10-01-kink-row-third-order-terms-account-6374-one-step/RESULT.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-exact-term-mixture-improves-torch-error-0-48-but) · [Copy provenance](../../../PROVENANCE.json)

