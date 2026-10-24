# Exact hub-coefficient shifts improve error by 3.36% against 3.54% for the cheaper approximation

2026-10-02T03:09Z · experiment · rejected · measured

F4b replaces the rule-A column shift by shifts with the exact hub coefficients (S1 and S2 at physical strength 1, columns at most two layers old). The four screen networks read -4.36% against F4's -2.33%, but on the 24 held-out newer ranks F4b gives -3.36% against F4's -3.54%, and with K4L -4.40% against -4.90%: the screen four misled by a factor two. A closed-form native fragment was drafted (45.8 flops per element, about +0.32% bill) and shelved.

Recorded measurement:

    screen four: -4.36% [-6.94,-2.86] 4 of 4 (S1 alone -2.33%, S2 alone -1.10%, S3+S4 bring 0.2% and early-layer damage); strength 0.5 S1..S4 -3.79%; 24 held-out: F4b -3.36% [-4.29,-2.42] 21 of 24, F4 -3.54% [-4.10,-2.95] 24 of 24, F4b vs F4 +0.19% [-0.84,+1.27]; F4b+K4L -4.40% vs F4+K4L -4.90%; first 8 fresh ranks: F4b -4.47% vs F4 0.6 -5.36%; native draft 45.8 flops per element (F4 16), about +0.32% bill

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_reexam_20261001_v1/physicist/f4b.py](code/opus_reexam_20261001_v1/physicist/f4b.py)
- [code/opus_reexam_20261001_v1/physicist/f4b_cert.py](code/opus_reexam_20261001_v1/physicist/f4b_cert.py)
- [code/opus_reexam_20261001_v1/physicist/t4_fr24.py](code/opus_reexam_20261001_v1/physicist/t4_fr24.py)
- [code/opus_reexam_20261001_v1/engineer/a18_f4b/fragment_f4b.py](code/opus_reexam_20261001_v1/engineer/a18_f4b/fragment_f4b.py)
- [code/opus_reexam_20261001_v1/engineer/a18_f4b/check_projection.py](code/opus_reexam_20261001_v1/engineer/a18_f4b/check_projection.py)
- [code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/native_parameters.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/native_parameters.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/train_segment.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/train_segment.py)

<details>
<summary>Additional programs and dependencies (18)</summary>

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
- [code/opus_reexam_20261001_v1/physicist/f4.py](code/opus_reexam_20261001_v1/physicist/f4.py)
- [code/opus_reexam_20261001_v1/physicist/pcard.py](code/opus_reexam_20261001_v1/physicist/pcard.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L73](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L73)
- [TEAM_LOG_2026-10-01_to_10-22.md#L87](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L87)
- [TEAM_LOG_2026-10-01_to_10-22.md#L88](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L88)
- [TEAM_LOG_2026-10-01_to_10-22.md#L95](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L95)
- [TEAM_LOG_2026-10-01_to_10-22.md#L97](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L97)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L375](../../../records/SESSION_NOTES_2026-10-01_to_10-03.md#L375)

<details>
<summary>All result files and records (3)</summary>

- [RESULT.md](../../../records/research/2026-10-01-kink-row-third-order-terms-account-6374-one-step/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [SESSION_NOTES_2026-10-01_to_10-03.md](../../../records/SESSION_NOTES_2026-10-01_to_10-03.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-exact-hub-coefficient-shifts-improve-error-3-36-against) · [Copy provenance](../../../PROVENANCE.json)

