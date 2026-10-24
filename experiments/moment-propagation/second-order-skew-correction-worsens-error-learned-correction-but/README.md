# Second-order skew correction worsens error with the learned correction but improves it without

2026-10-02T06:26Z · experiment · rejected · measured

The card's per-unit mean formula stops at first order in k3 and k4; the missing second-order skew term was added at layers 1-15 on held-out rank 651. With the learned mean correction on the final-layer error is 0.8% worse, with it off 2.2% better: the term is real at the 2% level but the learned correction has already absorbed it, so it is not a no-refit item and entered only as a trainable strength. With true inputs the formula (Gaussian + k3 + k4) has no measurable error left at layers 14-15 and the term's ceiling is 0.3-0.7% per layer.

Recorded measurement:

    rank 651: learned on 16.19e-9 -> 16.32e-9 (+0.8%; layers 8-9 +3.5%, layer 3 -2.4%); learned off 17.06e-9 -> 16.69e-9 (-2.2%; layers 12-15 -1.4..-2.2%, layers 7-9 +2..+3%); open loop with true inputs: card formula 0.00-0.02e-9 at layers 14-15, 0.1-0.2e-9 at 8-13 (card's own error 8-14e-9); k3^2 term 0.02-0.10e-9 (0.047e-9 at layer 15 = 0.3%), k3 k4 cross term 0.069e-9

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_reexam_20261001_v1/e16b_k4_closed_loop.py](code/opus_reexam_20261001_v1/e16b_k4_closed_loop.py)
- [code/opus_reexam_20261001_v1/physicist/t16_mean_orders.py](code/opus_reexam_20261001_v1/physicist/t16_mean_orders.py)
- [code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/native_parameters.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/native_parameters.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/train_segment.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/train_segment.py)
- [code/fable_analyst_v1/s2card.py](code/fable_analyst_v1/s2card.py)
- [code/fable_analyst_v1/s6bcard.py](code/fable_analyst_v1/s6bcard.py)
- [code/fable_covhunter_v1/covstep.py](code/fable_covhunter_v1/covstep.py)

<details>
<summary>Additional programs and dependencies (28)</summary>

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
- [code/opus_reexam_20261001_v1/analyst/fresh/loader.py](code/opus_reexam_20261001_v1/analyst/fresh/loader.py)
- [code/opus_reexam_20261001_v1/k31c.py](code/opus_reexam_20261001_v1/k31c.py)
- [code/opus_reexam_20261001_v1/k31diag.py](code/opus_reexam_20261001_v1/k31diag.py)
- [code/opus_reexam_20261001_v1/k31l.py](code/opus_reexam_20261001_v1/k31l.py)
- [code/opus_reexam_20261001_v1/k31m.py](code/opus_reexam_20261001_v1/k31m.py)
- [code/opus_reexam_20261001_v1/ncar.py](code/opus_reexam_20261001_v1/ncar.py)
- [code/opus_reexam_20261001_v1/ncar3.py](code/opus_reexam_20261001_v1/ncar3.py)
- [code/opus_reexam_20261001_v1/ncar4.py](code/opus_reexam_20261001_v1/ncar4.py)
- [code/opus_reexam_20261001_v1/ncar5.py](code/opus_reexam_20261001_v1/ncar5.py)
- [code/opus_reexam_20261001_v1/ncar6.py](code/opus_reexam_20261001_v1/ncar6.py)
- [code/opus_reexam_20261001_v1/physicist/f4.py](code/opus_reexam_20261001_v1/physicist/f4.py)
- [code/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py](code/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py)
- [code/opus_reexam_20261001_v1/physicist/ncarry.py](code/opus_reexam_20261001_v1/physicist/ncarry.py)
- [code/opus_reexam_20261001_v1/physicist/pcard.py](code/opus_reexam_20261001_v1/physicist/pcard.py)
- [code/opus_reexam_20261001_v1/physicist/pcard_light.py](code/opus_reexam_20261001_v1/physicist/pcard_light.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L157](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L157)
- [TEAM_LOG_2026-10-01_to_10-22.md#L158](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L158)
- [TEAM_LOG_2026-10-01_to_10-22.md#L196](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L196)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L360](../../../records/SESSION_NOTES_2026-10-01_to_10-03.md#L360)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L438](../../../records/SESSION_NOTES_2026-10-01_to_10-03.md#L438)
- [RESULT.md#L1314](../../../records/research/2026-10-01-kink-row-third-order-terms-account-6374-one-step/RESULT.md#L1314)

<details>
<summary>All result files and records (4)</summary>

- [rank0651.log](rank0651.log) · log
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [SESSION_NOTES_2026-10-01_to_10-03.md](../../../records/SESSION_NOTES_2026-10-01_to_10-03.md) · record
- [RESULT.md](../../../records/research/2026-10-01-kink-row-third-order-terms-account-6374-one-step/RESULT.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-second-order-skew-correction-worsens-error-learned-correction-but) · [Copy provenance](../../../PROVENANCE.json)

