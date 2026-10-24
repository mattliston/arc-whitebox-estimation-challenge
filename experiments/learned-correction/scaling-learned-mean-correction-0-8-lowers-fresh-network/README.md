# Scaling the learned mean correction to 0.8 lowers fresh-network error by 0.84%

2026-10-02T05:32Z · experiment · adopted · measured

On fresh ranks of the S37 card the learned mean correction multiplied by 0.9 gives -0.48%, by 0.8 -0.84% and by 0.7 -0.60%; its regression slope on the residual it should remove is 0.85-0.98 at layers 5-14 (0.82 at layer 15) with cosine 0.17-0.32, so it is calibrated to within 10-15% but explains only 3-10% of the per-layer residual. The scale entered the dial-only candidates (0.8 in S38, 0.784 in S39) through the head literals; a native lever multiplies the per-layer gain constant at no cost.

Recorded measurement:

    x0.9 -0.48% (se 0.17, 7 of 8); x0.8 -0.84% [95% -1.50,-0.16] (8 of 10); x0.7 -0.60% [-1.63,+0.40]; slope on the residual 0.85-0.98 (layers 5-14), 0.82 (layer 15), cosine 0.17-0.32; on rank 651 the whole correction is worth 5.4% (13.5% on the old validation nets)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_reexam_20261001_v1/e24_fresh24_shrink.sh](code/opus_reexam_20261001_v1/e24_fresh24_shrink.sh)
- [code/opus_reexam_20261001_v1/e25_delta_slope.py](code/opus_reexam_20261001_v1/e25_delta_slope.py)
- [code/opus_reexam_20261001_v1/engineer/a23_ds/unit_ds.py](code/opus_reexam_20261001_v1/engineer/a23_ds/unit_ds.py)
- [code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/native_parameters.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/native_parameters.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/train_segment.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/train_segment.py)
- [code/fable_analyst_v1/s2card.py](code/fable_analyst_v1/s2card.py)
- [code/fable_analyst_v1/s6bcard.py](code/fable_analyst_v1/s6bcard.py)

<details>
<summary>Additional programs and dependencies (29)</summary>

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
- [code/opus_reexam_20261001_v1/analyst/fresh/loader.py](code/opus_reexam_20261001_v1/analyst/fresh/loader.py)
- [code/opus_reexam_20261001_v1/e16b_k4_closed_loop.py](code/opus_reexam_20261001_v1/e16b_k4_closed_loop.py)
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

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L137](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L137)
- [TEAM_LOG_2026-10-01_to_10-22.md#L141](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L141)
- [TEAM_LOG_2026-10-01_to_10-22.md#L148](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L148)
- [TEAM_LOG_2026-10-01_to_10-22.md#L158](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L158)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L402](../../../records/SESSION_NOTES_2026-10-01_to_10-03.md#L402)

<details>
<summary>All result files and records (3)</summary>

- [e25_fresh6.log](e25_fresh6.log) · log
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [SESSION_NOTES_2026-10-01_to_10-03.md](../../../records/SESSION_NOTES_2026-10-01_to_10-03.md) · record

</details>

## Related work

- shipped in: [Submission #333481: fitted dials, masked covariance correction and three-request operand loads](../../../submissions/333481/README.md)

[Research graph](../../../graph/index.html#node=2026-10-02-scaling-learned-mean-correction-0-8-lowers-fresh-network) · [Copy provenance](../../../PROVENANCE.json)

