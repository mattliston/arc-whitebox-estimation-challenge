# Removing the covariance-square form needs 16-32 subspace columns: 1.3% more bill for 1.2 points of MSE

2026-10-02T14:14Z · experiment · rejected · measured

On networks 287 and 641 the exact four-form removal beat the multiple of C alone by several points; the difference is the C^2 form, not the mean forms or the births. Regressing the diagonal of the carried matrix on (tr C v_i, 1, mu_i^2, 2 mu_i (C mu)_i, (C^2)_ii) gives all four coefficients to the digit of the exact fit, but the matrix C^2 itself must be formed: from 16 warm-started subspace columns it works on 287 / 641 / 651 but loses on network 35 (-3.8% against -6.2% with no C^2 removal; 32 columns needed, -5.6%), about 1.3% of bill for 1.2 points. Closed for now; a stand-in with 4-8 Krylov vectors was left to test.

Recorded measurement:

    287 / 641 / 651: exact four forms -5.7 / -4.2 / -13.8%, multiple of C -0.6 / -0.7 / -12.6%, table -1.9 / -0.0 / -12.4%; diagonal regression with (C^2)_ii -5.7 / -4.3 / -13.9%; without C^2 regressor -1.8 / -0.9 / -13.0%; C^2 from top 16 / 8 / 4 eigenpairs -5.4 / -4.0 / -13.6%, -4.6 / -2.0 / -13.3%, -3.5 / -0.9 / -13.1%; native subspace K 16: -5.6 / -3.5 / -12.4%, K 8 -4.9 / -0.8 / -12.2%, no sweep +0.7 / +1.4 / -10.0%; network 35 K 16 -3.8% vs -6.2%, K 32 -5.6%; seven networks four exact forms + K31L4 -8.81% vs -7.60%; Rayleigh quotients on 287 K 4 / 8 / 16 / 32: -3.2 / -4.5 / -5.2 / -5.5%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_reexam_20261001_v1/e53_forms.sh](code/opus_reexam_20261001_v1/e53_forms.sh)
- [code/opus_reexam_20261001_v1/e54_center.sh](code/opus_reexam_20261001_v1/e54_center.sh)
- [code/opus_reexam_20261001_v1/e55_ptop.sh](code/opus_reexam_20261001_v1/e55_ptop.sh)
- [code/opus_reexam_20261001_v1/e56_bulk.sh](code/opus_reexam_20261001_v1/e56_bulk.sh)
- [code/opus_reexam_20261001_v1/e56_spectrum.py](code/opus_reexam_20261001_v1/e56_spectrum.py)
- [code/opus_reexam_20261001_v1/e57_pdiag.sh](code/opus_reexam_20261001_v1/e57_pdiag.sh)
- [code/opus_reexam_20261001_v1/e58_pmu.sh](code/opus_reexam_20261001_v1/e58_pmu.sh)
- [code/opus_reexam_20261001_v1/e59_const.sh](code/opus_reexam_20261001_v1/e59_const.sh)
- [code/opus_reexam_20261001_v1/e60_c2sub.sh](code/opus_reexam_20261001_v1/e60_c2sub.sh)
- [code/opus_reexam_20261001_v1/e63_ceiling.sh](code/opus_reexam_20261001_v1/e63_ceiling.sh)
- [code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)

<details>
<summary>Additional programs and dependencies (36)</summary>

- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
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

- [TEAM_LOG_2026-10-01_to_10-22.md#L333](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L333)
- [TEAM_LOG_2026-10-01_to_10-22.md#L343](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L343)
- [TEAM_LOG_2026-10-01_to_10-22.md#L357](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L357)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L559](../../../records/SESSION_NOTES_2026-10-01_to_10-03.md#L559)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L584](../../../records/SESSION_NOTES_2026-10-01_to_10-03.md#L584)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [SESSION_NOTES_2026-10-01_to_10-03.md](../../../records/SESSION_NOTES_2026-10-01_to_10-03.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-removing-covariance-square-form-needs-16-32-subspace-columns) · [Copy provenance](../../../PROVENANCE.json)

