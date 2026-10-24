# True norm-coupling diagonal used directly as kurtosis raises final MSE 12.6% with learned correction

2026-10-02T19:30Z · experiment · rejected · measured

The norm-coupling diagonal explains 27-37% of true unit-specific kurtosis, but inserting it directly into the mean formula destroys cancellation among variance, skew and kurtosis contributions. A learned input is therefore retained instead of a fixed physics correction.

Recorded measurement:

    true diagonal full strength on network35: +12.6% learned on, +9.1% off; half +1.0% / -0.1%; with (2,2) companion +14.2%, companion alone -1.25%; model diagonal full +6.1% / +0.05% on35/651, half +0.1% / -2.7%; ages3-4 /3-6 /all mean-only additions +1.5% /+2.0% /+1.8%; true-kurtosis explained 27-37%, coefficient 0.85-1.0; estimated/true diagonal correlation 0.42-0.48

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

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
- [code/fable_covhunter_v1/covstep_s2.py](code/fable_covhunter_v1/covstep_s2.py)
- [code/fable_covhunter_v1/gn3_knobs.py](code/fable_covhunter_v1/gn3_knobs.py)

<details>
<summary>Additional programs and dependencies (27)</summary>

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
- [code/opus_reexam_20261001_v1/physicist/e16b_f4b_frozen_s4g.py](code/opus_reexam_20261001_v1/physicist/e16b_f4b_frozen_s4g.py)
- [code/opus_reexam_20261001_v1/physicist/f4.py](code/opus_reexam_20261001_v1/physicist/f4.py)
- [code/opus_reexam_20261001_v1/physicist/f4b_frozen_fr24k.py](code/opus_reexam_20261001_v1/physicist/f4b_frozen_fr24k.py)
- [code/opus_reexam_20261001_v1/physicist/k31l_audit_frozen_s3m.py](code/opus_reexam_20261001_v1/physicist/k31l_audit_frozen_s3m.py)
- [code/opus_reexam_20261001_v1/physicist/ncar6_audit_frozen_s4g.py](code/opus_reexam_20261001_v1/physicist/ncar6_audit_frozen_s4g.py)
- [code/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py](code/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py)
- [code/opus_reexam_20261001_v1/physicist/ncar_fold_frozen_s2y.py](code/opus_reexam_20261001_v1/physicist/ncar_fold_frozen_s2y.py)
- [code/opus_reexam_20261001_v1/physicist/ncarry_frozen_s3a.py](code/opus_reexam_20261001_v1/physicist/ncarry_frozen_s3a.py)
- [code/opus_reexam_20261001_v1/physicist/ntor_frozen_s2z.py](code/opus_reexam_20261001_v1/physicist/ntor_frozen_s2z.py)
- [code/opus_reexam_20261001_v1/physicist/pbase_frozen_s3x.py](code/opus_reexam_20261001_v1/physicist/pbase_frozen_s3x.py)
- [code/opus_reexam_20261001_v1/physicist/pcard.py](code/opus_reexam_20261001_v1/physicist/pcard.py)
- [code/opus_reexam_20261001_v1/physicist/so2_frozen_s2b.py](code/opus_reexam_20261001_v1/physicist/so2_frozen_s2b.py)
- [code/opus_reexam_20261001_v1/physicist/t50b_k4_second_chaos.py](code/opus_reexam_20261001_v1/physicist/t50b_k4_second_chaos.py)
- [code/opus_reexam_20261001_v1/physicist/t51_error_cancellation.py](code/opus_reexam_20261001_v1/physicist/t51_error_cancellation.py)
- [code/opus_reexam_20261001_v1/physicist/t52_second_chaos_triplet.py](code/opus_reexam_20261001_v1/physicist/t52_second_chaos_triplet.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L441](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L441)

<details>
<summary>All result files and records (5)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [t4_ora67.log](t4_ora67.log) · log
- [t4_ora68.log](t4_ora68.log) · log
- [t4_ora69.log](t4_ora69.log) · log
- [t4_ora70.log](t4_ora70.log) · log

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-true-norm-coupling-diagonal-used-directly-kurtosis-raises-final) · [Copy provenance](../../../PROVENANCE.json)

