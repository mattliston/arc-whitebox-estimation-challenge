# Native parameter export supports ten fitted dials with gradient-direction cosine 0.970–0.992

2026-10-02T06:41Z · tool · informational · measured

One command turns a trainer parameter file into a native candidate: the exporter writes the dials it owns into the source, a lever string is derived for the per-age column-shift and kurtosis strengths, the (2,2) scale and the covariance-output knobs, and the stack builder appends the current lever stack and runs the einsum gate. A rehearsal moving ten dials at once had native parity with the torch card (per-layer cosine 0.970-0.992, size ratio 1.01-1.04, leftover 4.96 / 5.77e-6 rms = two rounding-level builds); every dial candidate of S38-S42 was built this way.

Recorded measurement:

    rehearsal (F4 0.7 / 0.1, K4L 1.5 / 0.4, k22 1.004, bsrc 0.85, rc -0.28, iso -0.064, k21n -0.058, k31 -0.062): cosine 0.970-0.992 (network 35), 0.969-0.990 (100), size ratio 1.01-1.04, leftover 4.96 / 5.77e-6 rms; dial by dial within +-0.055 (35), +-0.11 (100) except iso +0.155 (se 0.044)

## Programs

- [code/opus_reexam_20261001_v1/engineer/stack/from_params.sh](code/opus_reexam_20261001_v1/engineer/stack/from_params.sh)
- [code/opus_reexam_20261001_v1/engineer/stack/levers_from_params.py](code/opus_reexam_20261001_v1/engineer/stack/levers_from_params.py)
- [code/opus_reexam_20261001_v1/engineer/stack/export_for_build.py](code/opus_reexam_20261001_v1/engineer/stack/export_for_build.py)
- [code/opus_reexam_20261001_v1/engineer/a20_runm/parity_regress.py](code/opus_reexam_20261001_v1/engineer/a20_runm/parity_regress.py)
- [code/opus_reexam_20261001_v1/engineer/a20_runm/torch_arms.py](code/opus_reexam_20261001_v1/engineer/a20_runm/torch_arms.py)
- [code/opus_reexam_20261001_v1/trainer/export.py](code/opus_reexam_20261001_v1/trainer/export.py)
- [code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/fable_analyst_v1/s2card.py](code/fable_analyst_v1/s2card.py)

<details>
<summary>Additional programs and dependencies (37)</summary>

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
- [code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch.py](code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch.py)
- [code/opus_reexam_20261001_v1/ncar.py](code/opus_reexam_20261001_v1/ncar.py)
- [code/opus_reexam_20261001_v1/ncar4.py](code/opus_reexam_20261001_v1/ncar4.py)
- [code/opus_reexam_20261001_v1/ncar5.py](code/opus_reexam_20261001_v1/ncar5.py)
- [code/opus_reexam_20261001_v1/physicist/b6cols.py](code/opus_reexam_20261001_v1/physicist/b6cols.py)
- [code/opus_reexam_20261001_v1/physicist/b7cols.py](code/opus_reexam_20261001_v1/physicist/b7cols.py)
- [code/opus_reexam_20261001_v1/physicist/f4b.py](code/opus_reexam_20261001_v1/physicist/f4b.py)
- [code/opus_reexam_20261001_v1/physicist/ncar6_audit_frozen_s3d.py](code/opus_reexam_20261001_v1/physicist/ncar6_audit_frozen_s3d.py)
- [code/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py](code/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py)
- [code/opus_reexam_20261001_v1/physicist/ncar_fold_frozen_s2w.py](code/opus_reexam_20261001_v1/physicist/ncar_fold_frozen_s2w.py)
- [code/opus_reexam_20261001_v1/physicist/ncarry.py](code/opus_reexam_20261001_v1/physicist/ncarry.py)
- [code/opus_reexam_20261001_v1/physicist/pbase_frozen_s4a.py](code/opus_reexam_20261001_v1/physicist/pbase_frozen_s4a.py)
- [code/opus_reexam_20261001_v1/trainer/frozen/k31l_t1508.py](code/opus_reexam_20261001_v1/trainer/frozen/k31l_t1508.py)
- [code/opus_reexam_20261001_v1/trainer/frozen/k31l_t1508y.py](code/opus_reexam_20261001_v1/trainer/frozen/k31l_t1508y.py)
- [code/opus_reexam_20261001_v1/trainer/frozen/ncar6_t1339.py](code/opus_reexam_20261001_v1/trainer/frozen/ncar6_t1339.py)
- [code/opus_reexam_20261001_v1/trainer/frozen/ncar6_t1402.py](code/opus_reexam_20261001_v1/trainer/frozen/ncar6_t1402.py)
- [code/opus_reexam_20261001_v1/trainer/frozen/ncar6_t1402x.py](code/opus_reexam_20261001_v1/trainer/frozen/ncar6_t1402x.py)
- [code/opus_reexam_20261001_v1/trainer/frozen/ncar6_t1402y.py](code/opus_reexam_20261001_v1/trainer/frozen/ncar6_t1402y.py)
- [code/opus_reexam_20261001_v1/trainer/otr_common.py](code/opus_reexam_20261001_v1/trainer/otr_common.py)
- [code/opus_reexam_20261001_v1/trainer/otr_engine.py](code/opus_reexam_20261001_v1/trainer/otr_engine.py)
- [code/opus_reexam_20261001_v1/trainer/otr_m.py](code/opus_reexam_20261001_v1/trainer/otr_m.py)
- [code/opus_reexam_20261001_v1/trainer/otr_pl.py](code/opus_reexam_20261001_v1/trainer/otr_pl.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L102](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L102)
- [TEAM_LOG_2026-10-01_to_10-22.md#L139](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L139)
- [TEAM_LOG_2026-10-01_to_10-22.md#L150](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L150)
- [TEAM_LOG_2026-10-01_to_10-22.md#L161](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L161)
- [TEAM_LOG_2026-10-01_to_10-22.md#L166](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L166)
- [RESULT.md#L608](../../experiments/arithmetic-kernels/half-zero-strip-kernel-odd-32-dimensions-lowers-charged/RESULT.md#L608)

<details>
<summary>All result files and records (2)</summary>

- [RESULT.md](../../experiments/arithmetic-kernels/half-zero-strip-kernel-odd-32-dimensions-lowers-charged/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-10-02-native-parameter-export-supports-ten-fitted-dials-gradient-direction) · [Copy provenance](../../PROVENANCE.json)

