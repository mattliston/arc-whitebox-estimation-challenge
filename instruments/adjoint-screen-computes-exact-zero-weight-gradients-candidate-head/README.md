# Adjoint screen computes exact zero-weight gradients of candidate head-input columns

2026-10-10T05:06Z · tool · adopted · measured

t63_adjoint_dump.py runs the candidate generation 42 (S42) PyTorch moment-closure estimator in the trainer's gradient mode with a zero per-unit bias leaf on the learned head's first layer at every layer 1-15, so G_l = dL/d(bias) yields the exact zero-weight gradient of any candidate head-input column as sum_l x_l' G_l without a gradient run per candidate; certified on rank 17 against the trainer's own head gradient (5e-7) and enc0_w gradient (6e-7). t64_adjoint_screen.py ranked layer 1-14 candidates on the 36 gradient ranks by their partial gradient (the part not held by the head's 126 inputs + column 6): fold (fourth-chaos diagonal of pairs aged 3+), local quartic of old pairs (qold), kmu, kg, qk, qb, eold, e34 and s1 had signal; m2, skew- and kurtosis-corrected transport gate (gk), gb, k3d2, rg, the pushed head correction and mode loadings had none. The six new columns were handed over as b7cols.py.

Recorded measurement:

    certificate: HEADU gradient to 5e-7, enc0_w to 6e-7, final MSE 13.2936e-9 = trainer's; partial gradients (weights beyond 2.5 se of 128 / mean t^2 / first order at step 0.05): fold 102 / 27.7 / -0.455%; qold 90 / 17.9 / -0.33%; kmu 109 / 48.4 / -0.083%; kg 109 / 41.7 / -0.058%; qk 105 / 41.6 / -0.073%; qb 104 / 36.5 / -0.063%; eold 69 / 9.0 / -0.15%; e34 40 / 4.4 / -0.14%; s1 37 / 4.1 / -0.14%; no signal m2, gk, gb, k3d2, rg (mean t^2 0.9-2.7)

## Programs

- [code/opus_reexam_20261001_v1/physicist/t63_adjoint_dump.py](code/opus_reexam_20261001_v1/physicist/t63_adjoint_dump.py)
- [code/opus_reexam_20261001_v1/physicist/t64_adjoint_screen.py](code/opus_reexam_20261001_v1/physicist/t64_adjoint_screen.py)
- [code/opus_reexam_20261001_v1/physicist/t4_run_t63.sh](code/opus_reexam_20261001_v1/physicist/t4_run_t63.sh)
- [code/opus_reexam_20261001_v1/physicist/b7cols.py](code/opus_reexam_20261001_v1/physicist/b7cols.py)
- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py](code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py)
- [code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py)

<details>
<summary>Additional programs and dependencies (27)</summary>

- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py](code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
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
- [code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch.py](code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch.py)
- [code/opus_reexam_20261001_v1/physicist/b6cols.py](code/opus_reexam_20261001_v1/physicist/b6cols.py)
- [code/opus_reexam_20261001_v1/physicist/f4b.py](code/opus_reexam_20261001_v1/physicist/f4b.py)
- [code/opus_reexam_20261001_v1/trainer/otr_common.py](code/opus_reexam_20261001_v1/trainer/otr_common.py)
- [code/opus_reexam_20261001_v1/trainer/otr_engine.py](code/opus_reexam_20261001_v1/trainer/otr_engine.py)
- [code/opus_reexam_20261001_v1/trainer/otr_m.py](code/opus_reexam_20261001_v1/trainer/otr_m.py)
- [code/opus_reexam_20261001_v1/trainer/otr_pl.py](code/opus_reexam_20261001_v1/trainer/otr_pl.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L556](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L556)
- [TEAM_LOG_2026-10-01_to_10-22.md#L602](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L602)

<details>
<summary>All result files and records (3)</summary>

- [t64_screen_36_full.txt](t64_screen_36_full.txt) · result
- [t63_chain.log](t63_chain.log) · log
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-10-10-adjoint-screen-computes-exact-zero-weight-gradients-candidate-head) · [Copy provenance](../../PROVENANCE.json)

