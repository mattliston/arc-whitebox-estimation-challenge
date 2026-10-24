# Capacity redistribution rejected: early birth reduction adds 18.0% MSE; late capacity costs exceed gains

2026-10-10T12:51Z · experiment · rejected · measured

Moving capacity out of layers 1-5 is catastrophic and adding births or response rows at 8-12 buys nothing; the old-tier rank at the L10 join trades 1:1 with metered FLOPs; more early births gain less than their metered FLOPs. The moment-closure estimator sits at the knee: cutting is steep, adding is flat.

Recorded measurement:

    birth floor 0.05 at L1-5 +18.0% (0 of 6) for -3.3% bill; births at 8-11 -0.05% for +3.4%; L10 rank 320 -> 384 -1.26% (11 of 12) for +1.675% native bill; floor 0.005 at L1-5 -0.43% for +0.66% proxy

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_reexam_20261001_v1/physicist/t76_realloc.py](code/opus_reexam_20261001_v1/physicist/t76_realloc.py)
- [code/opus_reexam_20261001_v1/physicist/t77_realloc_table.py](code/opus_reexam_20261001_v1/physicist/t77_realloc_table.py)
- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py](code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py)
- [code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)

<details>
<summary>Additional programs and dependencies (27)</summary>

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
- [code/opus_reexam_20261001_v1/idle_gate.sh](code/opus_reexam_20261001_v1/idle_gate.sh)
- [code/opus_reexam_20261001_v1/physicist/b6cols.py](code/opus_reexam_20261001_v1/physicist/b6cols.py)
- [code/opus_reexam_20261001_v1/physicist/b7cols.py](code/opus_reexam_20261001_v1/physicist/b7cols.py)
- [code/opus_reexam_20261001_v1/physicist/f4b.py](code/opus_reexam_20261001_v1/physicist/f4b.py)
- [code/opus_reexam_20261001_v1/trainer/otr_common.py](code/opus_reexam_20261001_v1/trainer/otr_common.py)
- [code/opus_reexam_20261001_v1/trainer/otr_engine.py](code/opus_reexam_20261001_v1/trainer/otr_engine.py)
- [code/opus_reexam_20261001_v1/trainer/otr_m.py](code/opus_reexam_20261001_v1/trainer/otr_m.py)
- [code/opus_reexam_20261001_v1/trainer/otr_pl.py](code/opus_reexam_20261001_v1/trainer/otr_pl.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L612](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L612)
- [TEAM_LOG_2026-10-01_to_10-22.md#L615](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L615)
- [TEAM_LOG_2026-10-01_to_10-22.md#L616](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L616)
- [TEAM_LOG_2026-10-01_to_10-22.md#L617](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L617)
- [TEAM_LOG_2026-10-01_to_10-22.md#L618](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L618)
- [TEAM_LOG_2026-10-01_to_10-22.md#L619](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L619)
- [TEAM_LOG_2026-10-01_to_10-22.md#L621](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L621)
- [TEAM_LOG_2026-10-01_to_10-22.md#L622](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L622)
- [TEAM_LOG_2026-10-01_to_10-22.md#L623](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L623)
- [TEAM_LOG_2026-10-01_to_10-22.md#L628](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L628)
- [TEAM_LOG_2026-10-01_to_10-22.md#L629](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L629)

<details>
<summary>All result files and records (2)</summary>

- [RESULT.md](../../../records/research/2026-10-01-kink-row-third-order-terms-account-6374-one-step/RESULT.md) · explainer
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- enables: [Fixed-rank join-sketch refinements rejected: exact eigenvectors give -0.80% MSE for about 9% more FLOPs](../../join-sketch/fixed-rank-join-sketch-refinements-rejected-exact-eigenvectors-give/README.md)
- enables: [Birth floors and final-layer row cuts rejected: added accuracy is outweighed by FLOPs or error](../../birth-terms/birth-floors-final-layer-row-cuts-rejected-added-accuracy/README.md)

[Research graph](../../../graph/index.html#node=2026-10-10-capacity-redistribution-rejected-early-birth-reduction-adds-18-0) · [Copy provenance](../../../PROVENANCE.json)

