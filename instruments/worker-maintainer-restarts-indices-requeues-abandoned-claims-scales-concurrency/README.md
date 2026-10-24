# Worker maintainer restarts indices, requeues abandoned claims and scales concurrency on memory and swap

2026-10-12T07:42:08Z · tool · adopted · measured

maintain_workers_v1..v5.sh keeps a refit round's PyTorch workers alive: it relaunches any dead index (closing a stall where guard_footprint retired each worker at 4 h and only a same-index relaunch could re-serve its claim), requeues the claim of an index dead 10 minutes, raises the target from 6 to 8 (v1), 10 (v3) and 12 (v4) after 15 minutes of memory_pressure free >= 60% / 50% with swap within +64 MiB, steps back on swap growth or low free memory, stops the highest add-on worker when another lane's gate waits two polls in a row, and from v5 reads WORKER_CAP / launcher-flavor files set by the 8-vs-12 A/B. It also rotated the candidate generation 51 (S51) workers one at a time onto guard_footprint v2.

Recorded measurement:

    targets 6 -> 8 (free >= 50%, swap +64 MiB, 15 min), 8 -> 10 (free >= 60%), 10 -> 12 (free >= 60%, back at free < 52%); stepped 12 -> 10 at free 51% (15:46Z) and 10 -> 8 at 49% (15:54Z); workers 2.9-3.5 GB each; main-thread bound at ~0.8 core

## Programs

- [code/opus_reexam_20261001_v1/trainer/cert/RAD/maintain_workers_v1.sh](code/opus_reexam_20261001_v1/trainer/cert/RAD/maintain_workers_v1.sh)
- [code/opus_reexam_20261001_v1/trainer/cert/RAD/maintain_workers_v2.sh](code/opus_reexam_20261001_v1/trainer/cert/RAD/maintain_workers_v2.sh)
- [code/opus_reexam_20261001_v1/trainer/cert/RAD/maintain_workers_v3.sh](code/opus_reexam_20261001_v1/trainer/cert/RAD/maintain_workers_v3.sh)
- [code/opus_reexam_20261001_v1/trainer/cert/RAD/maintain_workers_v4.sh](code/opus_reexam_20261001_v1/trainer/cert/RAD/maintain_workers_v4.sh)
- [code/opus_reexam_20261001_v1/trainer/cert/RAD/maintain_workers_v5.sh](code/opus_reexam_20261001_v1/trainer/cert/RAD/maintain_workers_v5.sh)
- [code/opus_reexam_20261001_v1/trainer/cert/RAD/S51/rotate_g2_v1.sh](code/opus_reexam_20261001_v1/trainer/cert/RAD/S51/rotate_g2_v1.sh)
- [code/opus_reexam_20261001_v1/trainer/cert/RAD/S51/ab_count_v1.sh](code/opus_reexam_20261001_v1/trainer/cert/RAD/S51/ab_count_v1.sh)
- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py](code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py)
- [code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)

<details>
<summary>Additional programs and dependencies (40)</summary>

- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py)
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
- [code/opus_reexam_20261001_v1/analyst/fresh/loader.py](code/opus_reexam_20261001_v1/analyst/fresh/loader.py)
- [code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch.py](code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch.py)
- [code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch_v2.py](code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch_v2.py)
- [code/opus_reexam_20261001_v1/physicist/b6cols.py](code/opus_reexam_20261001_v1/physicist/b6cols.py)
- [code/opus_reexam_20261001_v1/physicist/b7cols.py](code/opus_reexam_20261001_v1/physicist/b7cols.py)
- [code/opus_reexam_20261001_v1/physicist/f4b.py](code/opus_reexam_20261001_v1/physicist/f4b.py)
- [code/opus_reexam_20261001_v1/trainer/cert/RAD/S51/ab_analyze_v1.py](code/opus_reexam_20261001_v1/trainer/cert/RAD/S51/ab_analyze_v1.py)
- [code/opus_reexam_20261001_v1/trainer/otr_common.py](code/opus_reexam_20261001_v1/trainer/otr_common.py)
- [code/opus_reexam_20261001_v1/trainer/otr_cut_runtime_v2.py](code/opus_reexam_20261001_v1/trainer/otr_cut_runtime_v2.py)
- [code/opus_reexam_20261001_v1/trainer/otr_engine.py](code/opus_reexam_20261001_v1/trainer/otr_engine.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_manifest_v1.py](code/opus_reexam_20261001_v1/trainer/otr_generated_manifest_v1.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_v1.py](code/opus_reexam_20261001_v1/trainer/otr_generated_v1.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_v2.py](code/opus_reexam_20261001_v1/trainer/otr_generated_v2.py)
- [code/opus_reexam_20261001_v1/trainer/otr_m.py](code/opus_reexam_20261001_v1/trainer/otr_m.py)
- [code/opus_reexam_20261001_v1/trainer/otr_m_v5.py](code/opus_reexam_20261001_v1/trainer/otr_m_v5.py)
- [code/opus_reexam_20261001_v1/trainer/otr_pl.py](code/opus_reexam_20261001_v1/trainer/otr_pl.py)
- [code/opus_reexam_20261001_v1/trainer/worker_generated_v12.py](code/opus_reexam_20261001_v1/trainer/worker_generated_v12.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1404](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1404)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1467](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1467)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1492](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1492)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1506](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1506)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1511](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1511)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1513](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1513)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-10-12-worker-maintainer-restarts-indices-requeues-abandoned-claims-scales-concurrency) · [Copy provenance](../../PROVENANCE.json)

