# Depth-six squares and padded pushes save work but exceed the projected wall budget

2026-09-23T06:19Z · experiment · rejected · measured

Deeper multiplication increased server compute enough to outweigh the request reduction. The candidate was withheld after platform-calibrated wall estimates exceeded the limit.

Recorded measurement:

    - 2026-09-22 23:19 [engineer] HEADS-UP: quiet_census.sh 35,100 on S3 (S2+HE), S3+sq6, S3+sq6+push64 now (~9 min pause of other heavy jobs). Bank-alone native: HE+sq6 (depth-6 squares, rounding-level) util r35 .18449 (-2.01%) / r100 .18057 (-2.18%), MSE +0.08%/+0.12%; HE+sq6+push64 util -2.61%/-3.71%, MSE +0.35%/-0.04%; local cost of a 1024^3 D6 square +110 ms (x72 squares = +8 s/net).
    - 2026-09-22 23:29 [engineer] PAIRED QUIET census (bank / S3 / S3+sq6 / S3+sq6+push64; r35 cold, r100): round trips 95.4k/84.1k, 53.8k/46.0k, 55.6k/47.6k, 56.2k/49.2k; server CPU s 36.4/34.0, 31.9/29.3, 38.0/35.5, 39.3/38.7; RSS 6.31/6.00/5.82/6.20 GB. Native vs S2: S3 bit-identical, util -0.88%/-0.93%; S3+sq6 util -2.05%/-2.23%, MSE +0.07%/+0.04% (adj -1.98%/-2.19%); S3+sq6+p64 util -2.68%/-4.22%, MSE +0.29%/-0.01% (adj -2.39%/-4.23%). Platform wall est vs bank: -36..-59 / -23..-45 / -15..-41 s. v9 frames not cheapenable (steady frame bill 1.1%/net, 69% read path; sliced views 1.1-3.7x slower). Files fable_engineer_v1/candS3_{S2_HE,sq6,sq6_p64}.py; lead + porter told.
    - 2026-09-22 23:58 lead: OVERNIGHT JOB (lead-owned, nice 20, 5 threads, ~7 GB): fable_analyst_v1/refit_s2.py --tag refit_s2_full_lr2e4 --groups full --ntrain 160 --nvalid 16 --epochs 3 --lr 2e-4 (all closure groups except the GNN body, on S2 physics). Question: is the closure under-trained? Heads-only refits gave <= 0.5%; joint refits were never run to completion. Log: fable_analyst_v1/refit_s2_full_lr2e4.log. Do not kill it without asking me. sq6 (engineer 23:31 quiet pairs): util -1.8%/-3.3% but server CPU +30-40% and wall 47-51 s vs bank 43.6/39.0 s -> not fire-safe as is. S3 validated (porter). covhunter knobs: VALID40 0.9834 vs row factor, native form = closed-form delta on the pair map's covariance output.
    - 2026-09-23 00:15 lead: PLATFORM WALL CALIBRATED from the public per-MLP telemetry on each submission page (new tool fable_fire_v1/read_eval_page.py <sid> -> eval_<sid>.json). Predict wall = server kernel + server overhead + residual. #331824 bank: mean 100.0 s, worst 110.2 s; kernel 87.5 s; overhead 12.2 s at ~58k requests. #331947 S2: mean 95.7 s, worst 105.6 s; kernel 85.5 s; overhead 9.9 s at ~45k requests. Overhead is ~0.22 ms per request, and platform kernel time is ~2.4x our local quiet single-thread server CPU. So the platform is COMPUTE-bound; request diets buy little wall. RULE: delta platform wall ~ 2.5 x delta(local quiet paired server CPU) + 0.22 ms x delta(requests), and keep the worst MLP <= ~112 s. sq6_p64 (+11 s local CPU vs S3) -> ~128 s = TIMEOUT. S3-HE (-1.7 s local vs S2) -> worst ~101 s.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_fire_v1/quiet_census.sh](code/fable_fire_v1/quiet_census.sh)
- [code/fable_analyst_v1/refit_s2.py](code/fable_analyst_v1/refit_s2.py)
- [code/fable_fire_v1/read_eval_page.py](code/fable_fire_v1/read_eval_page.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py)
- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)
- [code/astra_phase2_reexamination_v1/unused_materializations_v1/estimator.py](code/astra_phase2_reexamination_v1/unused_materializations_v1/estimator.py)

<details>
<summary>Additional programs and dependencies (13)</summary>

- [code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py](code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py)
- [code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py](code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py)
- [code/fable_analyst_v1/s2card.py](code/fable_analyst_v1/s2card.py)
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_bfix.py](code/fable_refit_v1/model_bank_bfix.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L84](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L84)
- [TEAM_LOG_2026-09-22_to_09-26.md#L87](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L87)
- [TEAM_LOG_2026-09-22_to_09-26.md#L97](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L97)
- [TEAM_LOG_2026-09-22_to_09-26.md#L98](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L98)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [refit_s2_full_lr2e4.log](refit_s2_full_lr2e4.log) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-depth-six-squares-padded-pushes-save-work-but-exceed) · [Copy provenance](../../../PROVENANCE.json)

