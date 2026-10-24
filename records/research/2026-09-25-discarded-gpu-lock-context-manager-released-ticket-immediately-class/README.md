# Discarded GPU-lock context manager released its ticket immediately; class lifetime and with-block fix verified

2026-09-25T07:38Z · incident · failed · measured

The generator-based lock was garbage-collected immediately after entry, so two jobs ran without protection. The revised lock and driver preserve ticket lifetime and pass FIFO and process-exit tests.

Recorded measurement:

    - 2026-09-25 00:40 [physicist] MPS LOCK BUG (reported to engineer, covhunter, lead): fable_engineer_v1/refit_mps_v3.py calls mps_lock.hold(...).__enter__() and discards the context manager; CPython garbage-collects the generator at once and its finally deletes the ticket right after "MPS queue HOLD" prints (verified in a separate test queue dir: 0 tickets with the discarded pattern, 1 with a kept reference). So refit_mps_v3 runs (and copies: my refit_mps_v3j.py, covhunter's refit_mps_v3_ch.py) never hold MPS. Now: my loopR3 (pid 75589, since 00:24:43, no ticket) and sym_restart (pid 76130, took the lock 00:25:32) share MPS. My pause+manual-ticket fix was blocked by the permission system; nothing changed. loopR3 continues unless the lead says kill (ETA ~00:45-00:50). Fixed driver for future runs: refit6b/refit_mps_v3j2.py (reference kept). Covhunter's capacity chain waits for 'loopR3' in the lock status, which will never appear; done signal = refit6b/s6b_loopR3_joint_s23/dials_epoch0.json.
    - 2026-09-25 00:38 lead: MPS LOCK BUG (found by the physicist): refit_mps_v3.py calls mps_lock.hold(...).__enter__() and discards the object, so the generator's finally deletes the ticket at once and the lock is NEVER held. Copies: covhunter's refit_mps_v3_ch.py and the physicist's refit_mps_v3j.py. Two MPS jobs are running now: loopR3 (pid 75589, ETA ~00:45-00:50) and sym_restart_mc.py (pid 76130); memory fine (26 GB free). Both are allowed to finish. NO new MPS launches until the engineer ships refit_mps_v4.py (a with-block plus a ticket-persistence self-test). Covhunter's capacity test is on HOLD until rebased.
    2026-09-25 00:57 covhunter: MPS LOCK SELF-TEST (PASS) of fable_covhunter_v1/refit_mps_v3_ch.py (ticket kept by reference): scratch queue /private/tmp/claude-501/-Users-matt-arc-white-box-estimation/f2e42c0d-29ca-4077-b5c8-c7b876aea2ee/scratchpad/mpsq_selftest; ticket files at HOLD: 1, at +30 s: 1, process alive at +30 s: yes; test killed.
    - 2026-09-25 01:29 [engineer] MPS LOCK FIXED (physicist's find): mps_lock v2 (hold() is a class; an entered hold is kept alive until __exit__ or process exit, so even the discarded one-liner mps_lock.hold(...).__enter__() now holds for the job's lifetime; v1 kept as mps_lock_v1_gc_bug.py.bak) + refit_mps_v4.py (the lock in a with-block around the whole run; body = refit_mps_body_v4.py, v3 minus its lock line). mps_lock_selftest.py ALL PASS: v2 one-liner held + released at exit; v1 one-liner reproduces the bug; refit_mps_v4 self-test passes and is visible from outside; FIFO order. mps_card's docstring recipe fixed (with-block). Physicist, covhunter, analyst told. The running loopR3/sym_restart loaded v1 and hold no ticket; new launches are covered.
    - 2026-09-25 01:30 lead: MPS LOCK FIXED (engineer).
      - mps_lock v2 edited in place, with v1 archived as mps_lock_v1_gc_bug.py.bak: hold() is a class whose entered holds stay in a module-level list until exit, so even the buggy one-liner now holds; every copy importing it is fixed on its next launch.
      - refit_mps_v4.py wraps the run in a with-block.
      - Self-test PASS: ticket lifetime, v1 bug reproduced, FIFO.
      The two ticketless MPS jobs have exited; MPS launches are safe again.

## Programs

- [../../../instruments/gpu-mixed-precision-refit-certificate-466-seconds-versus-1454/code/fable_engineer_v1/refit_mps_v3.py](../../../instruments/gpu-mixed-precision-refit-certificate-466-seconds-versus-1454/code/fable_engineer_v1/refit_mps_v3.py)
- [../../../experiments/learned-correction/joint-learned-parameter-physics-dial-gpu-refit-passes-cpu/code/fable_physicist_v1/refit6b/refit_mps_v3j.py](../../../experiments/learned-correction/joint-learned-parameter-physics-dial-gpu-refit-passes-cpu/code/fable_physicist_v1/refit6b/refit_mps_v3j.py)
- [refit_mps_v3_ch.py](refit_mps_v3_ch.py)
- [../../../experiments/learned-correction/joint-learned-parameter-physics-dial-gpu-refit-passes-cpu/code/fable_physicist_v1/refit6b/refit_mps_v3j2.py](../../../experiments/learned-correction/joint-learned-parameter-physics-dial-gpu-refit-passes-cpu/code/fable_physicist_v1/refit6b/refit_mps_v3j2.py)
- [sym_restart_mc.py](sym_restart_mc.py)
- [refit_mps_v4.py](refit_mps_v4.py)
- [refit_mps_body_v4.py](refit_mps_body_v4.py)
- [mps_lock_selftest.py](mps_lock_selftest.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/build.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/build.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/build_v2.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/build_v2.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/core.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/core.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/core_v2.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/core_v2.py)

<details>
<summary>Additional programs and dependencies (35)</summary>

- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/reference_numpy.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/reference_numpy.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/native.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/native.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/native_v2.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/native_v2.py)
- [../../../experiments/third-order-planes/aligning-pair-precision-covariance-arithmetic-still-fails-fidelity-maximum/code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](../../../experiments/third-order-planes/aligning-pair-precision-covariance-arithmetic-still-fails-fidelity-maximum/code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/unused_materializations_v1/estimator.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/unused_materializations_v1/estimator.py)
- [../../../experiments/learned-correction/covariance-edge-refit-passes-native40-but-fails-independent-public8/code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py](../../../experiments/learned-correction/covariance-edge-refit-passes-native40-but-fails-independent-public8/code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py)
- [../../../submissions/331824/estimator.py](../../../submissions/331824/estimator.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py)
- [../../../instruments/composite-estimator-builder-verifies-gain-correction-plane-scale-join/code/fable_analyst_v1/s2card.py](../../../instruments/composite-estimator-builder-verifies-gain-correction-plane-scale-join/code/fable_analyst_v1/s2card.py)
- [../2026-09-23-training-normalization-constants-caused-70-loss-spike-network-weight/s6bcard.py](../2026-09-23-training-normalization-constants-caused-70-loss-spike-network-weight/s6bcard.py)
- [../2026-09-23-partial-truth-plane-correction-outperforms-full-replacement-kink-row/covstep.py](../2026-09-23-partial-truth-plane-correction-outperforms-full-replacement-kink-row/covstep.py)
- [../../../experiments/dials/extra-kink-row-carrier-scales-add-no-useful-gain/code/fable_covhunter_v1/covstep_s2.py](../../../experiments/dials/extra-kink-row-carrier-scales-add-no-useful-gain/code/fable_covhunter_v1/covstep_s2.py)
- [../../../experiments/late-layer-exactness/local-final-layer-error-prediction-null-held-out-r/code/fable_covhunter_v1/gn3_knobs.py](../../../experiments/late-layer-exactness/local-final-layer-error-prediction-null-held-out-r/code/fable_covhunter_v1/gn3_knobs.py)
- [../../../experiments/dials/alternating-learned-dial-fit-refit-improves-held-out-error/code/fable_covhunter_v1/s6bcard_jfitR1.py](../../../experiments/dials/alternating-learned-dial-fit-refit-improves-held-out-error/code/fable_covhunter_v1/s6bcard_jfitR1.py)
- [../../../experiments/dials/consistent-semantics-physics-refit-improves-raw-error-1-98/code/fable_covhunter_v1/s6bcard_joint2.py](../../../experiments/dials/consistent-semantics-physics-refit-improves-raw-error-1-98/code/fable_covhunter_v1/s6bcard_joint2.py)
- [../../../experiments/dials/third-dial-fit-round-nystrom-structure-null-sixteen-network/code/fable_covhunter_v1/s6bcard_joint2t.py](../../../experiments/dials/third-dial-fit-round-nystrom-structure-null-sixteen-network/code/fable_covhunter_v1/s6bcard_joint2t.py)
- [../../../experiments/dials/consistent-semantics-physics-refit-improves-raw-error-1-98/code/fable_engineer_v1/gn_linear_hook.py](../../../experiments/dials/consistent-semantics-physics-refit-improves-raw-error-1-98/code/fable_engineer_v1/gn_linear_hook.py)
- [../../../experiments/dials/consistent-semantics-physics-refit-improves-raw-error-1-98/code/fable_engineer_v1/join_nystrom_torch.py](../../../experiments/dials/consistent-semantics-physics-refit-improves-raw-error-1-98/code/fable_engineer_v1/join_nystrom_torch.py)
- [../../../instruments/native-matching-low-rank-residual-plane-training-avoids-differentiating/code/fable_engineer_v1/lrstar_torch_hook.py](../../../instruments/native-matching-low-rank-residual-plane-training-avoids-differentiating/code/fable_engineer_v1/lrstar_torch_hook.py)
- [../../../instruments/gpu-mixed-precision-refit-certificate-466-seconds-versus-1454/code/fable_engineer_v1/mps_card.py](../../../instruments/gpu-mixed-precision-refit-certificate-466-seconds-versus-1454/code/fable_engineer_v1/mps_card.py)
- [../../../instruments/gpu-mixed-precision-refit-certificate-466-seconds-versus-1454/code/fable_engineer_v1/mps_lock.py](../../../instruments/gpu-mixed-precision-refit-certificate-466-seconds-versus-1454/code/fable_engineer_v1/mps_lock.py)
- [../../../instruments/termwise-hermite-pair-series-implementation-reduces-training-intermediates-small/code/pair_series_torch.py](../../../instruments/termwise-hermite-pair-series-implementation-reduces-training-intermediates-small/code/pair_series_torch.py)
- [../../../experiments/dials/consistent-semantics-physics-refit-improves-raw-error-1-98/code/fable_physicist_v1/refit6b/lrstar_s1zero_hook.py](../../../experiments/dials/consistent-semantics-physics-refit-improves-raw-error-1-98/code/fable_physicist_v1/refit6b/lrstar_s1zero_hook.py)
- [../../../instruments/gpu-mixed-precision-refit-certificate-466-seconds-versus-1454/code/fable_physicist_v1/refit6b/s6bcard_loop3.py](../../../instruments/gpu-mixed-precision-refit-certificate-466-seconds-versus-1454/code/fable_physicist_v1/refit6b/s6bcard_loop3.py)
- [../../../experiments/learned-correction/joint-learned-parameter-physics-dial-gpu-refit-passes-cpu/code/fable_physicist_v1/refit6b/s6bcard_loop4j.py](../../../experiments/learned-correction/joint-learned-parameter-physics-dial-gpu-refit-passes-cpu/code/fable_physicist_v1/refit6b/s6bcard_loop4j.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/fable_refit_v1/common.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/fable_refit_v1/common.py)
- [../2026-09-23-rank-one-covariancemean-gain-mixture-explains-6368-2-1/model_bank_bfix.py](../2026-09-23-rank-one-covariancemean-gain-mixture-explains-6368-2-1/model_bank_bfix.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/fable_refit_v1/model_bank_research.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/fable_refit_v1/model_bank_research.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L828](../../TEAM_LOG_2026-09-22_to_09-26.md#L828)
- [TEAM_LOG_2026-09-22_to_09-26.md#L829](../../TEAM_LOG_2026-09-22_to_09-26.md#L829)
- [TEAM_LOG_2026-09-22_to_09-26.md#L837](../../TEAM_LOG_2026-09-22_to_09-26.md#L837)
- [TEAM_LOG_2026-09-22_to_09-26.md#L869](../../TEAM_LOG_2026-09-22_to_09-26.md#L869)
- [TEAM_LOG_2026-09-22_to_09-26.md#L870](../../TEAM_LOG_2026-09-22_to_09-26.md#L870)
- [TEAM_LOG_2026-09-22_to_09-26.md#L871](../../TEAM_LOG_2026-09-22_to_09-26.md#L871)
- [TEAM_LOG_2026-09-22_to_09-26.md#L872](../../TEAM_LOG_2026-09-22_to_09-26.md#L872)
- [TEAM_LOG_2026-09-22_to_09-26.md#L873](../../TEAM_LOG_2026-09-22_to_09-26.md#L873)
- [TEAM_LOG_2026-09-22_to_09-26.md#L874](../../TEAM_LOG_2026-09-22_to_09-26.md#L874)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log
- [dials_epoch0.json](../../../experiments/learned-correction/joint-learned-parameter-physics-dial-gpu-refit-passes-cpu/dials_epoch0.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-discarded-gpu-lock-context-manager-released-ticket-immediately-class) · [Copy provenance](../../../PROVENANCE.json)

