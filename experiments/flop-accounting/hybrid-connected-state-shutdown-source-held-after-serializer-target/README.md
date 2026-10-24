# Hybrid connected-state shutdown source held after serializer and target-custody defects

2026-08-24 · experiment · rejected · measured

Standalone dense-fp32 CS3 preregistration SHA-256 3ddf814fa95737303fa0184d963f9ff734e6ff250d76ef5ab9dbd1db8197275e was sealed HOLD before source under author receipt SHA-256 9ab43dd3e385f0bf043e23668ef0eb5563f32f4c5e913fcdd4e6a5e64cb5ca70 and independent receipt SHA-256 060cfbab98c85f7f9d46d49edd29726a985ae50fbc2c85b47660a35704a178db. Its distinct CS3B successor freezes a resource-selected exact-fp64 low-rank or dense-fp32 hybrid after correcting the null-face and fp64 gather laws. The corrected worst-case all-14 projection is 2,191,899,544,519 FLOPs (99.67605112792626% of 2^41), with at most 9,742 projected calls; this is only a projection, not a meter pass.

Recorded measurement:

    Standalone dense-fp32 CS3 preregistration SHA-256 3ddf814fa95737303fa0184d963f9ff734e6ff250d76ef5ab9dbd1db8197275e was sealed HOLD before source under author receipt SHA-256 9ab43dd3e385f0bf043e23668ef0eb5563f32f4c5e913fcdd4e6a5e64cb5ca70 and independent receipt SHA-256 060cfbab98c85f7f9d46d49edd29726a985ae50fbc2c85b47660a35704a178db. Its distinct CS3B successor freezes a resource-selected exact-fp64 low-rank or dense-fp32 hybrid after correcting the null-face and fp64 gather laws. The corrected worst-case all-14 projection is 2,191,899,544,519 FLOPs (99.67605112792626% of 2^41), with at most 9,742 projected calls; this is only a projection, not a meter pass. Exact source SHA-256 49060ee223137d98fc2b43d94e4d5e5d99290c7289bccd3043e3e078775b0aaa is frozen under HOLD SHA-256 ac29611a20280de259c7ab2f970d865b09ca1c417d1ac09d64aeab5a89d23bb5 and passed independent static source audit SHA-256 27e5d66752520ebf287f7bc362330489849816b3d8cec0a1c7628aa931ca6c6c. The V1 runner/release is terminally held because its result serializer uses undefined lowercase `true`. V2 corrected only that spelling but independent audit SHA-256 a4375efde9b86cace58bf59ba4af294fc3b645eed7c0b49cf487b2e482403753 found four custody/fail-closed defects and requires a distinct V3. Neither runner was executed; the candidate was never imported or executed and both output namespaces remain absent. This closes only the exact V1/V2 probe custody forms, not the frozen CS3B source or the broader repair science. No full-width, literal-meter, accuracy, package, or submission claim exists.

Program source is in [`code/`](code/), preserving the relative layout below `.`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_connected_shutdown_v3/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_connected_shutdown_v3/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_connected_shutdown_v3/run_source_probe.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_connected_shutdown_v3/run_source_probe.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_connected_shutdown_v3/run_source_probe_v2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_connected_shutdown_v3/run_source_probe_v2.py)
- [code/estimator.py](code/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_connected_shutdown_v2/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_connected_shutdown_v2/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_connected_shutdown_v2/run_target_free_fullwidth_comparator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_connected_shutdown_v2/run_target_free_fullwidth_comparator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_connected_shutdown_v2/verify_independent_fullwidth_comparator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_connected_shutdown_v2/verify_independent_fullwidth_comparator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_connected_shutdown_v2/verify_independent_literal_meter_terminal.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_connected_shutdown_v2/verify_independent_literal_meter_terminal.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_manual_rho_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_manual_rho_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/meter_protocol/run_synthetic_meter.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/meter_protocol/run_synthetic_meter.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/meter_protocol/verify_meter_run.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/meter_protocol/verify_meter_run.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/q24_cs3_localized_masked_pairing_covariance_design_v1/run_risk_microprobe.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/q24_cs3_localized_masked_pairing_covariance_design_v1/run_risk_microprobe.py)

<details>
<summary>Additional programs and dependencies (1)</summary>

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/q24_cs3b_hybrid_exact_lowrank_pairing_covariance_design_v1/verify_independent_source_audit_scaffold_cs3b_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/q24_cs3b_hybrid_exact_lowrank_pairing_covariance_design_v1/verify_independent_source_audit_scaffold_cs3b_v1.py)

</details>

## Evidence

- [INDEPENDENT_FULLWIDTH_COMPARATOR_AUDIT_V1.json#L1](../../wall-and-residual/complete-connected-state-shutdown-residual-0-4026352697983384-s-exceeded/INDEPENDENT_FULLWIDTH_COMPARATOR_AUDIT_V1.json#L1)
- [BOARD_PROMOTION_GATE_RESULT_V1.json#L1](../../wall-and-residual/complete-connected-state-shutdown-residual-0-4026352697983384-s-exceeded/BOARD_PROMOTION_GATE_RESULT_V1.json#L1)
- [INDEPENDENT_LITERAL_METER_TERMINAL_AUDIT_CS2_V1.json#L1](../../wall-and-residual/complete-connected-state-shutdown-residual-0-4026352697983384-s-exceeded/INDEPENDENT_LITERAL_METER_TERMINAL_AUDIT_CS2_V1.json#L1)
- [PREREGISTER.md#L1](PREREGISTER.md#L1)
- [SOURCE_HOLD_V1.json#L1](SOURCE_HOLD_V1.json#L1)
- [INDEPENDENT_DENSE_CS3_DESIGN_HOLD_V1.json#L1](INDEPENDENT_DENSE_CS3_DESIGN_HOLD_V1.json#L1)
- [PREREGISTER.md#L212](../../../instruments/connected-state-shutdown-probe-intercepted-first-reset-before-recursive/PREREGISTER.md#L212)
- [PREREGISTER_CORRECTION_V1.md#L1](PREREGISTER_CORRECTION_V1.md#L1)
- [NULL_NONFINITE_CLARIFICATION_V1.md#L1](NULL_NONFINITE_CLARIFICATION_V1.md#L1)
- [RESOURCE_LAW_CORRECTION_V1.md#L110](RESOURCE_LAW_CORRECTION_V1.md#L110)
- [DESIGN_RECEIPT_V1.json#L82](DESIGN_RECEIPT_V1.json#L82)
- [INDEPENDENT_CORRECTED_DESIGN_AUDIT_CS3B_V1.json#L126](../../../instruments/connected-state-shutdown-probe-intercepted-first-reset-before-recursive/INDEPENDENT_CORRECTED_DESIGN_AUDIT_CS3B_V1.json#L126)
- [SOURCE_BUILD_RELEASE_V1.json#L1](SOURCE_BUILD_RELEASE_V1.json#L1)
- [SOURCE_HOLD_V1.json#L1](../../../instruments/connected-state-shutdown-probe-intercepted-first-reset-before-recursive/SOURCE_HOLD_V1.json#L1)
- [INDEPENDENT_STATIC_SOURCE_AUDIT_CS3B_V1.json#L122](../../../instruments/connected-state-shutdown-probe-intercepted-first-reset-before-recursive/INDEPENDENT_STATIC_SOURCE_AUDIT_CS3B_V1.json#L122)
- [SOURCE_PROBE_PREREGISTER_V1.md#L1](../../../instruments/connected-state-shutdown-probe-intercepted-first-reset-before-recursive/SOURCE_PROBE_PREREGISTER_V1.md#L1)
- [SOURCE_PROBE_STATIC_BINDING_V1.json#L1](../../../instruments/connected-state-shutdown-probe-intercepted-first-reset-before-recursive/SOURCE_PROBE_STATIC_BINDING_V1.json#L1)
- [SOURCE_PROBE_PREREGISTER_TRACE_CLARIFICATION_V1.json#L1](../../../instruments/connected-state-shutdown-probe-intercepted-first-reset-before-recursive/SOURCE_PROBE_PREREGISTER_TRACE_CLARIFICATION_V1.json#L1)
- [SOURCE_PROBE_RELEASE_V1.json#L1](SOURCE_PROBE_RELEASE_V1.json#L1)
- [SOURCE_PROBE_V1_ERRATUM.json#L1](../../../instruments/connected-state-shutdown-probe-intercepted-first-reset-before-recursive/SOURCE_PROBE_V1_ERRATUM.json#L1)
- [SOURCE_PROBE_RELEASE_V2.json#L1](SOURCE_PROBE_RELEASE_V2.json#L1)
- [INDEPENDENT_SOURCE_PROBE_RELEASE_AUDIT_CS3B_V2.json#L1](../../../instruments/connected-state-shutdown-probe-intercepted-first-reset-before-recursive/INDEPENDENT_SOURCE_PROBE_RELEASE_AUDIT_CS3B_V2.json#L1)
- [SOURCE_PROBE_V2_ERRATUM.json#L1](../../../instruments/connected-state-shutdown-probe-intercepted-first-reset-before-recursive/SOURCE_PROBE_V2_ERRATUM.json#L1)

<details>
<summary>All result files and records (23)</summary>

- [INDEPENDENT_FULLWIDTH_COMPARATOR_AUDIT_V1.json](../../wall-and-residual/complete-connected-state-shutdown-residual-0-4026352697983384-s-exceeded/INDEPENDENT_FULLWIDTH_COMPARATOR_AUDIT_V1.json) · result
- [BOARD_PROMOTION_GATE_RESULT_V1.json](../../wall-and-residual/complete-connected-state-shutdown-residual-0-4026352697983384-s-exceeded/BOARD_PROMOTION_GATE_RESULT_V1.json) · result
- [INDEPENDENT_LITERAL_METER_TERMINAL_AUDIT_CS2_V1.json](../../wall-and-residual/complete-connected-state-shutdown-residual-0-4026352697983384-s-exceeded/INDEPENDENT_LITERAL_METER_TERMINAL_AUDIT_CS2_V1.json) · result
- [PREREGISTER.md](PREREGISTER.md) · record
- [SOURCE_HOLD_V1.json](SOURCE_HOLD_V1.json) · result
- [INDEPENDENT_DENSE_CS3_DESIGN_HOLD_V1.json](INDEPENDENT_DENSE_CS3_DESIGN_HOLD_V1.json) · result
- [PREREGISTER.md](../../../instruments/connected-state-shutdown-probe-intercepted-first-reset-before-recursive/PREREGISTER.md) · record
- [PREREGISTER_CORRECTION_V1.md](PREREGISTER_CORRECTION_V1.md) · record
- [NULL_NONFINITE_CLARIFICATION_V1.md](NULL_NONFINITE_CLARIFICATION_V1.md) · record
- [RESOURCE_LAW_CORRECTION_V1.md](RESOURCE_LAW_CORRECTION_V1.md) · record
- [DESIGN_RECEIPT_V1.json](DESIGN_RECEIPT_V1.json) · result
- [INDEPENDENT_CORRECTED_DESIGN_AUDIT_CS3B_V1.json](../../../instruments/connected-state-shutdown-probe-intercepted-first-reset-before-recursive/INDEPENDENT_CORRECTED_DESIGN_AUDIT_CS3B_V1.json) · result
- [SOURCE_BUILD_RELEASE_V1.json](SOURCE_BUILD_RELEASE_V1.json) · result
- [SOURCE_HOLD_V1.json](../../../instruments/connected-state-shutdown-probe-intercepted-first-reset-before-recursive/SOURCE_HOLD_V1.json) · result
- [INDEPENDENT_STATIC_SOURCE_AUDIT_CS3B_V1.json](../../../instruments/connected-state-shutdown-probe-intercepted-first-reset-before-recursive/INDEPENDENT_STATIC_SOURCE_AUDIT_CS3B_V1.json) · result
- [SOURCE_PROBE_PREREGISTER_V1.md](../../../instruments/connected-state-shutdown-probe-intercepted-first-reset-before-recursive/SOURCE_PROBE_PREREGISTER_V1.md) · record
- [SOURCE_PROBE_STATIC_BINDING_V1.json](../../../instruments/connected-state-shutdown-probe-intercepted-first-reset-before-recursive/SOURCE_PROBE_STATIC_BINDING_V1.json) · result
- [SOURCE_PROBE_PREREGISTER_TRACE_CLARIFICATION_V1.json](../../../instruments/connected-state-shutdown-probe-intercepted-first-reset-before-recursive/SOURCE_PROBE_PREREGISTER_TRACE_CLARIFICATION_V1.json) · result
- [SOURCE_PROBE_RELEASE_V1.json](SOURCE_PROBE_RELEASE_V1.json) · result
- [SOURCE_PROBE_V1_ERRATUM.json](../../../instruments/connected-state-shutdown-probe-intercepted-first-reset-before-recursive/SOURCE_PROBE_V1_ERRATUM.json) · result
- [SOURCE_PROBE_RELEASE_V2.json](SOURCE_PROBE_RELEASE_V2.json) · result
- [INDEPENDENT_SOURCE_PROBE_RELEASE_AUDIT_CS3B_V2.json](../../../instruments/connected-state-shutdown-probe-intercepted-first-reset-before-recursive/INDEPENDENT_SOURCE_PROBE_RELEASE_AUDIT_CS3B_V2.json) · result
- [SOURCE_PROBE_V2_ERRATUM.json](../../../instruments/connected-state-shutdown-probe-intercepted-first-reset-before-recursive/SOURCE_PROBE_V2_ERRATUM.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-24-hybrid-connected-state-shutdown-source-held-after-serializer-target) · [Copy provenance](../../../PROVENANCE.json)

