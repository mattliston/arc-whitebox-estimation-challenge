# Complete connected-state shutdown: residual 0.4026352697983384 s exceeded the 0.4 s cap

2026-08-24 · experiment · rejected · measured

Exact immutable V2 source SHA-256 4efb2c40d21d16c7b51ee87741323c3b99aded664fd75a2d5e00164f7b584779 under SOURCE_HOLD SHA-256 c35332c6df5bfb2c7ad7105f200117078d55901956164d7717abdb8f85c2417c passed the independent strict source-and-carrier audit SHA-256 cba8bfbb437f32aa377ca72da74b9504894e6380429e2acf0e5a059d7ef8afcb with verifier SHA-256 5673d297a3c5fd7dd7bd1fd2ec1a57c6ad9f51259d94870fbfa6bc3513a4a342. The audit found zero carrier or one-push violations across 140 recursive states and 2,327 certified-invalid unit-reset activations. The separately released, target-free width-1024 comparator then passed its exact source-stability gate under receipt SHA-256 151de92da658f15de016b256c74cb00255bd1d4aea8c4d8ef3934bab10665645: all six parent/child arms completed finite, 51 shutdown activations occurred, and every carrier violation count was zero.

Recorded measurement:

    Exact immutable V2 source SHA-256 4efb2c40d21d16c7b51ee87741323c3b99aded664fd75a2d5e00164f7b584779 under SOURCE_HOLD SHA-256 c35332c6df5bfb2c7ad7105f200117078d55901956164d7717abdb8f85c2417c passed the independent strict source-and-carrier audit SHA-256 cba8bfbb437f32aa377ca72da74b9504894e6380429e2acf0e5a059d7ef8afcb with verifier SHA-256 5673d297a3c5fd7dd7bd1fd2ec1a57c6ad9f51259d94870fbfa6bc3513a4a342. The audit found zero carrier or one-push violations across 140 recursive states and 2,327 certified-invalid unit-reset activations. The separately released, target-free width-1024 comparator then passed its exact source-stability gate under receipt SHA-256 151de92da658f15de016b256c74cb00255bd1d4aea8c4d8ef3934bab10665645: all six parent/child arms completed finite, 51 shutdown activations occurred, and every carrier violation count was zero. However, the independently frozen board-promotion preregistration SHA-256 e837715025dfd314afa296681a619ffae5e3b2829956342e9a9ed8c28e0a6d88 required each seed's final parent/child disagreement MSE to be at most 1e-8; all three failed at 4.0925566476689264e-8, 6.451118991829646e-8, and 8.449135636469015e-8. The one-use literal meter then matched the exact 2,131,652,533,303-FLOP and 9,504-call projection and passed FLOP, 25.368778-second wall, and 4,589,715,456-byte descendant-RSS gates, but residual time was 0.4026352697983384 seconds, exceeding the 0.4-second cap by 0.002635269798338391 seconds. Independent terminal audit SHA-256 a136f4031c14b354a00d983e249a406b5c03642a202042032e9f426034c836cc confirms a finite float32 [16,1024] pre-fallback candidate tensor and a residual-only terminal resource failure. The scored zero fallback is not candidate accuracy. This exact CS2 spelling is closed with no retry, package, submission, board revival, or raw-MSE claim.

Program source is in [`code/`](code/), preserving the relative layout below `.`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_connected_shutdown_v2/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_connected_shutdown_v2/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_connected_shutdown_v2/run_source_probe.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_connected_shutdown_v2/run_source_probe.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_connected_shutdown_v2/verify_independent_strict_audit.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_connected_shutdown_v2/verify_independent_strict_audit.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_connected_shutdown_v2/run_target_free_fullwidth_comparator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_connected_shutdown_v2/run_target_free_fullwidth_comparator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_connected_shutdown_v2/verify_independent_fullwidth_comparator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_connected_shutdown_v2/verify_independent_fullwidth_comparator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_connected_shutdown_v2/verify_independent_literal_meter_terminal.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_connected_shutdown_v2/verify_independent_literal_meter_terminal.py)
- [code/estimator.py](code/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_connected_shutdown_v2_board_promotion_v1/verify_promotion_hold.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_connected_shutdown_v2_board_promotion_v1/verify_promotion_hold.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_manual_rho_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_manual_rho_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/meter_protocol/run_synthetic_meter.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/meter_protocol/run_synthetic_meter.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/meter_protocol/verify_meter_run.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/meter_protocol/verify_meter_run.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/q24_two_point_moment_repair_probe_v1/run_probe.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/q24_two_point_moment_repair_probe_v1/run_probe.py)

## Evidence

- [CARRIER_COHERENCE_ERRATUM_V1.json#L1](CARRIER_COHERENCE_ERRATUM_V1.json#L1)
- [REDTEAM_RECEIPT_V1.json#L1](REDTEAM_RECEIPT_V1.json#L1)
- [DO_NOT_FREEZE_CARRIER_COHERENCE_HOLD.json#L1](DO_NOT_FREEZE_CARRIER_COHERENCE_HOLD.json#L1)
- [PREREGISTER.md#L1](PREREGISTER.md#L1)
- [DESIGN.md#L1](DESIGN.md#L1)
- [POST_SOURCE_RECONCILIATION_V1.md#L1](POST_SOURCE_RECONCILIATION_V1.md#L1)
- [DESIGN_RECEIPT_V1.json#L1](DESIGN_RECEIPT_V1.json#L1)
- [PREREGISTER-e83d1bfd.md#L1](PREREGISTER-e83d1bfd.md#L1)
- [results.json#L1](results.json#L1)
- [SOURCE_HOLD_V2.json#L1](SOURCE_HOLD_V2.json#L1)
- [INDEPENDENT_STRICT_AUDIT_CS2.json#L1](INDEPENDENT_STRICT_AUDIT_CS2.json#L1)
- [PREREGISTER_FULLWIDTH_COMPARATOR_AND_METER_V1.json#L1](PREREGISTER_FULLWIDTH_COMPARATOR_AND_METER_V1.json#L1)
- [ROOT_FULLWIDTH_COMPARATOR_RELEASE_SCAFFOLD_V1.json#L1](ROOT_FULLWIDTH_COMPARATOR_RELEASE_SCAFFOLD_V1.json#L1)
- [ROOT_LITERAL_METER_RELEASE_SCAFFOLD_V1.json#L1](ROOT_LITERAL_METER_RELEASE_SCAFFOLD_V1.json#L1)
- [STATIC_RELEASE_SCAFFOLD_VERIFICATION_V1.json#L1](STATIC_RELEASE_SCAFFOLD_VERIFICATION_V1.json#L1)
- [CLAIM.json#L1](CLAIM.json#L1)
- [receipt.json#L1](receipt.json#L1)
- [PREREGISTER-e8377150.md#L1](PREREGISTER-e8377150.md#L1)
- [INDEPENDENT_FULLWIDTH_COMPARATOR_AUDIT_V1.json#L143](INDEPENDENT_FULLWIDTH_COMPARATOR_AUDIT_V1.json#L143)
- [BOARD_PROMOTION_GATE_RESULT_V1.json#L113](BOARD_PROMOTION_GATE_RESULT_V1.json#L113)
- [STATIC_SCAFFOLD_AND_REJECTION_AUDIT_V1.json#L1](STATIC_SCAFFOLD_AND_REJECTION_AUDIT_V1.json#L1)
- [ROOT_LITERAL_METER_RELEASE_CS2_V1.json#L171](ROOT_LITERAL_METER_RELEASE_CS2_V1.json#L171)
- [receipt-93ccaef3.json#L615](receipt-93ccaef3.json#L615)
- [INDEPENDENT_LITERAL_METER_TERMINAL_AUDIT_CS2_V1.json#L142](INDEPENDENT_LITERAL_METER_TERMINAL_AUDIT_CS2_V1.json#L142)

<details>
<summary>All result files and records (24)</summary>

- [CARRIER_COHERENCE_ERRATUM_V1.json](CARRIER_COHERENCE_ERRATUM_V1.json) · result
- [REDTEAM_RECEIPT_V1.json](REDTEAM_RECEIPT_V1.json) · result
- [DO_NOT_FREEZE_CARRIER_COHERENCE_HOLD.json](DO_NOT_FREEZE_CARRIER_COHERENCE_HOLD.json) · result
- [PREREGISTER.md](PREREGISTER.md) · record
- [DESIGN.md](DESIGN.md) · record
- [POST_SOURCE_RECONCILIATION_V1.md](POST_SOURCE_RECONCILIATION_V1.md) · record
- [DESIGN_RECEIPT_V1.json](DESIGN_RECEIPT_V1.json) · result
- [PREREGISTER-e83d1bfd.md](PREREGISTER-e83d1bfd.md) · record
- [results.json](results.json) · result
- [SOURCE_HOLD_V2.json](SOURCE_HOLD_V2.json) · result
- [INDEPENDENT_STRICT_AUDIT_CS2.json](INDEPENDENT_STRICT_AUDIT_CS2.json) · result
- [PREREGISTER_FULLWIDTH_COMPARATOR_AND_METER_V1.json](PREREGISTER_FULLWIDTH_COMPARATOR_AND_METER_V1.json) · result
- [ROOT_FULLWIDTH_COMPARATOR_RELEASE_SCAFFOLD_V1.json](ROOT_FULLWIDTH_COMPARATOR_RELEASE_SCAFFOLD_V1.json) · result
- [ROOT_LITERAL_METER_RELEASE_SCAFFOLD_V1.json](ROOT_LITERAL_METER_RELEASE_SCAFFOLD_V1.json) · result
- [STATIC_RELEASE_SCAFFOLD_VERIFICATION_V1.json](STATIC_RELEASE_SCAFFOLD_VERIFICATION_V1.json) · result
- [CLAIM.json](CLAIM.json) · result
- [receipt.json](receipt.json) · result
- [PREREGISTER-e8377150.md](PREREGISTER-e8377150.md) · record
- [INDEPENDENT_FULLWIDTH_COMPARATOR_AUDIT_V1.json](INDEPENDENT_FULLWIDTH_COMPARATOR_AUDIT_V1.json) · result
- [BOARD_PROMOTION_GATE_RESULT_V1.json](BOARD_PROMOTION_GATE_RESULT_V1.json) · result
- [STATIC_SCAFFOLD_AND_REJECTION_AUDIT_V1.json](STATIC_SCAFFOLD_AND_REJECTION_AUDIT_V1.json) · result
- [ROOT_LITERAL_METER_RELEASE_CS2_V1.json](ROOT_LITERAL_METER_RELEASE_CS2_V1.json) · result
- [receipt-93ccaef3.json](receipt-93ccaef3.json) · result
- [INDEPENDENT_LITERAL_METER_TERMINAL_AUDIT_CS2_V1.json](INDEPENDENT_LITERAL_METER_TERMINAL_AUDIT_CS2_V1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-24-complete-connected-state-shutdown-residual-0-4026352697983384-s-exceeded) · [Copy provenance](../../../PROVENANCE.json)

