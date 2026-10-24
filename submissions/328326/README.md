# Submission #328326: transport-energy residual observer refitted on all 688 development networks

2026-08-25T02:22:44Z · submission · adopted · measured

The native-tape residual observer was re-fitted for six epochs on all 688 observer_fit rows (488 fresh rows plus 200 byte-reused pilot rows), exported immutably and spliced into the order-24 Hermite pair-series closure (Q24) closure. The first literal meter failed only on residual wall (0.4945 s against the 0.4 s cap); a mechanical change that meters first-call scratch creation and then reuses fixed-width scratch arenas kept exact sequential byte parity and brought all three metered calls under the cap. The graded result improved #328317 by 5.62% and remained the raw-mean squared error (MSE) baseline (2.9409e-8) of every later Phase 2 card through #329754.

Recorded measurement:

    board raw 2.9409640056599073e-08, adjusted 2.847207506864414e-08 (vs #328317: -5.622354% raw, -5.623483% adjusted); in-sample observer_fit diagnostic 2.9736792457285063e-8 vs native host 9.016075357240868e-8; three literal calls: residual 0.3215303928591311 / 0.34646319644525647 / 0.35143496841192245 s, FLOPs 2,132,366,833,051 / 2,123,756,968,456 / 2,132,363,680,256, walls 24.89165858272463 / 24.72982758283615 / 24.954987000208348 s, max RSS 3,421,503,488 bytes; first meter residual 0.49450173741206527 s

Public score: 2.847207506864414e-08. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)
- [run_final_fit.py](run_final_fit.py)
- [generate_all688_row.py](generate_all688_row.py)
- [../../experiments/flop-accounting/dynamic-multistate-observer-full-688-out-fold-error-3/code/public_full688_local_innovation_transport_v1/dataset.py](../../experiments/flop-accounting/dynamic-multistate-observer-full-688-out-fold-error-3/code/public_full688_local_innovation_transport_v1/dataset.py)
- [../../experiments/learned-correction/learned-backward-adjoint-residual-observer-rejected-full-width-cross/code/public_full688_transport_energy_coherence_cv_v1/run_experiment.py](../../experiments/learned-correction/learned-backward-adjoint-residual-observer-rejected-full-width-cross/code/public_full688_transport_energy_coherence_cv_v1/run_experiment.py)
- [../../experiments/learned-correction/host-uncertainty-observable-tape-augmentation-measured-rejected/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_transport_energy_pilot_v1/instrumented_estimator.py](../../experiments/learned-correction/host-uncertainty-observable-tape-augmentation-measured-rejected/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_transport_energy_pilot_v1/instrumented_estimator.py)

## Evidence

- [BOARD_RESULT_328326.json#L1](BOARD_RESULT_328326.json#L1)
- [FABLE_HANDOFF_20260825.md#L13](FABLE_HANDOFF_20260825.md#L13)
- [FABLE_HANDOFF_20260825.md#L41](FABLE_HANDOFF_20260825.md#L41)
- [SUBMISSION_LEDGER.md#L675](../../records/SUBMISSION_LEDGER.md#L675)

<details>
<summary>All result files and records (6)</summary>

- [BOARD_RESULT_328326.json](BOARD_RESULT_328326.json) · record
- [METER_RESULT.json](METER_RESULT.json) · result
- [FINAL_FIT_SEAL.json](FINAL_FIT_SEAL.json) · record
- [SEQUENTIAL_PARITY_RESULT.json](SEQUENTIAL_PARITY_RESULT.json) · result
- [FABLE_HANDOFF_20260825.md](FABLE_HANDOFF_20260825.md) · record
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record

</details>

## Related work

- supersedes: [Submission #328317: native-trajectory residual observer on the order-24 closure](../328317/README.md)
- ← supersedes: [Submission #328400: conditional positive-semidefinite repair timed out to zero output](../328400/README.md)
- ← supersedes: [Submission #328433: pair-series trim to order 7 and output clamp](../328433/README.md)

[Research graph](../../graph/index.html#node=submission-328326) · [Copy provenance](../../PROVENANCE.json)

