# Submission #328317: native-trajectory residual observer on the order-24 closure

2026-08-25T00:00:23Z · submission · adopted · measured

Observer J (the per-channel transport-energy coherence residual observer) was re-fitted on the native tape of the deployable order-24 Hermite pair-series closure (Q24) host rather than the CP1792 development host, so that the whole estimator fits the budget. The protected transfer pilot passed all five frozen conjuncts on the fixed 40 validation networks (autonomous final mean squared error (MSE) about 0.35 of the host, all 40 improved, every rollout finite) and the promoted pilot model was banked. This was the first Phase 2 card to carry a learned correction and cut the raw error of the pure closure by about three times.

Recorded measurement:

    pilot: autonomous final MSE 3.107235383605394e-8 vs host 8.919980524938374e-8 (ratio 0.3483455344905993), 40/40 improve, max per-network 4.093294327260766e-8; board raw 3.116165888883416e-08, adjusted 3.0168601176930186e-08

Public score: 3.0168601176930186e-08. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)
- [PILOT_LITERAL_PAYLOAD.py](PILOT_LITERAL_PAYLOAD.py)
- [run_transfer_pilot.py](run_transfer_pilot.py)
- [../../experiments/learned-correction/host-uncertainty-observable-tape-augmentation-measured-rejected/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_transport_energy_pilot_v1/instrumented_estimator.py](../../experiments/learned-correction/host-uncertainty-observable-tape-augmentation-measured-rejected/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_transport_energy_pilot_v1/instrumented_estimator.py)
- [../../experiments/flop-accounting/dynamic-multistate-observer-full-688-out-fold-error-3/code/public_full688_local_innovation_transport_v1/dataset.py](../../experiments/flop-accounting/dynamic-multistate-observer-full-688-out-fold-error-3/code/public_full688_local_innovation_transport_v1/dataset.py)
- [../../experiments/learned-correction/learned-backward-adjoint-residual-observer-rejected-full-width-cross/code/public_full688_transport_energy_coherence_cv_v1/run_experiment.py](../../experiments/learned-correction/learned-backward-adjoint-residual-observer-rejected-full-width-cross/code/public_full688_transport_energy_coherence_cv_v1/run_experiment.py)

## Evidence

- [BOARD_RESULT_328317.json#L1](BOARD_RESULT_328317.json#L1)
- [PILOT_RESULT.json#L1](PILOT_RESULT.json#L1)

<details>
<summary>All result files and records (3)</summary>

- [BOARD_RESULT_328317.json](BOARD_RESULT_328317.json) · record
- [PILOT_RESULT.json](PILOT_RESULT.json) · result
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record

</details>

## Related work

- shipped in: [Transport-energy residual observer: out-of-fold error 3.2334958229470984e-8, above the acceptance threshold](../../experiments/flop-accounting/transport-energy-residual-observer-out-fold-error-3-2334958229470984e/README.md)
- supersedes: [Submission #328277: variance-floor trigger widened to twice the floor, raw 8.758639268080515e-8](../328277/README.md)
- ← supersedes: [Submission #328326: transport-energy residual observer refitted on all 688 development networks](../328326/README.md)

[Research graph](../../graph/index.html#node=submission-328317) · [Copy provenance](../../PROVENANCE.json)

