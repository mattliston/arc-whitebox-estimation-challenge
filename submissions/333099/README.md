# Submission #333099: scatter-based gathers failed one wall and three residual-time limits

2026-09-30T10:21:44Z · submission · failed · measured

The scatter-gather source passed local40-network scoring and20-network serving but failed four official cases. Failures concentrated by worker, but the record did not identify the underlying cause; the prior best submission remained unchanged.

Recorded measurement:

    Official score0.007422294264012133;96 completed/4failed;index46 roundtrip120.23137727199992s TIME_EXHAUSTED;indices54/61/73 residual.433952342000012/.40215605199999516/.4023950760000048s.

Public score: 0.007422294264012133. Status: GRADED. Failed networks: 4.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [official_failure_333099_v1.json#L4](official_failure_333099_v1.json#L4)
- [SUBMISSION_LEDGER.md#L897](../../records/SUBMISSION_LEDGER.md#L897)
- [HANDOFF_CURRENT.md.before_s35_rr_20260930T1308Z#L136](../../experiments/setup-budget/all-setup-keys-fused-eagerly-degraded-setup-drops-bindings/HANDOFF_CURRENT.md.before_s35_rr_20260930T1308Z#L136)

<details>
<summary>All result files and records (25)</summary>

- [official_failure_333099_v1.json](official_failure_333099_v1.json) · record
- [submission_receipt_v1.json](submission_receipt_v1.json) · record
- [eval_333099_20260930T105510241753Z_30892.json](eval_333099_20260930T105510241753Z_30892.json) · result
- [eval_333099_public_20260930T105512423815Z.json](eval_333099_public_20260930T105512423815Z.json) · result
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [HANDOFF_CURRENT.md.before_s35_rr_20260930T1308Z](../../experiments/setup-budget/all-setup-keys-fused-eagerly-degraded-setup-drops-bindings/HANDOFF_CURRENT.md.before_s35_rr_20260930T1308Z) · record
- [eval_333099_20260930T102159334106Z_18630.json](eval_333099_20260930T102159334106Z_18630.json) · result
- [eval_333099_20260930T102159334106Z_18630_attempt.json](eval_333099_20260930T102159334106Z_18630_attempt.json) · result
- [eval_333099_20260930T102159334106Z_18630_summary.json](eval_333099_20260930T102159334106Z_18630_summary.json) · result
- [eval_333099_20260930T102322697170Z_18973.json](eval_333099_20260930T102322697170Z_18973.json) · result
- [eval_333099_20260930T102322697170Z_18973_attempt.json](eval_333099_20260930T102322697170Z_18973_attempt.json) · result
- [eval_333099_20260930T102322697170Z_18973_summary.json](eval_333099_20260930T102322697170Z_18973_summary.json) · result
- [eval_333099_20260930T105510241753Z_30892_attempt.json](eval_333099_20260930T105510241753Z_30892_attempt.json) · result
- [eval_333099_20260930T105510241753Z_30892_summary.json](eval_333099_20260930T105510241753Z_30892_summary.json) · result
- [eval_333099_public_20260930T102324308879Z.json](eval_333099_public_20260930T102324308879Z.json) · result
- [eval_333099_public_20260930T102454785969Z.json](eval_333099_public_20260930T102454785969Z.json) · result
- [eval_333099_public_20260930T102635015897Z.json](eval_333099_public_20260930T102635015897Z.json) · result
- [eval_333099_public_20260930T102735034620Z.json](eval_333099_public_20260930T102735034620Z.json) · result
- [eval_333099_public_20260930T102938515364Z.json](eval_333099_public_20260930T102938515364Z.json) · result
- [eval_333099_public_20260930T103145735878Z.json](eval_333099_public_20260930T103145735878Z.json) · result
- [eval_333099_public_20260930T103409524762Z.json](eval_333099_public_20260930T103409524762Z.json) · result
- [eval_333099_public_20260930T103739452045Z.json](eval_333099_public_20260930T103739452045Z.json) · result
- [eval_333099_public_20260930T104130846928Z.json](eval_333099_public_20260930T104130846928Z.json) · result
- [eval_333099_public_20260930T104832415958Z.json](eval_333099_public_20260930T104832415958Z.json) · result
- [eval_333099.json](eval_333099.json) · result

</details>

## Related work

- ← shipped in: [Birth-direction linear projection and delta factorization: total adjusted gain reaches 0.7325507236314643%](../../experiments/arithmetic-kernels/birth-direction-linear-projection-delta-factorization-total-adjusted-gain/README.md)
- ← shipped in: [Old-tier intermediate finite scans removed: two-network adjusted score improves 0.08089221%](../../experiments/flop-accounting/old-tier-intermediate-finite-scans-removed-two-network-adjusted/README.md)
- ← shipped in: [Packed pair-block symmetry: exact outputs on all eight networks with 0.2085143273% adjusted gain](../../experiments/arithmetic-kernels/packed-pair-block-symmetry-exact-outputs-all-eight-networks/README.md)
- ← shipped in: [Packed pairs, depth-seven reads and scan removal combined: adjusted gain 0.5912725672%](../../experiments/flop-accounting/packed-pairs-depth-seven-reads-scan-removal-combined-adjusted/README.md)
- ← shipped in: [Three-shape depth-seven read with eager bindings: eight-network adjusted gain is 0.3015740835%](../../experiments/arithmetic-kernels/three-shape-depth-seven-read-eager-bindings-eight-network/README.md)
- ← shipped in: [Unused birth-direction projection omitted: eight-network adjusted increment improves 0.0474183352%](../../experiments/flop-accounting/unused-birth-direction-projection-omitted-eight-network-adjusted-increment/README.md)

[Research graph](../../graph/index.html#node=submission-333099) · [Copy provenance](../../PROVENANCE.json)

