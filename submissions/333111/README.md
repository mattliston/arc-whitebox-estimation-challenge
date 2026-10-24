# Submission #333111: scatter-based gather with an eager setup prefix of 128 frames

2026-09-30T12:15:41Z · submission · adopted · measured

Submission #333111 used scatter-based gather with an eager setup prefix of 128 frames. It graded successfully on all 100 public networks and became the best public adjusted score at that time.

Recorded measurement:

    Official adjusted score 2.0671897807351535e-09; utilization 0.1207592737860341; 100 networks, 0 failures. Official raw MSE 1.7121260107444414e-08; adjusted improvement over #333050 1.4206319025635517%; telemetry {"completed":100,"failed":0,"max_predict_wall_s":94.92989715899967,"mean_predict_wall_s":82.46909589151994,"max_residual_wall_s":0.3386857149999969,"mean_residual_wall_s":0.30424053573999954,"max_setup_wall_s":4.2903561200000695,"mean_setup_wall_s":4.28096739980001,"setup_executions":5,"mean_compute_utilization":0.1207592737860341}. Local qualification {"n":40,"adjusted_pct":-1.3891618281817286,"adjusted_ci95_pct":[-1.5390645416199964,-1.237427150886874],"adjusted_wins":40,"raw_pct":0.05635591945092333,"raw_ci95_pct":[-0.09417374727095168,0.2090074338372256],"mean_util_pct":-1.4443124984922506,"mean_cpu_pct":5.1344712104882895,"max_per_network_adjusted_regression_pct":-0.16676471192293985}.

Public score: 2.0671897807351535e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [official_bank_S35_v1.json#L3](official_bank_S35_v1.json#L3)
- [SUBMISSION_LEDGER.md#L897](../../records/SUBMISSION_LEDGER.md#L897)

<details>
<summary>All result files and records (41)</summary>

- [official_bank_S35_v1.json](official_bank_S35_v1.json) · record
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [submission_receipt_v1.json](submission_receipt_v1.json) · record
- [eval_333111.json](eval_333111.json) · result
- [FIRE_CHECKS_v1.json](FIRE_CHECKS_v1.json) · result
- [broad_assessment.json](../../records/research/2026-09-30-eager-setup-prefix-size-changes-scheduling-reliability-after-all/broad_assessment.json) · result
- [archive_receipt.json](archive_receipt.json) · record
- [eval_333111_20260930T121627870840Z_7396.json](eval_333111_20260930T121627870840Z_7396.json) · result
- [eval_333111_20260930T121627870840Z_7396_attempt.json](eval_333111_20260930T121627870840Z_7396_attempt.json) · result
- [eval_333111_20260930T121627870840Z_7396_summary.json](eval_333111_20260930T121627870840Z_7396_summary.json) · result
- [eval_333111_20260930T122139484156Z_8873.json](eval_333111_20260930T122139484156Z_8873.json) · result
- [eval_333111_20260930T122139484156Z_8873_attempt.json](eval_333111_20260930T122139484156Z_8873_attempt.json) · result
- [eval_333111_20260930T122139484156Z_8873_summary.json](eval_333111_20260930T122139484156Z_8873_summary.json) · result
- [eval_333111_20260930T124427622669Z_16180.json](eval_333111_20260930T124427622669Z_16180.json) · result
- [eval_333111_20260930T124427622669Z_16180_attempt.json](eval_333111_20260930T124427622669Z_16180_attempt.json) · result
- [eval_333111_20260930T124427622669Z_16180_summary.json](eval_333111_20260930T124427622669Z_16180_summary.json) · result
- [eval_333111_20260930T124614689306Z_17069.json](eval_333111_20260930T124614689306Z_17069.json) · result
- [eval_333111_20260930T124614689306Z_17069_attempt.json](eval_333111_20260930T124614689306Z_17069_attempt.json) · result
- [eval_333111_20260930T124614689306Z_17069_summary.json](eval_333111_20260930T124614689306Z_17069_summary.json) · result
- [eval_333111_public_20260930T121809128001Z.json](eval_333111_public_20260930T121809128001Z.json) · result
- [eval_333111_public_20260930T121809128001Z_summary.json](eval_333111_public_20260930T121809128001Z_summary.json) · result
- [eval_333111_public_20260930T122055850007Z.json](eval_333111_public_20260930T122055850007Z.json) · result
- [eval_333111_public_20260930T122055850007Z_summary.json](eval_333111_public_20260930T122055850007Z_summary.json) · result
- [eval_333111_public_20260930T122441075109Z.json](eval_333111_public_20260930T122441075109Z.json) · result
- [eval_333111_public_20260930T122441075109Z_summary.json](eval_333111_public_20260930T122441075109Z_summary.json) · result
- [eval_333111_public_20260930T122829100589Z.json](eval_333111_public_20260930T122829100589Z.json) · result
- [eval_333111_public_20260930T122829100589Z_summary.json](eval_333111_public_20260930T122829100589Z_summary.json) · result
- [eval_333111_public_20260930T123246960743Z.json](eval_333111_public_20260930T123246960743Z.json) · result
- [eval_333111_public_20260930T123246960743Z_summary.json](eval_333111_public_20260930T123246960743Z_summary.json) · result
- [eval_333111_public_20260930T123448244347Z.json](eval_333111_public_20260930T123448244347Z.json) · result
- [eval_333111_public_20260930T123448244347Z_summary.json](eval_333111_public_20260930T123448244347Z_summary.json) · result
- [eval_333111_public_20260930T123750807533Z.json](eval_333111_public_20260930T123750807533Z.json) · result
- [eval_333111_public_20260930T123750807533Z_summary.json](eval_333111_public_20260930T123750807533Z_summary.json) · result
- [eval_333111_public_20260930T124056028762Z.json](eval_333111_public_20260930T124056028762Z.json) · result
- [eval_333111_public_20260930T124056028762Z_summary.json](eval_333111_public_20260930T124056028762Z_summary.json) · result
- [eval_333111_public_20260930T124317433501Z.json](eval_333111_public_20260930T124317433501Z.json) · result
- [eval_333111_public_20260930T124317433501Z_summary.json](eval_333111_public_20260930T124317433501Z_summary.json) · result
- [eval_333111_public_20260930T124619335394Z.json](eval_333111_public_20260930T124619335394Z.json) · result
- [eval_333111_public_20260930T124619335394Z_summary.json](eval_333111_public_20260930T124619335394Z_summary.json) · result
- [eval_333111-a12a2092.json](eval_333111-a12a2092.json) · result
- [eval_333111-fe6e25c3.json](eval_333111-fe6e25c3.json) · result

</details>

## Related work

- ← shipped in: [Birth-direction linear projection and delta factorization: total adjusted gain reaches 0.7325507236314643%](../../experiments/arithmetic-kernels/birth-direction-linear-projection-delta-factorization-total-adjusted-gain/README.md)
- ← shipped in: [Covariance cast and star reuse: 40-network gain 0.7287995864472% stays below submission threshold](../../experiments/flop-accounting/covariance-cast-star-reuse-40-network-gain-0-7287995864472/README.md)
- ← shipped in: [Exact plane-buffer and cast reuse: eight-network adjusted increment improves 0.05731041992435015%](../../experiments/flop-accounting/exact-plane-buffer-cast-reuse-eight-network-adjusted-increment/README.md)
- ← shipped in: [First 128 setup keys fused eagerly: degraded setup retains all bindings and exact FLOPs](../../experiments/setup-budget/first-128-setup-keys-fused-eagerly-degraded-setup-retains/README.md)
- ← shipped in: [Old-tier intermediate finite scans removed: two-network adjusted score improves 0.08089221%](../../experiments/flop-accounting/old-tier-intermediate-finite-scans-removed-two-network-adjusted/README.md)
- ← shipped in: [Packed pair-block symmetry: exact outputs on all eight networks with 0.2085143273% adjusted gain](../../experiments/arithmetic-kernels/packed-pair-block-symmetry-exact-outputs-all-eight-networks/README.md)
- ← shipped in: [Packed pairs, depth-seven reads and scan removal combined: adjusted gain 0.5912725672%](../../experiments/flop-accounting/packed-pairs-depth-seven-reads-scan-removal-combined-adjusted/README.md)
- ← shipped in: [Read cache and cast composition: exact eight-network adjusted increment improves 0.0543709922464%](../../experiments/flop-accounting/read-cache-cast-composition-exact-eight-network-adjusted-increment/README.md)
- ← shipped in: [Sentinel scatter gathers with a forward-only cache: all 40 validation scores improve](../../experiments/flop-accounting/sentinel-scatter-gathers-forward-only-cache-all-40-validation/README.md)
- ← shipped in: [Three redundant dtype casts removed: two-network adjusted increment improves 0.02372371701289122%](../../experiments/flop-accounting/three-redundant-dtype-casts-removed-two-network-adjusted-increment/README.md)
- ← shipped in: [Three-shape depth-seven read with eager bindings: eight-network adjusted gain is 0.3015740835%](../../experiments/arithmetic-kernels/three-shape-depth-seven-read-eager-bindings-eight-network/README.md)
- ← shipped in: [Unused birth-direction projection omitted: eight-network adjusted increment improves 0.0474183352%](../../experiments/flop-accounting/unused-birth-direction-projection-omitted-eight-network-adjusted-increment/README.md)
- ← shipped in: [Zero/scatter row expansions with cast removal: 40-network gain passes but long-worker memory fails](../../experiments/flop-accounting/zero-scatter-row-expansions-cast-removal-40-network-gain/README.md)

[Research graph](../../graph/index.html#node=submission-333111) · [Copy provenance](../../PROVENANCE.json)

