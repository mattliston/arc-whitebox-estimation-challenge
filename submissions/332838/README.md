# Submission #332838: precision changes with symmetric 128-row diagonal covariance blocks

2026-09-28T11:56:19Z · submission · adopted · measured

Submission #332838 used precision changes with symmetric 128-row diagonal covariance blocks. It graded successfully on all 100 public networks and became the best public adjusted score at that time.

Recorded measurement:

    Official adjusted score 2.115138679262642e-09; utilization 0.12442903127238424; 100 networks, 0 failures. Official raw MSE 1.700166714968532e-08; adjusted improvement over #332738 1.1606967395196621%; telemetry {"completed":100,"failed":0,"max_predict_wall_s":91.6020791269998,"mean_predict_wall_s":81.60478142314012,"max_residual_wall_s":0.3473809900000049,"mean_residual_wall_s":0.2967349890400001,"max_setup_wall_s":4.262517408002168,"mean_setup_wall_s":4.255802643599964,"setup_executions":5,"mean_compute_utilization":0.12442903127238424}. Local qualification {"n":40,"adjusted_pct":-1.1953067285634167,"adjusted_ci95_pct":[-1.3717380078348538,-1.028972795071422],"adjusted_wins":40,"raw_pct":-0.04596230681692681,"raw_ci95_pct":[-0.21478276864263676,0.11448015258999306],"mean_util_pct":-1.1488436239046718,"mean_cpu_pct":1.3620744266900298,"max_per_network_adjusted_regression_pct":-0.2935647961872623}.

Public score: 2.115138679262642e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [official_bank_S33_v1.json#L3](official_bank_S33_v1.json#L3)
- [SUBMISSION_LEDGER.md#L897](../../records/SUBMISSION_LEDGER.md#L897)

<details>
<summary>All result files and records (24)</summary>

- [official_bank_S33_v1.json](official_bank_S33_v1.json) · record
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [submission_receipt_v1.json](submission_receipt_v1.json) · record
- [eval_332838.json](eval_332838.json) · result
- [FIRE_CHECKS_v1.json](FIRE_CHECKS_v1.json) · result
- [broad_assessment.json](../../experiments/arithmetic-kernels/three-term-gaussian-evaluation-128-row-diagonal-blocks-all/broad_assessment.json) · result
- [archive_receipt.json](archive_receipt.json) · record
- [eval_332838_20260928T115929Z.json](eval_332838_20260928T115929Z.json) · result
- [eval_332838_20260928T120701Z_fresh.json](eval_332838_20260928T120701Z_fresh.json) · result
- [eval_332838_20260928T121220Z_fresh.json](eval_332838_20260928T121220Z_fresh.json) · result
- [eval_332838_20260928T121642Z_fresh.json](eval_332838_20260928T121642Z_fresh.json) · result
- [eval_332838_20260928T122438Z_fresh.json](eval_332838_20260928T122438Z_fresh.json) · result
- [eval_332838_20260928T122601Z_fresh.json](eval_332838_20260928T122601Z_fresh.json) · result
- [eval_332838_20260928T123600Z_fresh.json](eval_332838_20260928T123600Z_fresh.json) · result
- [eval_332838_20260928T115631783906Z_65966.json](eval_332838_20260928T115631783906Z_65966.json) · result
- [eval_332838_20260928T115631783906Z_65966_attempt.json](eval_332838_20260928T115631783906Z_65966_attempt.json) · result
- [eval_332838_20260928T115631783906Z_65966_summary.json](eval_332838_20260928T115631783906Z_65966_summary.json) · result
- [eval_332838_20260928T115824766904Z_67025.json](eval_332838_20260928T115824766904Z_67025.json) · result
- [eval_332838_20260928T115824766904Z_67025_attempt.json](eval_332838_20260928T115824766904Z_67025_attempt.json) · result
- [eval_332838_20260928T115824766904Z_67025_summary.json](eval_332838_20260928T115824766904Z_67025_summary.json) · result
- [eval_332838_20260928T122545771524Z_75237.json](eval_332838_20260928T122545771524Z_75237.json) · result
- [eval_332838_20260928T122545771524Z_75237_attempt.json](eval_332838_20260928T122545771524Z_75237_attempt.json) · result
- [eval_332838_20260928T122545771524Z_75237_summary.json](eval_332838_20260928T122545771524Z_75237_summary.json) · result
- [eval_332838-892e9121.json](eval_332838-892e9121.json) · result

</details>

## Related work

- ← shipped in: [Base-factor cache and 36 setup bindings combined: eight-network adjusted score improves 0.199586520%](../../experiments/flop-accounting/base-factor-cache-36-setup-bindings-combined-eight-network/README.md)
- ← shipped in: [Immutable common/difference base-factor cache: all outputs exact with 0.1158296% lower bill](../../experiments/flop-accounting/immutable-common-difference-base-factor-cache-all-outputs-exact/README.md)
- ← shipped in: [Nine depth-three setup bindings: requests fall by 996 cold and 870 warm](../../experiments/setup-budget/nine-depth-three-setup-bindings-requests-fall-996-cold/README.md)
- ← shipped in: [Three-term Gaussian evaluation and 128-row diagonal blocks: all 40 validation scores improve](../../experiments/arithmetic-kernels/three-term-gaussian-evaluation-128-row-diagonal-blocks-all/README.md)
- ← shipped in: [Twenty-seven additional setup bindings: cold bill falls 1,493,091,978 FLOPs](../../experiments/setup-budget/twenty-seven-additional-setup-bindings-cold-bill-falls-1/README.md)

[Research graph](../../graph/index.html#node=submission-332838) · [Copy provenance](../../PROVENANCE.json)

