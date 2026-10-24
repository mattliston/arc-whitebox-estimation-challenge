# Submission #335259: per-layer refit and a gate-floor 1e-3 dead-side push cut

2026-10-10T21:55:41Z · submission · adopted · measured

candidate generation 46 (S46) = candidate generation 45 (S45) + refit round late-band per-layer refit (jmI) (arrays, head rows, cohort-age local fourth-order term (K4L) age-1 0.738 and iso per layer at 10-14 through per-layer literal switch (DLY@), mean-correction covariance-fold coefficient (kappa_C) by band) + the push's dead-side row cut at gate 1e-3; the board read -0.3% where the untouched estimate said -0.66%.

Recorded measurement:

    1.581e-9 (-0.32% vs S45); final MSE 1.354e-8 (-0.22%); share 0.11678; 100/100; wall 83.75 / 92.73 s; residual 0.31 / 0.34 s; requests 43,398 / 42,543; official public score 1.5811980561457008e-09; official utilisation 0.11678280631680536; failed networks 0

Public score: 1.5811980561457008e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)
- [candidate.py](candidate.py)
- [../../records/research/2026-10-10-per-layer-fourth-order-isotropic-dial-refit-adopted-layers/candidate.py](../../records/research/2026-10-10-per-layer-fourth-order-isotropic-dial-refit-adopted-layers/candidate.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L708](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L708)
- [TEAM_LOG_2026-10-01_to_10-22.md#L712](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L712)
- [TEAM_LOG_2026-10-01_to_10-22.md#L714](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L714)
- [TEAM_LOG_2026-10-01_to_10-22.md#L728](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L728)
- [SUBMISSION_LEDGER.md#L929](../../records/SUBMISSION_LEDGER.md#L929)
- [SUBMISSION_LEDGER.md#L931](../../records/SUBMISSION_LEDGER.md#L931)
- [SUBMISSION_LEDGER.md#L932](../../records/SUBMISSION_LEDGER.md#L932)
- [SUBMISSION_LEDGER.md#L933](../../records/SUBMISSION_LEDGER.md#L933)
- [SUBMISSION_LEDGER.md#L934](../../records/SUBMISSION_LEDGER.md#L934)
- [SUBMISSION_LEDGER.md#L935](../../records/SUBMISSION_LEDGER.md#L935)
- [SUBMISSION_LEDGER.md#L936](../../records/SUBMISSION_LEDGER.md#L936)

<details>
<summary>All result files and records (7)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [stage_a.json](stage_a.json) · record
- [receipt.json](receipt.json) · record
- [eval_335259.json](eval_335259.json) · result
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [eval_335259-02e5f5a2.json](eval_335259-02e5f5a2.json) · result

</details>

## Related work

- ← shipped in: [Per-layer fourth-order and isotropic dial refit adopted at layers 10-14](../../records/research/2026-10-10-per-layer-fourth-order-isotropic-dial-refit-adopted-layers/README.md)
- ← supersedes: [Submission #335213: refitted learned arrays and jointly re-solved estimator dials](../335213/README.md)
- enables: [Submission #335261: kernel-price variant of the same estimator gives no FLOP advantage](../335261/README.md)
- enables: [Submission #335329 fails: network 14 exceeds the residual cap at 0.4083 s](../335329/README.md)
- supersedes: [Submission #335340: ridge-re-sized refit, request caches and seven-level full-packet strip pushes](../335340/README.md)

[Research graph](../../graph/index.html#node=submission-335259) · [Copy provenance](../../PROVENANCE.json)

