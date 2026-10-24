# Submission #335696: reassociated native algebra and identity caching

2026-10-13T04:29:08Z · submission · adopted · measured

s51k2 plus Astra's native algebra stack (rounding level) and the identity cache; sealed on 8 fresh networks and fast-gated; the board took the metered FLOPs cut with a small MSE gain.

Recorded measurement:

    1.53222e-9 (-0.302% vs #335661; paired mean log -0.296%, se 0.048, 34/50; final MSE -0.075%, bill -0.223%); util 0.11447; 0 failed; residual max 0.371 s; seal -0.2853% (se 0.1161, 6 of 8); official public score 1.5321795020740832e-09; official utilisation 0.11447214117681143; failed networks 0

Public score: 1.5321795020740832e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)
- [candidate.py](candidate.py)
- [../../records/research/2026-10-13-final-layer-hub-quartic-ages-3-4-strength-0/candidate.py](../../records/research/2026-10-13-final-layer-hub-quartic-ages-3-4-strength-0/candidate.py)
- [../../experiments/arithmetic-kernels/remainder-32-strided-read-variant-byte-identical-but-does/code/opus_reexam_20261001_v1/engineer/a62_modules/s49/s51k2tw_x/candidate.py](../../experiments/arithmetic-kernels/remainder-32-strided-read-variant-byte-identical-but-does/code/opus_reexam_20261001_v1/engineer/a62_modules/s49/s51k2tw_x/candidate.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1883](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1883)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1891](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1891)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1915](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1915)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1916](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1916)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1919](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1919)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1929](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1929)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1930](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1930)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1942](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1942)
- [SUBMISSION_LEDGER.md#L957](../../records/SUBMISSION_LEDGER.md#L957)
- [SUBMISSION_LEDGER.md#L958](../../records/SUBMISSION_LEDGER.md#L958)
- [SUBMISSION_LEDGER.md#L960](../../records/SUBMISSION_LEDGER.md#L960)

<details>
<summary>All result files and records (7)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [stage_a.json](stage_a.json) · record
- [receipt.json](receipt.json) · record
- [eval_335696.json](eval_335696.json) · result
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [seal_s51k3_PREREG.json](seal_s51k3_PREREG.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← shipped in: [Identity caching and repeated-row cuts adopted; float32 substitution rejected because rounding exceeds savings](../../records/research/2026-10-13-identity-caching-repeated-row-cuts-adopted-float32-substitution-rejected/README.md)
- ← shipped in: [Reassociated push/read algebra adopted: -0.195% FLOPs at rounding-level prediction change](../../records/research/2026-10-13-reassociated-push-read-algebra-adopted-0-195-flops-rounding/README.md)
- ← supersedes: [Submission #335661: byte-identical repeated-row and product reductions](../335661/README.md)
- supersedes: [Submission #335709: final-layer hub quartic ages 3-4 and byte-identical product cuts](../335709/README.md)

[Research graph](../../graph/index.html#node=submission-335696) · [Copy provenance](../../PROVENANCE.json)

