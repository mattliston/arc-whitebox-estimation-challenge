# Submission #333429: float32 arithmetic, half-zero strips, refreshed setup keys and separate tree levels

2026-10-02T02:26:54Z · submission · adopted · measured

S36 is the S35 estimator with the same physics and learned parameters plus the engineer's float32 fixes and full half-zero strips, a prebind list regenerated from a fresh key census, the release lane's un-fuse of output trees and the thin low-rank-star products as plain matmuls. It was fired on stage A, stage C and the first two networks of stage D (the machine was 6x slow) and graded 2.031e-9, a new bank, -1.74% against S35, entirely from the bill; the release lane's wall and residual projection was right to 1 s and 0.005 s.

Recorded measurement:

    adjusted 2.0308300278939022e-9 (S35 #333111 2.0672e-9, -1.74%; expected about 2.02e-9); final-layer MSE 1.712e-8 (unchanged to four digits), all-layer 6.775e-9, utilisation 0.11861112340797263 (-1.78%); 100/100 networks, wall 75.3 / 80.0 s (projected 74.4 / 85.7; S35 82.5 / 94.9), residual 0.283 / 0.314 s (projected 0.288 / 0.321); stage C bill 0.98182x of S35, requests 0.9468x; fresh error -0.107% (10 fresh) / -0.201% (10 validation) stack, -0.238% thin-star diet; setup 29,571,934,767 flops on all five workers; residual per request 6.93 microseconds; 5th on the board

Public score: 2.0308300278939022e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)
- [lead_stage_d_s36.sh](lead_stage_d_s36.sh)

## Evidence

- [SUBMISSION_LEDGER.md#L898](../../records/SUBMISSION_LEDGER.md#L898)
- [SUBMISSION_LEDGER.md#L901](../../records/SUBMISSION_LEDGER.md#L901)
- [TEAM_LOG_2026-10-01_to_10-22.md#L82](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L82)
- [TEAM_LOG_2026-10-01_to_10-22.md#L90](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L90)
- [TEAM_LOG_2026-10-01_to_10-22.md#L91](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L91)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L363](../../records/SESSION_NOTES_2026-10-01_to_10-03.md#L363)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L371](../../records/SESSION_NOTES_2026-10-01_to_10-03.md#L371)

<details>
<summary>All result files and records (10)</summary>

- [attempt.json](attempt.json) · record
- [receipt.json](receipt.json) · record
- [eval_summary.txt](eval_summary.txt) · result
- [eval_333429.json](eval_333429.json) · result
- [lead_d_s36.log](lead_d_s36.log) · log
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [eval_333429-fc59f64b.json](eval_333429-fc59f64b.json) · result
- [eval_333429-baaefcd6.json](eval_333429-baaefcd6.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [SESSION_NOTES_2026-10-01_to_10-03.md](../../records/SESSION_NOTES_2026-10-01_to_10-03.md) · record

</details>

## Related work

- ← shipped in: [Float32 reductions and live-row operations lower charged work by 0.669% with rounding-level error changes](../../experiments/flop-accounting/float32-reductions-live-row-operations-lower-charged-work-0/README.md)
- ← shipped in: [Half-zero strip kernel for odd-32 dimensions lowers charged work by 1.30%](../../experiments/arithmetic-kernels/half-zero-strip-kernel-odd-32-dimensions-lowers-charged/README.md)
- ← shipped in: [Refreshed setup-key census raises fused-tree usage from 56.7% to approximately 75%](../../experiments/setup-budget/refreshed-setup-key-census-raises-fused-tree-usage-56/README.md)
- ← shipped in: [Restricted tree-level fusion preserves output bytes and lowers kernel time by 13.1%](../../experiments/wall-and-residual/restricted-tree-level-fusion-preserves-output-bytes-lowers-kernel/README.md)
- ← shipped in: [Plain matrix multiplication for thin star products removes 4,350 calls but raises charged work by 0.48%](../../experiments/flop-accounting/plain-matrix-multiplication-thin-star-products-removes-4-350/README.md)
- ← supersedes: [Submission #333435: fourth-cumulant column shifts and per-unit kurtosis for pairs up to age two](../333435/README.md)

[Research graph](../../graph/index.html#node=submission-333429) · [Copy provenance](../../PROVENANCE.json)

