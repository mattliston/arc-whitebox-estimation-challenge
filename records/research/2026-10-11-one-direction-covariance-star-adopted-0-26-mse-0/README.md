# One-direction covariance star adopted: -0.26% MSE for +0.035% FLOPs

2026-10-11T16:55Z · decision · adopted · measured

The row-sum star plus its leading rank-one piece (8 n^2 flops a layer) delivers most of the factored star's gain at near-zero metered FLOPs; the native rank-8 / 16 / 32 forms cost +0.60 / +1.18 / +2.37% metered FLOPs against PyTorch's -0.49% and were closed.

Recorded measurement:

    k = -1 on 16 fresh: MSE -0.26% (se 0.11, 11 of 16), bill +0.035%, score -0.23% (2.2 se); torch curve ss 2 / 4 / 8 / 16 / 32 -0.23 / -0.30 / -0.43 / -0.49 / -0.49%; band 7-15 only -0.17%

## Programs

- [fragment_star_dir_v1.py](fragment_star_dir_v1.py)
- [star_ss_torch_v2.py](star_ss_torch_v2.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1109](../../TEAM_LOG_2026-10-01_to_10-22.md#L1109)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1113](../../TEAM_LOG_2026-10-01_to_10-22.md#L1113)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1115](../../TEAM_LOG_2026-10-01_to_10-22.md#L1115)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1117](../../TEAM_LOG_2026-10-01_to_10-22.md#L1117)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1118](../../TEAM_LOG_2026-10-01_to_10-22.md#L1118)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1125](../../TEAM_LOG_2026-10-01_to_10-22.md#L1125)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Module audit: learned modules use 5.4% of FLOPs and every removal costs more error than FLOPs saved](../2026-10-11-module-audit-learned-modules-use-5-4-flops-every/README.md)
- enables: [Native capacity retests reject rank 384 and rank-8 star: adjusted +0.378% and +0.454%](../../../experiments/join-sketch/native-capacity-retests-reject-rank-384-rank-8-star/README.md)
- shipped in: [Submission #335411: carrier-strength refit, one-direction covariance star and plane-read column cut](../../../submissions/335411/README.md)

[Research graph](../../../graph/index.html#node=2026-10-11-one-direction-covariance-star-adopted-0-26-mse-0) · [Copy provenance](../../../PROVENANCE.json)

