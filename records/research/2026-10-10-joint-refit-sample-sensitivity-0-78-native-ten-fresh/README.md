# Joint-refit sample sensitivity: -0.78% native on ten fresh networks versus -2.0% research on 50 ranks

2026-10-10T08:05Z · finding · informational · measured

Native candidate generation 44 (S44) on ten fresh networks missed the -1.5% submission rule, but PyTorch on the same ten also read only -1.00% (network 548 +3.6% on both) while PyTorch on the 50 read -2.0%; the submission went on the 50 with a 0.2% native offset and the board confirmed -2.26%.

Recorded measurement:

    native ten -0.78% pooled (se 0.61, 8 of 10); torch same ten -1.00%; per-network native / torch 548 +3.95 / +3.60; graded -2.26%

## Programs

- [../../../experiments/learned-correction/norm-coupling-diagonal-learned-input-lowers-native-mse-2/code/judge_ten.py](../../../experiments/learned-correction/norm-coupling-diagonal-learned-input-lowers-native-mse-2/code/judge_ten.py)
- [../2026-10-10-joint-refit-evaluation-0-16-native-mse-ten-fresh/parity_level.py](../2026-10-10-joint-refit-evaluation-0-16-native-mse-ten-fresh/parity_level.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L578](../../TEAM_LOG_2026-10-01_to_10-22.md#L578)
- [TEAM_LOG_2026-10-01_to_10-22.md#L580](../../TEAM_LOG_2026-10-01_to_10-22.md#L580)
- [TEAM_LOG_2026-10-01_to_10-22.md#L581](../../TEAM_LOG_2026-10-01_to_10-22.md#L581)
- [TEAM_LOG_2026-10-01_to_10-22.md#L583](../../TEAM_LOG_2026-10-01_to_10-22.md#L583)
- [TEAM_LOG_2026-10-01_to_10-22.md#L595](../../TEAM_LOG_2026-10-01_to_10-22.md#L595)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Joint learned-array and head-input refit adopted: -2.006% on 50 networks without the old-pair quartic](../2026-10-10-joint-learned-array-head-input-refit-adopted-2-006/README.md)

[Research graph](../../../graph/index.html#node=2026-10-10-joint-refit-sample-sensitivity-0-78-native-ten-fresh) · [Copy provenance](../../../PROVENANCE.json)

