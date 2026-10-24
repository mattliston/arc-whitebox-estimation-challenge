# Native effects below 1% require ten networks and at least two arithmetic draws of the reference

2026-10-02T17:28Z · lesson · informational · measured

Changing float32 operation order creates roughly 0.5% final-MSE noise per network even when the estimator is unchanged. Comparisons against one stored reference share its draw, so small native effects must be measured on ten networks against at least two reference draws.

Recorded measurement:

    per-network draw 0.49%; two-draw difference sd 0.69%; shared-reference correlation 0.79; two standing networks about 0.9 points too favorable; ten networks against one draw se 0.22% from rounding; 100-network board averages draw to 0.05%

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L396](../../TEAM_LOG_2026-10-01_to_10-22.md#L396)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [RESULT.md](../../../experiments/arithmetic-kernels/half-zero-strip-kernel-odd-32-dimensions-lowers-charged/RESULT.md) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-native-effects-below-1-require-ten-networks-least-two) · [Copy provenance](../../../PROVENANCE.json)

