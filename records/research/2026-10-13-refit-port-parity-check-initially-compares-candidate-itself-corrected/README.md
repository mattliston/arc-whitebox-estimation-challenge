# Refit port-parity check initially compares the candidate to itself; corrected change is below 2e-6

2026-10-13T20:35Z · finding · informational · measured

The candidate generation 56 (S56) port's native start-baseline parity had inadvertently used the candidate source on both sides, so it was not valid evidence; re-run with the absorbed start source against the frozen candidate on the check seed 7102248, the aggregate diagnostic formally fails (correlation 0.388, size 2.386, delivery D -3.843) because the PyTorch step itself moves the prediction 1.56e-6, below the 2e-6 rounding floor, while the native delta is 3.72e-6. A provenance audit found every checkpoint array, head row, carry and mapped literal exact, so the fail is the sub-floor step and not a port defect; the per-route pre-freeze gates and the fixed native seal rule stayed unchanged.

Recorded measurement:

    corrected parity on 7102248: corr 0.388, size 2.386, D -3.843; torch delta 1.56e-6 (floor 2e-6), native delta 3.72e-6; 28 checkpoint arrays, head rows, GK/K4L carry and literals exact; AST differs only in 10 learned-literal families

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2280](../../TEAM_LOG_2026-10-01_to_10-22.md#L2280)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2290](../../TEAM_LOG_2026-10-01_to_10-22.md#L2290)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2293](../../TEAM_LOG_2026-10-01_to_10-22.md#L2293)

<details>
<summary>All result files and records (2)</summary>

- [CORRECTED_PARITY_RESULT_v1.json](CORRECTED_PARITY_RESULT_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-13-refit-port-parity-check-initially-compares-candidate-itself-corrected) · [Copy provenance](../../../PROVENANCE.json)

