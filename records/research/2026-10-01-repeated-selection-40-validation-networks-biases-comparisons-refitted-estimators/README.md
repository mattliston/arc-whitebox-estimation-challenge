# Repeated selection on 40 validation networks biases comparisons of refitted estimators

2026-10-01T21:50Z · lesson · informational · measured

Least-squares refits of the final-layer correction gain 0.3-0.4% on untouched newer-rank networks and about zero on the 40 validation networks, because the current parameters were tuned on the old 160 training networks and selected on those 40. Rule: learned-parameter and rounding-level candidates are judged on newer-rank networks the candidate has not trained on (100 such networks were held out for good in the trainer's list).

Recorded measurement:

    held-out change by group: newer ranks -0.32%, old TRAIN160 ranks -0.03%, VAL-A +0.04%; rounding-level kernel variants +0.02..+0.5% on valid40, neutral on fresh

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L42](../../TEAM_LOG_2026-10-01_to_10-22.md#L42)
- [TEAM_LOG_2026-10-01_to_10-22.md#L47](../../TEAM_LOG_2026-10-01_to_10-22.md#L47)

<details>
<summary>All result files and records (2)</summary>

- [runB_val.txt](runB_val.txt) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-01-repeated-selection-40-validation-networks-biases-comparisons-refitted-estimators) · [Copy provenance](../../../PROVENANCE.json)

