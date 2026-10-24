# Trace repair for dropping the lowest-energy quarter of old pairs recovers only 0.96639% of omitted energy

2026-10-15T16:55:42Z · experiment · rejected · measured

The saved-factor bottom-quarter pruning oracle removed 1472 of 5888 pairs using weighted cubic norm. Trace compensation barely changed the omitted tensor and left the folded nonlinear-pair read untouched.

Recorded measurement:

    omitted squared tensor norm 1.40177%; Frobenius 11.8396%; trace recovers .96639%; Frobenius 11.8396% to 11.7823%; modeled .595% bill credit

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_read_pair_coreset_audit_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/bottom_quarter_pair_pruning_oracle_v1.py](code/bottom_quarter_pair_pruning_oracle_v1.py)
- [code/bottom_quarter_trace_oracle_v1.py](code/bottom_quarter_trace_oracle_v1.py)
- [code/saved_pair_coreset_oracle_v1.py](code/saved_pair_coreset_oracle_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3432](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3432)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3431](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3431)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3434](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3434)

<details>
<summary>All result files and records (7)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [BOTTOM_QUARTER_FEASIBILITY_v1.json](../cubic-tensor-coreset-256-pairs-68-447-residual-tensor/BOTTOM_QUARTER_FEASIBILITY_v1.json) · result
- [BOTTOM_QUARTER_TRACE_CLOSURE_v1.json](../cubic-tensor-coreset-256-pairs-68-447-residual-tensor/BOTTOM_QUARTER_TRACE_CLOSURE_v1.json) · result
- [CLOSURE_v1.json](../cubic-tensor-coreset-256-pairs-68-447-residual-tensor/CLOSURE_v1.json) · result
- [BOTTOM_QUARTER_PAIR_PRUNING_ORACLE_v1.json](../cubic-tensor-coreset-256-pairs-68-447-residual-tensor/BOTTOM_QUARTER_PAIR_PRUNING_ORACLE_v1.json) · result
- [BOTTOM_QUARTER_TRACE_ORACLE_v1.json](../cubic-tensor-coreset-256-pairs-68-447-residual-tensor/BOTTOM_QUARTER_TRACE_ORACLE_v1.json) · result
- [SAVED_PAIR_CORESET_ORACLE_v1.json](../cubic-tensor-coreset-256-pairs-68-447-residual-tensor/SAVED_PAIR_CORESET_ORACLE_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-15-trace-repair-dropping-lowest-energy-quarter-old-pairs-recovers) · [Copy provenance](../../../PROVENANCE.json)

