# All-layer-loss refit rejected: late-layer residual prediction removes less than 0.1% of error power

2026-10-11T08:42Z · experiment · rejected · measured

A gradient-boosting fit of the per-layer residual from the head's own 128 inputs, plus cheap variance-mode candidates and the 30 raw features, removes under 0.1% of the noise-free residual power at layers 12-14, as at layer 15; more supervision cannot find what the layer does not hold.

Recorded measurement:

    layer 12: 0.06 / -0.16 / -0.30%; layer 13: 0.06 / 0.05 / -0.10%; layer 14: 0.02 / 0.04 / -0.06% (gate 5%)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/physicist`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/t96_ceiling.py](code/t96_ceiling.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1047](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1047)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1052](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1052)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1059](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1059)

<details>
<summary>All result files and records (2)</summary>

- [t96_out.txt](t96_out.txt) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Additional layer-15 inputs rejected: opening all ten masks adds 7.5% MSE](../../late-layer-exactness/additional-layer-15-inputs-rejected-opening-all-ten-masks/README.md)

[Research graph](../../../graph/index.html#node=2026-10-11-all-layer-loss-refit-rejected-late-layer-residual-prediction) · [Copy provenance](../../../PROVENANCE.json)

