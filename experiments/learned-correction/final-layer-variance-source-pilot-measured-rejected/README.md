# Final-layer variance-source pilot: measured and rejected

2026-09-06T06:36Z · experiment · rejected · measured

All 24 target-free rows passed the unchanged-prefix and decomposition checks. The fixed matched 20-train / 4-held-out final-head pilot used the original 1e9-sample mean references, one 40-epoch fit per arm, and frozen P. All 24 target-free rows passed the unchanged-prefix and decomposition checks.

Recorded measurement:

    All 24 target-free rows passed the unchanged-prefix and decomposition checks. The fixed matched 20-train / 4-held-out final-head pilot used the original 1e9-sample mean references, one 40-epoch fit per arm, and frozen P. The source arm's pooled final raw MSE was 1.0039325113 times the bank and 1.0006956792 times the zero-input control, with only 1/4 bank wins. The predeclared .90/.92/3-win gates fail. NumPy/MPS head differences were at most 4.366e-11, so this is not a head-packaging discrepancy.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/compact_final_head_v1.py](code/compact_final_head_v1.py)
- [code/train_final_variance_source_probe_v1.py](code/train_final_variance_source_probe_v1.py)

## Evidence

- [FINAL_VARIANCE_SOURCE_PROBE_V1_CLOSED.md#L1](FINAL_VARIANCE_SOURCE_PROBE_V1_CLOSED.md#L1)
- [FINAL_VARIANCE_SOURCE_PROBE_V1_CLOSED.md#L12](FINAL_VARIANCE_SOURCE_PROBE_V1_CLOSED.md#L12)

<details>
<summary>All result files and records (3)</summary>

- [FINAL_VARIANCE_SOURCE_PROBE_V1_CLOSED.md](FINAL_VARIANCE_SOURCE_PROBE_V1_CLOSED.md) · record
- [final_variance_source_probe_verdict_v1.json](final_variance_source_probe_verdict_v1.json) · result
- [FINAL_VARIANCE_SOURCE_PROBE_DESIGN_V1.md](FINAL_VARIANCE_SOURCE_PROBE_DESIGN_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-final-layer-variance-source-pilot-measured-rejected) · [Copy provenance](../../../PROVENANCE.json)

