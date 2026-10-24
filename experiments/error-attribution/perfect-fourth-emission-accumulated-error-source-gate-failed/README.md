# Perfect fourth emission: accumulated-error source gate failed

2026-09-06T01:56Z · experiment · rejected · measured

Phase2, September5,2026. The v1 transfer-order failure is preserved; v2 produces both planned8M streams at unchanged seeds/counts/formulas. This rejects the specified natural-strength SINGLE-CELL correction to the current accumulated tensor error.

Recorded measurement:

    Phase2, September5,2026. The v1 transfer-order failure is preserved; v2 produces both planned8M streams at unchanged seeds/counts/formulas. Post-mean parity with the original third-oracle streams is BYTE EXACT for both. Q31/Q22 diagonal consistency is .00114945/.00114387; small CPU Torch/NumPy algebra differences below9e-19. Shared MPS bias is not removed. True fourth emission minus ACTUAL current radial distinct emission has delta-slice norms .28896/.28859. Against the accumulated post-third error, the q4 weighted signal error ratio is .992040038, cosine .103912887, correction/error norm .157185073. Cross-stream ratios1.00261099/.98356907. Unweighted signal ratio .995155449. The fixed .95/.4 and cross-stream gates FAIL; do not fit a new fourth core or expand sampling on this result. Streams90.25s/93.44s, host peaks2.277GB/2.096GB, sequential one MPS worker. All jobs completed. Receipts: fourth_emission_truth_r35_l12_A_v2.json/.npz, B counterparts, and fourth_emission_oracle_r35_l12_v2.json/.npz. Official bank remains329926; no candidate, raw score, panel or submission.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/analyze_fourth_emission_oracle_v2.py](code/analyze_fourth_emission_oracle_v2.py)
- [code/residual_one_high_oracle_v1.py](code/residual_one_high_oracle_v1.py)

## Evidence

- [FOURTH_EMISSION_ORACLE_V2_CLOSED.md#L1](FOURTH_EMISSION_ORACLE_V2_CLOSED.md#L1)
- [FOURTH_EMISSION_ORACLE_V2_CLOSED.md#L23](FOURTH_EMISSION_ORACLE_V2_CLOSED.md#L23)

<details>
<summary>All result files and records (4)</summary>

- [FOURTH_EMISSION_ORACLE_V2_CLOSED.md](FOURTH_EMISSION_ORACLE_V2_CLOSED.md) · record
- [fourth_emission_truth_r35_l12_A_v2.json](fourth_emission_truth_r35_l12_A_v2.json) · result
- [fourth_emission_oracle_r35_l12_v2.json](fourth_emission_oracle_r35_l12_v2.json) · result
- [FOURTH_EMISSION_ORACLE_DESIGN_V1.md](FOURTH_EMISSION_ORACLE_DESIGN_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-perfect-fourth-emission-accumulated-error-source-gate-failed) · [Copy provenance](../../../PROVENANCE.json)

