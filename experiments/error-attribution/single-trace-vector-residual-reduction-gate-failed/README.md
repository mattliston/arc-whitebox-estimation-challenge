# Single trace-vector residual reduction: gate failed

2026-09-06T01:14Z · experiment · rejected · measured

The explicit normal-operator/least-squares certificate passes below4.3e-14. An oracle vector b fitted ONLY to the four source-slice residual matrices of each independent stream leaves68.88%/69.93% relative slice error. This closes only the proposed reduction of the missing sector to Sym(b,C).

Recorded measurement:

    The explicit normal-operator/least-squares certificate passes below4.3e-14. An oracle vector b fitted ONLY to the four source-slice residual matrices of each independent stream leaves68.88%/69.93% relative slice error. After one-high projection, exact coordinate-repeat removal and the fixed next W read, weighted signal error ratio is .783437450, cosine .468384133. Cross-stream ratios .797512943/.781322756. These fail the predeclared .7 and .6 gates; no vector predictor or damping/metric sweep follows.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fit_residual_trace_representation_v1.py](code/fit_residual_trace_representation_v1.py)
- [code/residual_one_high_oracle_v1.py](code/residual_one_high_oracle_v1.py)

## Evidence

- [RESIDUAL_TRACE_REPRESENTATION_V1_CLOSED.md#L1](RESIDUAL_TRACE_REPRESENTATION_V1_CLOSED.md#L1)
- [RESIDUAL_TRACE_REPRESENTATION_V1_CLOSED.md#L7](RESIDUAL_TRACE_REPRESENTATION_V1_CLOSED.md#L7)

<details>
<summary>All result files and records (3)</summary>

- [RESIDUAL_TRACE_REPRESENTATION_V1_CLOSED.md](RESIDUAL_TRACE_REPRESENTATION_V1_CLOSED.md) · record
- [residual_trace_representation_r35_l12_v1.json](residual_trace_representation_r35_l12_v1.json) · result
- [RESIDUAL_TRACE_REPRESENTATION_DESIGN_V1.md](RESIDUAL_TRACE_REPRESENTATION_DESIGN_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-single-trace-vector-residual-reduction-gate-failed) · [Copy provenance](../../../PROVENANCE.json)

