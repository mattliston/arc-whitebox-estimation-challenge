# Trace-preserving chi-radius fourth-plane intervention: closed

2026-09-06T18:52Z · experiment · rejected · measured

All seven existing coherent rank35 cells, pre-layers8..14, were included. The references are the original8M post covariances with the same fixed next-weight/response. The prescribed one-step plane-read information gate FAILS:

Recorded measurement:

    | Fixed measurement | New / original error | Required | | --- | ---: | ---: | | Pooled covariance-entry error | .9995147882954947 | <=.95 | | Pooled next-Gaussian-mean-response error | .9967214554461685 | <=.90 | | Next-value winning cells | 7/7 | >=5/7 | All seven existing coherent rank35 cells, pre-layers8..14, were included. The references are the original8M post covariances with the same fixed next-weight/response. These are not final-layer raw MSE or adjusted score. The direction was consistently weakly helpful, but only~.33% pooled next-value improvement, far below the frozen10% gate. This does not license scaling that direction: no amplitude or radius fit follows. - Actual original mean and covariance reproduced byte-exactly on7/7. - Explicit original standardized K31 reproduced the same result exactly. - Original r was uniquely recovered from cached k4; two cells required an adjacent representable double to undo q=r/3 rounding. No alternate radius was evaluated against targets. - Current pre-frame K22 trace preservation relative error<=5.58e-16. - All original symmetric F32 covariance targets reproduced byte-exactly. - Original C remains in both nonlinear pair reads. Only the construction of the new fourth tensor uses the explicitly reported symmetric C. - Current-layer mean feedback is absent in this established diagnostic; previous coherent history is fixed. Both means and covariance centering were recomputed by the actual nonlinear map, not a linearized shortcut. Next-value increment/target norm was.009508..013595, cosine.11795..18138. Worker completed5.162seconds, peak1,484,554,240bytes; no job remains live. Receipt: chi_radius_trace_shape_value_r35_v1.json/.npz. Source SHA f4451abeedfeb3cec75b9bc150bbb37008ab740882d6c362a9900edb6a40b2ae.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/certify_chi_radius_trace_shape_v1.py](code/astra_phase2_reexamination_v1/certify_chi_radius_trace_shape_v1.py)
- [code/astra_phase2_reexamination_v1/check_chi_radius_trace_shape_value_v1.py](code/astra_phase2_reexamination_v1/check_chi_radius_trace_shape_value_v1.py)
- [code/astra_phase2_reexamination_v1/chi_radius_trace_shape_v1.py](code/astra_phase2_reexamination_v1/chi_radius_trace_shape_v1.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)
- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)

## Evidence

- [CHI_RADIUS_TRACE_SHAPE_VALUE_V1_CLOSED.md#L1](CHI_RADIUS_TRACE_SHAPE_VALUE_V1_CLOSED.md#L1)
- [CHI_RADIUS_TRACE_SHAPE_VALUE_V1_CLOSED.md#L27](CHI_RADIUS_TRACE_SHAPE_VALUE_V1_CLOSED.md#L27)

<details>
<summary>All result files and records (3)</summary>

- [CHI_RADIUS_TRACE_SHAPE_VALUE_V1_CLOSED.md](CHI_RADIUS_TRACE_SHAPE_VALUE_V1_CLOSED.md) · record
- [chi_radius_trace_shape_value_r35_v1.json](chi_radius_trace_shape_value_r35_v1.json) · result
- [CHI_RADIUS_TRACE_SHAPE_VALUE_DESIGN_V1.md](CHI_RADIUS_TRACE_SHAPE_VALUE_DESIGN_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-trace-preserving-chi-radius-fourth-plane-intervention-closed) · [Copy provenance](../../../PROVENANCE.json)

