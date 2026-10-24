# Frozen response-row reduction improves independent14 adjusted score by 3.25073% and is retained

2026-09-07T03:20Z · experiment · adopted · measured

On the independent 8-public + 6-virgin panel with stored 4M Monte Carlo truths (noise-corrected raw = observed final MSE minus mean mc_se15^2, adjusted = per-net raw x max(.1, F/2^41)), the frozen response-row candidate (SHA 5a07832f) improved both groups over the matched bank in one fresh process each, so it was retained as a local component and packaged (Whest 0.16 archive 755010 bytes) but not uploaded because 4.834e-9 does not cross the below-4e-9 milestone. The NumPy-free client/OpenBLAS server persistent test on contexts 35/100/500/3016500/35 passed (wall 33.985-34.882 s, residual <= .144213 s, peak combined RSS 7240744960 bytes).

Recorded measurement:

    Public8 5.437274909258601e-9 -> 5.2532154360922335e-9 (3.38514%); virgin6 4.409194462222713e-9 -> 4.275608385711634e-9 (3.02972%); combined14 4.996669003386079e-9 -> 4.83424098592912e-9 (3.25073%); combined NC raw 2.2592877636382216e-8 -> 2.2454337457623213e-8 (0.61320% lower); mean complete FLOPs 485374855176.7857 -> 472413324017.2143 (2.67042% lower); max native-vs-remote output difference 3.09944e-6.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/check_response_rows_panel_v1.py](code/check_response_rows_panel_v1.py)
- [code/qualify_response_rows_remote_v1.py](code/qualify_response_rows_remote_v1.py)
- [code/candidate_response_rows_current_runtime_v1.py](code/candidate_response_rows_current_runtime_v1.py)
- [code/build_response_rows_cost_v1.py](code/build_response_rows_cost_v1.py)
- [code/build_response_rows_cost_v2.py](code/build_response_rows_cost_v2.py)
- [code/build_response_rows_current_runtime_v1.py](code/build_response_rows_current_runtime_v1.py)
- [code/candidate_fullpair_runtime_d5_t8192_v1.py](code/candidate_fullpair_runtime_d5_t8192_v1.py)
- [code/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py](code/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py)
- [code/candidate_response_rows_cost_v1.py](code/candidate_response_rows_cost_v1.py)
- [code/candidate_response_rows_cost_v2.py](code/candidate_response_rows_cost_v2.py)
- [code/candidate_response_rows_parent_cost_v1.py](code/candidate_response_rows_parent_cost_v1.py)
- [code/check_remote_series_v1.py](code/check_remote_series_v1.py)

<details>
<summary>Additional programs and dependencies (9)</summary>

- [code/check_response_rows_adjusted_v1.py](code/check_response_rows_adjusted_v1.py)
- [code/k3_radial_numpy.py](code/k3_radial_numpy.py)
- [code/radial_binary_compact.py](code/radial_binary_compact.py)
- [code/radial_compact_feed_fullpair_v1.py](code/radial_compact_feed_fullpair_v1.py)
- [code/radial_compact_radialfeed_v1.py](code/radial_compact_radialfeed_v1.py)
- [code/radial_cubic.py](code/radial_cubic.py)
- [code/radial_mehler.py](code/radial_mehler.py)
- [code/radial_response_rows_v1.py](code/radial_response_rows_v1.py)
- [code/run_remote_series_v1.py](code/run_remote_series_v1.py)

</details>

## Evidence

- [RETAINED_COMPONENT_RESPONSE_ROWS_V1.md#L16](RETAINED_COMPONENT_RESPONSE_ROWS_V1.md#L16)
- [RETAINED_COMPONENT_RESPONSE_ROWS_V1.md#L48](RETAINED_COMPONENT_RESPONSE_ROWS_V1.md#L48)
- [RETAINED_COMPONENT_RESPONSE_ROWS_V1.md#L66](RETAINED_COMPONENT_RESPONSE_ROWS_V1.md#L66)
- [CONTINUATION_2026_09_06_2020_PDT.md#L27](CONTINUATION_2026_09_06_2020_PDT.md#L27)

<details>
<summary>All result files and records (5)</summary>

- [RETAINED_COMPONENT_RESPONSE_ROWS_V1.md](RETAINED_COMPONENT_RESPONSE_ROWS_V1.md) · result
- [RESPONSE_ROWS_PANEL_ACCEPTANCE_V1.md](RESPONSE_ROWS_PANEL_ACCEPTANCE_V1.md) · record
- [result.json](result.json) · result
- [response_rows_current_runtime_remote_qualified_v1.json](response_rows_current_runtime_remote_qualified_v1.json) · result
- [CONTINUATION_2026_09_06_2020_PDT.md](CONTINUATION_2026_09_06_2020_PDT.md) · record

</details>

## Related work

- ← enables: [Frozen response-row reduction improves two-network adjusted score by 3.63221%](../frozen-response-row-reduction-improves-two-network-adjusted-score/README.md)

[Research graph](../../../graph/index.html#node=2026-09-07-frozen-response-row-reduction-improves-independent14-adjusted-score-3) · [Copy provenance](../../../PROVENANCE.json)

