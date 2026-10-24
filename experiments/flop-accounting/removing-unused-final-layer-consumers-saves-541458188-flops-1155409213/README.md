# Removing unused final-layer consumers saves 541458188 FLOPs and .1155409213% adjusted error

2026-09-08 · experiment · adopted · measured

Exact terminal work removal passed independent-truth and production-boundary checks while preserving the estimator's mathematical outputs.

Recorded measurement:

    QUALIFIED_COMPONENT_2026_09_07_PDT.md:
    ## Same-configuration independent14 result
    | Public8 | 2.4316999741373497e-8 | 469195523392 | 5.196969836377938e-9 |
    | Historical virgin6 | 1.996803939251903e-8 | 464946370412.1667 | 4.23000801867066e-9 |
    | Combined14 | 2.245315959186444e-8 | 467374457829.2143 | 4.782557628789104e-9 |
    Same-frame parent combined14 adjusted4.788089831892398e-9; ratio
    .9988445907872394, a .1155409213% gain. Public and virgin ratios are
    .998847451516621 and .9988399045827347. Mean saving541458188 FLOPs.
    mean of per-network raw*max(.1,F/2^41) are used. Observed combined14 raw
    Native panel maximum wall19.4938634s and peakRSS7091453952 bytes. Those
    Wall29.6402-30.4614s; residual .117763-.128788s; combinedRSS7178469376
    bytes, below the unchanged8e9 gate. Maximum native-to-remote discrepancy

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/terminal_dead_consumers_v1/check_panel.py](code/terminal_dead_consumers_v1/check_panel.py)
- [code/terminal_dead_consumers_v1/check_remote.py](code/terminal_dead_consumers_v1/check_remote.py)
- [code/terminal_dead_consumers_v1/check_remote_v2.py](code/terminal_dead_consumers_v1/check_remote_v2.py)
- [code/terminal_dead_consumers_v1/check_synthetic.py](code/terminal_dead_consumers_v1/check_synthetic.py)
- [code/terminal_dead_consumers_v1/estimator.py](code/terminal_dead_consumers_v1/estimator.py)
- [code/terminal_dead_consumers_v1/package_component.py](code/terminal_dead_consumers_v1/package_component.py)
- [code/terminal_dead_consumers_v1/source.py](code/terminal_dead_consumers_v1/source.py)
- [code/build_pair_polynomial_f32_candidate_v1.py](code/build_pair_polynomial_f32_candidate_v1.py)
- [code/build_response_rows_cost_v1.py](code/build_response_rows_cost_v1.py)
- [code/build_response_rows_cost_v2.py](code/build_response_rows_cost_v2.py)
- [code/build_response_rows_current_runtime_v1.py](code/build_response_rows_current_runtime_v1.py)
- [code/candidate_fullpair_runtime_d5_t8192_v1.py](code/candidate_fullpair_runtime_d5_t8192_v1.py)

<details>
<summary>Additional programs and dependencies (21)</summary>

- [code/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py](code/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py)
- [code/candidate_response_pair_polynomial_f32_v1.py](code/candidate_response_pair_polynomial_f32_v1.py)
- [code/candidate_response_rows_cost_v1.py](code/candidate_response_rows_cost_v1.py)
- [code/candidate_response_rows_cost_v2.py](code/candidate_response_rows_cost_v2.py)
- [code/candidate_response_rows_current_runtime_v1.py](code/candidate_response_rows_current_runtime_v1.py)
- [code/candidate_response_rows_parent_cost_v1.py](code/candidate_response_rows_parent_cost_v1.py)
- [code/check_pair_polynomial_f32_dev_v1.py](code/check_pair_polynomial_f32_dev_v1.py)
- [code/check_pair_polynomial_f32_panel_v1.py](code/check_pair_polynomial_f32_panel_v1.py)
- [code/check_remote_series_v1.py](code/check_remote_series_v1.py)
- [code/check_response_rows_adjusted_v1.py](code/check_response_rows_adjusted_v1.py)
- [code/check_response_rows_panel_v1.py](code/check_response_rows_panel_v1.py)
- [code/k3_radial_numpy.py](code/k3_radial_numpy.py)
- [code/pair_polynomial_f32_source_v1.py](code/pair_polynomial_f32_source_v1.py)
- [code/radial_binary_compact.py](code/radial_binary_compact.py)
- [code/radial_compact_feed_fullpair_v1.py](code/radial_compact_feed_fullpair_v1.py)
- [code/radial_compact_radialfeed_v1.py](code/radial_compact_radialfeed_v1.py)
- [code/radial_cubic.py](code/radial_cubic.py)
- [code/radial_mehler.py](code/radial_mehler.py)
- [code/radial_response_rows_v1.py](code/radial_response_rows_v1.py)
- [code/run_remote_series_v1.py](code/run_remote_series_v1.py)
- [code/terminal_dead_consumers_v1/package_stage/estimator.py](code/terminal_dead_consumers_v1/package_stage/estimator.py)

</details>

## Evidence

- [QUALIFIED_COMPONENT_2026_09_07_PDT.md#L3](QUALIFIED_COMPONENT_2026_09_07_PDT.md#L3)
- [QUALIFIED_COMPONENT_2026_09_07_PDT.md#L8](QUALIFIED_COMPONENT_2026_09_07_PDT.md#L8)
- [QUALIFIED_COMPONENT_2026_09_07_PDT.md#L22](QUALIFIED_COMPONENT_2026_09_07_PDT.md#L22)
- [QUALIFIED_COMPONENT_2026_09_07_PDT.md#L23](QUALIFIED_COMPONENT_2026_09_07_PDT.md#L23)
- [QUALIFIED_COMPONENT_2026_09_07_PDT.md#L24](QUALIFIED_COMPONENT_2026_09_07_PDT.md#L24)

<details>
<summary>All result files and records (3)</summary>

- [QUALIFIED_COMPONENT_2026_09_07_PDT.md](QUALIFIED_COMPONENT_2026_09_07_PDT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [build.json](build.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-08-removing-unused-final-layer-consumers-saves-541458188-flops-1155409213) · [Copy provenance](../../../PROVENANCE.json)

