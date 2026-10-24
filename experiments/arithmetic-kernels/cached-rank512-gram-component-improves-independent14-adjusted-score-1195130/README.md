# Cached rank512 Gram component improves independent14 adjusted score by .1195130%

2026-09-08 · experiment · adopted · measured

The component reduced complete cost enough to offset a small raw-error increase and passed independent-truth and real-client qualification.

Recorded measurement:

    QUALIFIED_COMPONENT_2026_09_07_PDT.md:
    1,383,913,666F cold-minus-repeat initialization. Mean development adjusted
    ## Independent14: raw and complete cost from the same configuration
    Each net uses (MSE-mean(mc_se15^2))*max(.1,actual_completeF/2^41), without
    | public8 | 5.196969836377938e-9 | 5.191072991196909e-9 | .9988653301122219 |
    | virgin6 | 4.23000801867066e-9 | 4.2245336583437965e-9 | .9987058274351490 |
    | combined14 | 4.782557628789104e-9 | **4.7768418485455745e-9** | .9988048695515716 |
    Combined NCraw2.245315959186444e-8 ->2.2458484596872894e-8
    (-.1432208%, saved669377362.2857F). The full cost reduction outweighed
    the raw increase: adjusted improved.1195130%. Both original group gates
    passed. Candidate observed raw4.074936483190734e-8 minus mean reference
    noise1.8290880235034444e-8 gives the stated NCraw; this subtraction is not
    development acquisition. Native wall20.3724..21.1699s, peakRSS7,637,614,592B.
    Wall31.3018–33.5764s <60, residual.130858–.144507s <.4, combined sampled
    | public3(35,100,500) | 5.549931396363774e-9 | 5.54806875805055e-9 | .9996643853445748 |
    | virgin1(3016500) | 3.136717106683757e-9 | 3.135115430134888e-9 | .9994893780680904 |
    | combined4 | 4.946627823943769e-9 | 4.944830426071634e-9 | .9996366417818144 |
    The.0363358% subset gain passes both group gates but is smaller than the
    Starting from this native14 frame, if raw is unchanged and every bill
    
    CONTINUATION_2026_09_07_GRAM512_QUALIFIED_V1.md:
    Phase2 ONLY. Active final goal remains official adjusted STRICTLY<1e-9;
    Official329926 is UNCHANGED: adjusted4.982231460183304e-9,
    raw2.2448352261505987e-8, source3b7f55f9. No new submission.
    Fixed independent14, existing4M truths, unclipped per-net NCraw times OWN
    - public8: raw2.43238993902554e-8; meanF468531789024;
      adjusted5.191072991196909e-9; ratio to07b71 parent.9988653301122219.
    - virgin6: raw1.9971264872362867e-8; meanF464269469057.5;
      adjusted4.2245336583437965e-9; ratio.9987058274351490.
    - all14: raw2.2458484596872894e-8; meanF466705080466.9286;
      adjusted4.7768418485455745e-9; ratio.9988048695515716.
    Both groups pass; all complete bills lower. Raw increased.0237161%, full
    compute fell.1432208%, adjusted fell.1195130%. This is NOT an official
    score or untouched holdout. Native maxwall21.1699s/RSS7,637,614,592B,
    wall31.3018–33.5764s<60, residual.130858–.144507s<.4,
    Prospective actual-backend raw×bill check on4unique(repeat excluded)
    also passed BOTH public3/virgin1 groups. Combined adjusted
    4.946627823943769e-9 ->4.944830426071634e-9, ratio.9996366417818144.
    remote gates are wall<60, maxabs<1e-5, residual<.4, RSS<8e9.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/resident_gram512_v1/check_dev.py](code/resident_gram512_v1/check_dev.py)
- [code/resident_gram512_v1/check_panel.py](code/resident_gram512_v1/check_panel.py)
- [code/resident_gram512_v1/check_panel_v2.py](code/resident_gram512_v1/check_panel_v2.py)
- [code/resident_gram512_v1/check_remote.py](code/resident_gram512_v1/check_remote.py)
- [code/resident_gram512_v1/check_synthetic.py](code/resident_gram512_v1/check_synthetic.py)
- [code/resident_gram512_v1/estimator.py](code/resident_gram512_v1/estimator.py)
- [code/resident_gram512_v1/estimator_null.py](code/resident_gram512_v1/estimator_null.py)
- [code/resident_gram512_v1/gram.py](code/resident_gram512_v1/gram.py)
- [code/resident_gram512_v1/package_component.py](code/resident_gram512_v1/package_component.py)
- [code/resident_gram512_v1/source.py](code/resident_gram512_v1/source.py)
- [code/build_pair_polynomial_f32_candidate_v1.py](code/build_pair_polynomial_f32_candidate_v1.py)
- [code/build_response_rows_cost_v1.py](code/build_response_rows_cost_v1.py)

<details>
<summary>Additional programs and dependencies (29)</summary>

- [code/build_response_rows_cost_v2.py](code/build_response_rows_cost_v2.py)
- [code/build_response_rows_current_runtime_v1.py](code/build_response_rows_current_runtime_v1.py)
- [code/candidate_fullpair_runtime_d5_t8192_v1.py](code/candidate_fullpair_runtime_d5_t8192_v1.py)
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
- [code/resident_gram512_v1/package_stage/estimator.py](code/resident_gram512_v1/package_stage/estimator.py)
- [code/run_remote_series_v1.py](code/run_remote_series_v1.py)
- [code/terminal_dead_consumers_v1/check_panel.py](code/terminal_dead_consumers_v1/check_panel.py)
- [code/terminal_dead_consumers_v1/check_remote_v2.py](code/terminal_dead_consumers_v1/check_remote_v2.py)
- [code/terminal_dead_consumers_v1/check_synthetic.py](code/terminal_dead_consumers_v1/check_synthetic.py)
- [code/terminal_dead_consumers_v1/estimator.py](code/terminal_dead_consumers_v1/estimator.py)
- [code/terminal_dead_consumers_v1/source.py](code/terminal_dead_consumers_v1/source.py)

</details>

## Evidence

- [QUALIFIED_COMPONENT_2026_09_07_PDT.md#L3](QUALIFIED_COMPONENT_2026_09_07_PDT.md#L3)
- [QUALIFIED_COMPONENT_2026_09_07_PDT.md#L27](QUALIFIED_COMPONENT_2026_09_07_PDT.md#L27)
- [QUALIFIED_COMPONENT_2026_09_07_PDT.md#L31](QUALIFIED_COMPONENT_2026_09_07_PDT.md#L31)
- [QUALIFIED_COMPONENT_2026_09_07_PDT.md#L36](QUALIFIED_COMPONENT_2026_09_07_PDT.md#L36)
- [QUALIFIED_COMPONENT_2026_09_07_PDT.md#L43](QUALIFIED_COMPONENT_2026_09_07_PDT.md#L43)
- [CONTINUATION_2026_09_07_GRAM512_QUALIFIED_V1.md#L3](CONTINUATION_2026_09_07_GRAM512_QUALIFIED_V1.md#L3)
- [CONTINUATION_2026_09_07_GRAM512_QUALIFIED_V1.md#L4](CONTINUATION_2026_09_07_GRAM512_QUALIFIED_V1.md#L4)
- [CONTINUATION_2026_09_07_GRAM512_QUALIFIED_V1.md#L11](CONTINUATION_2026_09_07_GRAM512_QUALIFIED_V1.md#L11)
- [CONTINUATION_2026_09_07_GRAM512_QUALIFIED_V1.md#L12](CONTINUATION_2026_09_07_GRAM512_QUALIFIED_V1.md#L12)
- [CONTINUATION_2026_09_07_GRAM512_QUALIFIED_V1.md#L22](CONTINUATION_2026_09_07_GRAM512_QUALIFIED_V1.md#L22)

<details>
<summary>All result files and records (4)</summary>

- [QUALIFIED_COMPONENT_2026_09_07_PDT.md](QUALIFIED_COMPONENT_2026_09_07_PDT.md) · record
- [CONTINUATION_2026_09_07_GRAM512_QUALIFIED_V1.md](CONTINUATION_2026_09_07_GRAM512_QUALIFIED_V1.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [build.json](build.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-08-cached-rank512-gram-component-improves-independent14-adjusted-score-1195130) · [Copy provenance](../../../PROVENANCE.json)

