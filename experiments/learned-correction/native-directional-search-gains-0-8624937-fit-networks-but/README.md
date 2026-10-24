# Native directional search gains 0.8624937% on fit networks but worsens checks by 2.0150121%

2026-09-09 · experiment · rejected · measured

A fixed native objective search scored its fitted direction on disjoint research subsets with identical complete-cost ratios.

Recorded measurement:

    RESULT.md:
    The fixed candidate improved FIT4 adjusted by0.8624937% but worsened
    CHECK4 by2.0150121%. FIT won2/4; CHECK won0/4. Both required<=.98 adjusted
    ratio and>=3/4 wins, so the scientific advancement gate FAILED. There is
    raw*max(.1,F/2^41). FIT=(35,36,38,40), CHECK=(41,43,44,45), disjoint only
    |FIT baseline|4.3342444831004895e-9|2.072392389825825e-8|461672048141.5|
    |FIT candidate|4.296861895672558e-9|2.0530405449470482e-8|461672048141.5|
    |CHECK baseline|4.461207741162884e-9|2.110571155718128e-8|465702575514.5|
    |CHECK candidate|4.5511016177955235e-9|2.152763653224428e-8|465702575514.5|
    Raw ratios: FIT0.9906620749169981,CHECK1.0199910329447972. Complete cost
    ratios1.0 on both batches. The candidate does not save compute; no old raw
    The measured three FIT points had adjusted J0=4.3342444831004895e-9,
    wall19.9737002505s. Thus isolated acquisition passed its resource contract;

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/current_d50_native_cold_search_v1/common.py](code/current_d50_native_cold_search_v1/common.py)
- [code/current_d50_native_cold_search_v1/cpu.py](code/current_d50_native_cold_search_v1/cpu.py)
- [code/current_d50_native_cold_search_v1/interpolate.py](code/current_d50_native_cold_search_v1/interpolate.py)
- [code/current_d50_native_cold_search_v1/judge.py](code/current_d50_native_cold_search_v1/judge.py)
- [code/current_d50_native_cold_search_v1/native.py](code/current_d50_native_cold_search_v1/native.py)
- [code/current_d50_native_cold_search_v1/watch.py](code/current_d50_native_cold_search_v1/watch.py)
- [code/birth_direction_fit_v1/build_terminal_v1/estimator.py](code/birth_direction_fit_v1/build_terminal_v1/estimator.py)
- [code/birth_direction_fit_v1/fnp/build.py](code/birth_direction_fit_v1/fnp/build.py)
- [code/birth_direction_fit_v1/fnp/build_v2.py](code/birth_direction_fit_v1/fnp/build_v2.py)
- [code/birth_direction_fit_v1/fnp/core.py](code/birth_direction_fit_v1/fnp/core.py)
- [code/birth_direction_fit_v1/fnp/core_v2.py](code/birth_direction_fit_v1/fnp/core_v2.py)
- [code/birth_direction_fit_v1/fnp/reference_numpy.py](code/birth_direction_fit_v1/fnp/reference_numpy.py)

<details>
<summary>Additional programs and dependencies (27)</summary>

- [code/birth_direction_fit_v1/model.py](code/birth_direction_fit_v1/model.py)
- [code/birth_direction_fit_v1/native.py](code/birth_direction_fit_v1/native.py)
- [code/birth_direction_fit_v1/native_v2.py](code/birth_direction_fit_v1/native_v2.py)
- [code/centered_bank_autodiff_v1.py](code/centered_bank_autodiff_v1.py)
- [code/centered_bank_autodiff_v3.py](code/centered_bank_autodiff_v3.py)
- [code/centered_torch_numerics_v1.py](code/centered_torch_numerics_v1.py)
- [code/centered_torch_numerics_v3.py](code/centered_torch_numerics_v3.py)
- [code/current_bd_p_coadaptation_v1/model.py](code/current_bd_p_coadaptation_v1/model.py)
- [code/current_bd_p_coadaptation_v1/qualify.py](code/current_bd_p_coadaptation_v1/qualify.py)
- [code/current_bd_p_coadaptation_v1/train.py](code/current_bd_p_coadaptation_v1/train.py)
- [code/current_bd_p_coadaptation_v1/watch.py](code/current_bd_p_coadaptation_v1/watch.py)
- [code/current_d50_cpu_pilot_v1/worker.py](code/current_d50_cpu_pilot_v1/worker.py)
- [code/current_d50_hybrid_cp_qualification_v1/model.py](code/current_d50_hybrid_cp_qualification_v1/model.py)
- [code/current_d50_native_directional_search_v1/common.py](code/current_d50_native_directional_search_v1/common.py)
- [code/current_d50_native_directional_search_v1/cpu.py](code/current_d50_native_directional_search_v1/cpu.py)
- [code/current_d50_native_directional_search_v1/interpolate.py](code/current_d50_native_directional_search_v1/interpolate.py)
- [code/current_d50_native_directional_search_v1/judge.py](code/current_d50_native_directional_search_v1/judge.py)
- [code/current_d50_native_directional_search_v1/native.py](code/current_d50_native_directional_search_v1/native.py)
- [code/current_d50_native_directional_search_v1/watch.py](code/current_d50_native_directional_search_v1/watch.py)
- [code/current_transport_shadow_v1.py](code/current_transport_shadow_v1.py)
- [code/current_transport_shadow_v3.py](code/current_transport_shadow_v3.py)
- [code/full_carrier_autodiff_v1.py](code/full_carrier_autodiff_v1.py)
- [code/full_carrier_bank_autodiff_v1.py](code/full_carrier_bank_autodiff_v1.py)
- [code/gaussian_cdf_quadrature_v1.py](code/gaussian_cdf_quadrature_v1.py)
- [code/joint_prefix_final_data_v1.py](code/joint_prefix_final_data_v1.py)
- [code/tensor_clamp_min_v1.py](code/tensor_clamp_min_v1.py)
- [code/unused_materializations_v1/estimator.py](code/unused_materializations_v1/estimator.py)

</details>

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L10](RESULT.md#L10)
- [RESULT.md#L11](RESULT.md#L11)
- [RESULT.md#L12](RESULT.md#L12)
- [RESULT.md#L18](RESULT.md#L18)

<details>
<summary>All result files and records (2)</summary>

- [RESULT.md](RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config

</details>

[Research graph](../../../graph/index.html#node=2026-09-09-native-directional-search-gains-0-8624937-fit-networks-but) · [Copy provenance](../../../PROVENANCE.json)

