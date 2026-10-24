# Hybrid cubic-factor CPU/accelerator implementation passes forward and full-parameter gradient checks

2026-09-09 · tool · informational · measured

The fixed hybrid preserved more sensitive arithmetic while comparing full response-head gradients; this certified numerical execution rather than final accuracy.

Recorded measurement:

    RESULT.md:
    | Hybrid forward,12 calls including disabled controls |134.44s|6.4603GB|2.1815GB|
    | CPU full-P gradients,2 nets |72.98s|9.2499GB|0|
    | Hybrid full-P gradients,2 nets |71.65s|9.3362GB|2.2790GB|
    native d50 under changed P, or establish learnable raw-error reduction.
    complete-bill35/100/repeat35 loop. Retain all raw/native/head compatibility,
    Confirmed bank330390 remains4.53886349950395e-9 adjusted,2.120838555441651e-8
    raw. Goal stays official strictly<1e-9, with banks below4e-9,3e-9,2e-9.

## Programs

- [code/current_d50_hybrid_cp_qualification_v1/model.py](code/current_d50_hybrid_cp_qualification_v1/model.py)
- [code/current_d50_hybrid_cp_qualification_v1/qualify.py](code/current_d50_hybrid_cp_qualification_v1/qualify.py)
- [code/current_d50_hybrid_cp_qualification_v1/watch.py](code/current_d50_hybrid_cp_qualification_v1/watch.py)
- [code/birth_direction_fit_v1/build_terminal_v1/estimator.py](code/birth_direction_fit_v1/build_terminal_v1/estimator.py)
- [code/birth_direction_fit_v1/model.py](code/birth_direction_fit_v1/model.py)
- [code/centered_bank_autodiff_v1.py](code/centered_bank_autodiff_v1.py)
- [code/centered_bank_autodiff_v3.py](code/centered_bank_autodiff_v3.py)
- [code/centered_torch_numerics_v1.py](code/centered_torch_numerics_v1.py)
- [code/centered_torch_numerics_v3.py](code/centered_torch_numerics_v3.py)
- [code/current_bd_p_coadaptation_v1/model.py](code/current_bd_p_coadaptation_v1/model.py)
- [code/current_bd_p_coadaptation_v1/qualify.py](code/current_bd_p_coadaptation_v1/qualify.py)
- [code/current_bd_p_coadaptation_v1/train.py](code/current_bd_p_coadaptation_v1/train.py)

<details>
<summary>Additional programs and dependencies (8)</summary>

- [code/current_bd_p_coadaptation_v1/watch.py](code/current_bd_p_coadaptation_v1/watch.py)
- [code/current_transport_shadow_v1.py](code/current_transport_shadow_v1.py)
- [code/current_transport_shadow_v3.py](code/current_transport_shadow_v3.py)
- [code/full_carrier_autodiff_v1.py](code/full_carrier_autodiff_v1.py)
- [code/full_carrier_bank_autodiff_v1.py](code/full_carrier_bank_autodiff_v1.py)
- [code/gaussian_cdf_quadrature_v1.py](code/gaussian_cdf_quadrature_v1.py)
- [code/joint_prefix_final_data_v1.py](code/joint_prefix_final_data_v1.py)
- [code/tensor_clamp_min_v1.py](code/tensor_clamp_min_v1.py)

</details>

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L34](RESULT.md#L34)
- [RESULT.md#L35](RESULT.md#L35)
- [RESULT.md#L36](RESULT.md#L36)
- [RESULT.md#L61](RESULT.md#L61)

<details>
<summary>All result files and records (2)</summary>

- [RESULT.md](RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config

</details>

[Research graph](../../graph/index.html#node=2026-09-09-hybrid-cubic-factor-cpu-accelerator-implementation-passes-forward-full) · [Copy provenance](../../PROVENANCE.json)

