# Whole-estimator vector-Jacobian product agrees; finite dynamic step crosses memberships

2026-09-06T14:17Z · experiment · informational · measured

All16x1024 control-gradient VJPs, fixed independent random final projection, MPSf32 versus original certified CPUf64 BankFullCard: relative2.65834453e-6 /5.39764156e-6 on35/100. Maximum per-layer relative error9.701e-6. Retrospective audit so far: original CPUf64 readout and encoder-direction negative fits are NOT invalidated by this MPS bug.

Recorded measurement:

    All16x1024 control-gradient VJPs, fixed independent random final projection, MPSf32 versus original certified CPUf64 BankFullCard: relative2.65834453e-6 /5.39764156e-6 on35/100. Maximum per-layer relative error9.701e-6. All finite; both full forward2e-6 and baseline membership gates pass. CPU reverse11-13s/net; MPS2.4-3.0s/net in this mixed worker. The composite certificate nevertheless FAILS its dynamic finite-step membership gate and remains marked false. At layer13, eps2e-4/1e-4: rank35 minus2e-4 crosses a membership; rank100 plus BOTH amplitudes crosses. Rank35 eps1e-4 remains in cell, scalarFD vsreverse relative.00016805. Other discrepancies are.00502066 rank35large, .0500772/.0487778 rank100. Do not call this global smoothness, change amplitudes/direction or hide the failed composite flag. The wrong-cell comparison is not evidence of a wrong local AD implementation; it is the already known discrete-selector issue, now explicitly observed for this late-direction probe too. A separate zero-point BRANCH certificate may freeze both memberships AND ordered indices to the previously recorded MPSv3 baseline, using the SAME eps2e-4/1e-4 and same control/projection. Require zero-point prediction byte parity with the dynamic baseline and scalarFD agreement<=5% plus half-step agreement<=5%. This explicitly certifies only the derivative autograd computes with detached selectors. No training or new authority is inferred until this test completes; old global flag stays false. Time31.898s; max processRSS7,866,630,144B; largest observed GPU driver allocation6,504,054,784B (do not add these as independent peak totals). Swap5910.50MiB unchanged before/after. Output SHA e5848f4bb3a8c674347909a7682f79ff5cb2b5b083d83556f9a66b8cca9b018e.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/centered_bank_autodiff_v1.py](code/astra_phase2_reexamination_v1/centered_bank_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/centered_bank_autodiff_v3.py](code/astra_phase2_reexamination_v1/centered_bank_autodiff_v3.py)
- [code/astra_phase2_reexamination_v1/centered_torch_numerics_v1.py](code/astra_phase2_reexamination_v1/centered_torch_numerics_v1.py)
- [code/astra_phase2_reexamination_v1/centered_torch_numerics_v3.py](code/astra_phase2_reexamination_v1/centered_torch_numerics_v3.py)
- [code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py)
- [code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py)
- [code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py](code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py](code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/gaussian_cdf_quadrature_v1.py](code/astra_phase2_reexamination_v1/gaussian_cdf_quadrature_v1.py)
- [code/astra_phase2_reexamination_v1/tensor_clamp_min_v1.py](code/astra_phase2_reexamination_v1/tensor_clamp_min_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)

## Evidence

- [CENTERED_MPS_V3_GRADIENT_V1_VERDICT.md#L1](CENTERED_MPS_V3_GRADIENT_V1_VERDICT.md#L1)
- [CENTERED_MPS_V3_GRADIENT_V1_VERDICT.md#L7](CENTERED_MPS_V3_GRADIENT_V1_VERDICT.md#L7)

<details>
<summary>All result files and records (4)</summary>

- [CENTERED_MPS_V3_GRADIENT_V1_VERDICT.md](CENTERED_MPS_V3_GRADIENT_V1_VERDICT.md) · record
- [CENTERED_MPS_NUMERICS_V3_FINAL_VERDICT.md](CENTERED_MPS_NUMERICS_V3_FINAL_VERDICT.md) · record
- [CENTERED_MPS_SAME_STEP_CPU_DIAGNOSTIC_V1.md](CENTERED_MPS_SAME_STEP_CPU_DIAGNOSTIC_V1.md) · record
- [CENTERED_MPS_V3_FULL_DERIVATIVE_DESIGN_V1.md](CENTERED_MPS_V3_FULL_DERIVATIVE_DESIGN_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-whole-estimator-vector-jacobian-product-agrees-finite-dynamic-step) · [Copy provenance](../../../PROVENANCE.json)

