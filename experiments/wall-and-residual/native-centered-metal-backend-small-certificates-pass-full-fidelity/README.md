# Native centered Metal backend: small certificates pass, full fidelity fails

2026-09-06T14:01Z · experiment · informational · measured

September6,2026,06:57-06:59PDT. Gaussian CDF quadrature is a fixed numerical identity with32 Gauss-Legendre nodes, no fitted parameters.4017-point CPUf64 value/gradient errors<=7.217e-16. Full MPSf32 current bank35/100 still FAILS unchanged2e-6 gate: max mean errors2.28970e-5 /2.85331e-5, finalRMS4.54110e-6 /6.35443e-6.

Recorded measurement:

    September6,2026,06:57-06:59PDT. Gaussian CDF quadrature is a fixed numerical identity with32 Gauss-Legendre nodes, no fitted parameters.4017-point CPUf64 value/gradient errors<=7.217e-16. MPSf32 CDF max6.330e-8, derivative6.368e-8, Phi(0) exactly.5. All66 scalar/four pair-map value and reverse derivative tests pass unchanged gates on CPUf64/f32 and MPSf32. No CPU fallback; native operations only. Full MPSf32 current bank35/100 still FAILS unchanged2e-6 gate: max mean errors2.28970e-5 /2.85331e-5, finalRMS4.54110e-6 /6.35443e-6. Every30 active/birth membership array matches. Time1.988s for both, process RSS885,194,752B (NOT total unified GPU memory). Output SHA933f988bbc3a2f8d1424688a2f4003ba3775b117dddceb464e02cb02e6b3bbff. Rank35 first large error emerges L5/L6; early L1-L4 much smaller than the CPUf32 BLAS version. This resembles the old MPS failure, so scalar/raw central-moment cancellation is not established as its main cause.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/centered_bank_autodiff_v1.py](code/astra_phase2_reexamination_v1/centered_bank_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/centered_bank_autodiff_v2.py](code/astra_phase2_reexamination_v1/centered_bank_autodiff_v2.py)
- [code/astra_phase2_reexamination_v1/centered_pair_numerics_v1.py](code/astra_phase2_reexamination_v1/centered_pair_numerics_v1.py)
- [code/astra_phase2_reexamination_v1/centered_relu_marginals_v1.py](code/astra_phase2_reexamination_v1/centered_relu_marginals_v1.py)
- [code/astra_phase2_reexamination_v1/centered_torch_numerics_v1.py](code/astra_phase2_reexamination_v1/centered_torch_numerics_v1.py)
- [code/astra_phase2_reexamination_v1/centered_torch_numerics_v2.py](code/astra_phase2_reexamination_v1/centered_torch_numerics_v2.py)
- [code/astra_phase2_reexamination_v1/certify_centered_torch_numerics_v1.py](code/astra_phase2_reexamination_v1/certify_centered_torch_numerics_v1.py)
- [code/astra_phase2_reexamination_v1/certify_centered_torch_numerics_v2.py](code/astra_phase2_reexamination_v1/certify_centered_torch_numerics_v2.py)
- [code/astra_phase2_reexamination_v1/certify_gaussian_cdf_quadrature_v1.py](code/astra_phase2_reexamination_v1/certify_gaussian_cdf_quadrature_v1.py)
- [code/astra_phase2_reexamination_v1/check_centered_bank_baseline_v1.py](code/astra_phase2_reexamination_v1/check_centered_bank_baseline_v1.py)
- [code/astra_phase2_reexamination_v1/check_centered_bank_baseline_v2.py](code/astra_phase2_reexamination_v1/check_centered_bank_baseline_v2.py)
- [code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py)

<details>
<summary>Additional programs and dependencies (7)</summary>

- [code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py)
- [code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py](code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py](code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/gaussian_cdf_quadrature_v1.py](code/astra_phase2_reexamination_v1/gaussian_cdf_quadrature_v1.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)
- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)

</details>

## Evidence

- [CENTERED_NATIVE_MPS_V2_VERDICT.md#L1](CENTERED_NATIVE_MPS_V2_VERDICT.md#L1)
- [CENTERED_NATIVE_MPS_V2_VERDICT.md#L5](CENTERED_NATIVE_MPS_V2_VERDICT.md#L5)
- [CENTERED_NATIVE_MPS_V2_VERDICT.md#L11](CENTERED_NATIVE_MPS_V2_VERDICT.md#L11)

<details>
<summary>All result files and records (1)</summary>

- [CENTERED_NATIVE_MPS_V2_VERDICT.md](CENTERED_NATIVE_MPS_V2_VERDICT.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-native-centered-metal-backend-small-certificates-pass-full-fidelity) · [Copy provenance](../../../PROVENANCE.json)

