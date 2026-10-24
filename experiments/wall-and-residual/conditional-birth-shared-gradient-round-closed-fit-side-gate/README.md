# Conditional birth shared-gradient round: closed at the fit-side gate

2026-09-05T21:01Z · experiment · rejected · measured

The full reverse derivatives on 20 fit-side networks are finite. The single pooled negative-gradient direction has norm-one coefficients; its mean unnormalized gradient norm is .4883184 for mean squared error (MSE) / 2e-8. Stop this fixed shared-gradient round.

Recorded measurement:

    The full reverse derivatives on 20 fit-side networks are finite. The single pooled negative-gradient direction has norm-one coefficients; its mean unnormalized gradient norm is .4883184 for MSE / 2e-8. The forward-mode pooled response quadratic is a = 2.277363720257934e-8 b = -4.883174927026877e-9 c = 4.5102798930336855e-7 predicted MSE(t) = a + 2*b*t + c*t*t. The selected step is .01082676695, well inside the declared .25 radius. Its predicted raw ratio is .99767850008: only .23215% improvement, below the predeclared 10% fit-side gate. This is a linear-response diagnostic against existing documented-1e9 local means, NOT an actual card score. Elapsed399.43s; one nice-10, 3-thread CPU process, peakRSS11.079GB; swap5966.50MiB unchanged. Immutable per-network derivative and response receipts are in conditional_birth_information_progress_v1, with the combined conditional_birth_information_v1.json. All worker jobs completed. The usable official bank remains329926 at4.98223146018e-9 adjusted.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/check_conditional_birth_information_v1.py](code/astra_phase2_reexamination_v1/check_conditional_birth_information_v1.py)
- [code/astra_phase2_reexamination_v1/conditional_birth_autodiff_v1.py](code/astra_phase2_reexamination_v1/conditional_birth_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py)
- [code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py)
- [code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py](code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py](code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)

## Evidence

- [CONDITIONAL_BIRTH_INFORMATION_V1_CLOSED.md#L1](CONDITIONAL_BIRTH_INFORMATION_V1_CLOSED.md#L1)
- [CONDITIONAL_BIRTH_INFORMATION_V1_CLOSED.md#L12](CONDITIONAL_BIRTH_INFORMATION_V1_CLOSED.md#L12)

<details>
<summary>All result files and records (3)</summary>

- [CONDITIONAL_BIRTH_INFORMATION_V1_CLOSED.md](CONDITIONAL_BIRTH_INFORMATION_V1_CLOSED.md) · record
- [conditional_birth_information_v1.json](conditional_birth_information_v1.json) · result
- [CONDITIONAL_BIRTH_INFORMATION_V1.md](CONDITIONAL_BIRTH_INFORMATION_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-05-conditional-birth-shared-gradient-round-closed-fit-side-gate) · [Copy provenance](../../../PROVENANCE.json)

