# Coherent state rescaling: fidelity qualified for the bounded response test

2026-09-05T21:15Z · experiment · informational · measured

Synthetic complete T/Q scaling algebra passes at widths8/16; exact-arithmetic errors<3e-16, with f32 stored-factor error<5.83e-8. Zero-action NumPy bank parity is byte-exact on35/100, including both frozen learned consumers. The separate fixed-baseline membership AND ordering certificate is byte exact at zero, has half-step discrepancy.687%, and tangent errors.304%/ .635%, passing the existing5% local gates.

Recorded measurement:

    Synthetic complete T/Q scaling algebra passes at widths8/16; exact-arithmetic errors<3e-16, with f32 stored-factor error<5.83e-8. Zero-action NumPy bank parity is byte-exact on35/100, including both frozen learned consumers. CPU zero-action maxima are1.70847e-7/2.23704e-7. Actual secants at amplitudes .02/.01 agree with independent NumPy within.399%/.358% respectively. Rank35 passes the complete original dynamic fidelity test: half-step discrepancy.827%, tangent discrepancies.306%/.773%. Rank100's original flag remains FALSE. Both positive test steps change two active memberships atL12; halving the step changes the secant by14.595%, and the zero-point tangent differs from these actual secants14.892%/29.177%. The CPU model reproduces the actual selection-changing paths accurately; it is not justified to regard that branch crossing as a smooth derivative. The separate fixed-baseline membership AND ordering certificate is byte exact at zero, has half-step discrepancy.687%, and tangent errors.304%/ .635%, passing the existing5% local gates. This qualifies a local response solve, not globally smooth dynamics or any deployment gain. All actual forwards retain dynamic selection. The failed original flag is preserved. Next bounded test is fully declared in COHERENT_SCALE_INFORMATION_V1.md: complete12-dimensional response on four fit-side networks; fixed ridge and radius; require20% predicted fit-side reduction before touching four holdouts, then20% mean and10% every holdout. Independent nonlinear checks would still be required. No automatic larger fit or submission.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/check_coherent_scale_algebra_v1.py](code/astra_phase2_reexamination_v1/check_coherent_scale_algebra_v1.py)
- [code/astra_phase2_reexamination_v1/check_coherent_scale_autodiff_v1.py](code/astra_phase2_reexamination_v1/check_coherent_scale_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/check_coherent_scale_fixed_cell_v1.py](code/astra_phase2_reexamination_v1/check_coherent_scale_fixed_cell_v1.py)
- [code/astra_phase2_reexamination_v1/coherent_central_scale_autodiff_v1.py](code/astra_phase2_reexamination_v1/coherent_central_scale_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/coherent_central_scale_v1.py](code/astra_phase2_reexamination_v1/coherent_central_scale_v1.py)
- [code/astra_phase2_reexamination_v1/coherent_scale_reference_runner_v1.py](code/astra_phase2_reexamination_v1/coherent_scale_reference_runner_v1.py)
- [code/astra_phase2_reexamination_v1/compact_final_head_v1.py](code/astra_phase2_reexamination_v1/compact_final_head_v1.py)
- [code/astra_phase2_reexamination_v1/compact_fullpair_final_reference_v1.py](code/astra_phase2_reexamination_v1/compact_fullpair_final_reference_v1.py)
- [code/astra_phase2_reexamination_v1/compact_scalar_final_v1.py](code/astra_phase2_reexamination_v1/compact_scalar_final_v1.py)
- [code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py)
- [code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py)
- [code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py](code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py)

<details>
<summary>Additional programs and dependencies (11)</summary>

- [code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py](code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/generate_coherent_scale_fidelity_v1.py](code/astra_phase2_reexamination_v1/generate_coherent_scale_fidelity_v1.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_binary_compact.py](code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [code/astra_phase2_reexamination_v1/radial_cubic.py](code/astra_phase2_reexamination_v1/radial_cubic.py)
- [code/astra_phase2_reexamination_v1/radial_features.py](code/astra_phase2_reexamination_v1/radial_features.py)
- [code/astra_phase2_reexamination_v1/radial_gnn_numpy.py](code/astra_phase2_reexamination_v1/radial_gnn_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)
- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)

</details>

## Evidence

- [COHERENT_SCALE_FIDELITY_V1_VERDICT.md#L1](COHERENT_SCALE_FIDELITY_V1_VERDICT.md#L1)
- [COHERENT_SCALE_FIDELITY_V1_VERDICT.md#L8](COHERENT_SCALE_FIDELITY_V1_VERDICT.md#L8)
- [COHERENT_SCALE_FIDELITY_V1_VERDICT.md#L16](COHERENT_SCALE_FIDELITY_V1_VERDICT.md#L16)

<details>
<summary>All result files and records (2)</summary>

- [COHERENT_SCALE_FIDELITY_V1_VERDICT.md](COHERENT_SCALE_FIDELITY_V1_VERDICT.md) · record
- [COHERENT_SCALE_INFORMATION_V1.md](COHERENT_SCALE_INFORMATION_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-05-coherent-state-rescaling-fidelity-qualified-bounded-response-test) · [Copy provenance](../../../PROVENANCE.json)

