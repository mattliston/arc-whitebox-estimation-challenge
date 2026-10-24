# Fixed-count cohort retention: closed on the information prerequisite

2026-09-06T18:18Z · experiment · rejected · measured

Phase 2, September 6, 2026, 11:16:46 PDT. No target access, samples, carrier, fit, changed estimator, score or submission. Separate shape-only receipt uniform_half_cp_shape_cost_v1.json confirms the potential scale, not feasibility: uniform half per cohort rounded UP to 32 would retain 51.206% / 50.674% of pairs and save an optimistic 39.399% / 39.755% of measured first-context bank FLOPs, preserving the already-banked shared newest-pair push.

Recorded measurement:

    Phase 2, September 6, 2026, 11:16:46 PDT. No target access, samples, carrier, fit, changed estimator, score or submission. The worker finished in 3.9366s; a subsequent ps check found no remaining worker. MPS was released to root. Uniform fixed-count sampling within each birth cohort is not independent Bernoulli sampling. Its exact finite-population local output variance and the optimistic continuous allocation across cohorts were certified before the wide screen. All 120 tiny joint patterns agree with the formula to 4.40e-16 relative; the continuous optimum satisfies the interior KKT condition to 2.22e-16 on the wide cases. One coherent repaired-MPS current-bank trajectory per rank supplied BOTH pair norms and cohort sums. The older NumPy cohort sums were deliberately not combined with MPS norms. Same frozen P0/Q, complete signed pairs, unchanged nonnegative final clipping and the full diagonal-plus-rank-one final k3 Jacobian. Full-forward max differences 1.398398e-6 / 1.397164e-6 pass the unchanged 2e-6 gate, all 30 active/birth memberships agree, pair identities are 5.61e-16 / 5.25e-16, and J reverse errors are 2.34e-8 / 2.05e-8. - Half retained totals: 5968 / 5936 of 11936 / 11872 pairs. - Best CONTINUOUS cohort allocation: added MSE 2.11939241137e-8 / 2.00296372747e-8. Both FAIL the fixed <=1e-9 gate by over 20x. - Uniform half per cohort: 8.82130343779e-8 / 7.92292938489e-8. - Coherent independent uniform half: 8.82063052238e-8 / 7.92275660941e-8, reproducing the previous independent receipt. - The removable common cohort component is only 0.1515% / 0.1598% of total squared pair effects. Fixing the count therefore adds essentially no cancellation benefit. Separate shape-only receipt uniform_half_cp_shape_cost_v1.json confirms the potential scale, not feasibility: uniform half per cohort rounded UP to 32 would retain 51.206% / 50.674% of pairs and save an optimistic 39.399% / 39.755% of measured first-context bank FLOPs, preserving the already-banked shared newest-pair push. Accuracy failed before a carrier could realize that saving. Acquisition and changed non-GEMM costs are omitted; this is neither measured candidate utilization nor runtime.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/cohort_fixed_count_thinning_v1.py](code/astra_phase2_reexamination_v1/cohort_fixed_count_thinning_v1.py)
- [code/astra_phase2_reexamination_v1/centered_bank_autodiff_v1.py](code/astra_phase2_reexamination_v1/centered_bank_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/centered_bank_autodiff_v3.py](code/astra_phase2_reexamination_v1/centered_bank_autodiff_v3.py)
- [code/astra_phase2_reexamination_v1/centered_torch_numerics_v1.py](code/astra_phase2_reexamination_v1/centered_torch_numerics_v1.py)
- [code/astra_phase2_reexamination_v1/centered_torch_numerics_v3.py](code/astra_phase2_reexamination_v1/centered_torch_numerics_v3.py)
- [code/astra_phase2_reexamination_v1/certify_pair_jet_coalescence_v1.py](code/astra_phase2_reexamination_v1/certify_pair_jet_coalescence_v1.py)
- [code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py)
- [code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py)
- [code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py](code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py](code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/gaussian_cdf_quadrature_v1.py](code/astra_phase2_reexamination_v1/gaussian_cdf_quadrature_v1.py)
- [code/astra_phase2_reexamination_v1/paired_cp_thinning_bound_v1.py](code/astra_phase2_reexamination_v1/paired_cp_thinning_bound_v1.py)

<details>
<summary>Additional programs and dependencies (3)</summary>

- [code/astra_phase2_reexamination_v1/price_uniform_half_cp_shape_v1.py](code/astra_phase2_reexamination_v1/price_uniform_half_cp_shape_v1.py)
- [code/astra_phase2_reexamination_v1/tensor_clamp_min_v1.py](code/astra_phase2_reexamination_v1/tensor_clamp_min_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)

</details>

## Evidence

- [COHORT_FIXED_COUNT_THINNING_V1_CLOSED.md#L1](COHORT_FIXED_COUNT_THINNING_V1_CLOSED.md#L1)
- [COHORT_FIXED_COUNT_THINNING_V1_CLOSED.md#L4](COHORT_FIXED_COUNT_THINNING_V1_CLOSED.md#L4)
- [COHORT_FIXED_COUNT_THINNING_V1_CLOSED.md#L12](COHORT_FIXED_COUNT_THINNING_V1_CLOSED.md#L12)
- [COHORT_FIXED_COUNT_THINNING_V1_CLOSED.md#L21](COHORT_FIXED_COUNT_THINNING_V1_CLOSED.md#L21)

<details>
<summary>All result files and records (3)</summary>

- [COHORT_FIXED_COUNT_THINNING_V1_CLOSED.md](COHORT_FIXED_COUNT_THINNING_V1_CLOSED.md) · record
- [uniform_half_cp_shape_cost_v1.json](uniform_half_cp_shape_cost_v1.json) · result
- [COHORT_FIXED_COUNT_THINNING_DESIGN_V1.md](COHORT_FIXED_COUNT_THINNING_DESIGN_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-fixed-count-cohort-retention-closed-information-prerequisite) · [Copy provenance](../../../PROVENANCE.json)

