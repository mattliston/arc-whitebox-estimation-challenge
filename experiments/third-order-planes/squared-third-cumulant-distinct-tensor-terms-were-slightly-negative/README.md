# Squared-third-cumulant distinct-tensor terms were slightly negative at width 1024

2026-08-30 · experiment · informational · measured

KAPPA3^2 STEP 2 (distinct-tensor pattern pairs, exact coefficients, broadcast form crosschecked to 1e-12 against the GH-validated scalar reference): WIDTH-DEPENDENT VERDICT. w256 gate: deep tensor IMPROVES (K21_rel l31 0.500->0.474, l24 0.456->0.433; shallow slightly worse at rho_absmax 0.75 = bond_cap-3 truncation; free-run FINAL 2.41e-5->2.50e-5 = the known co-adaptation pattern of that frame). w1024 (validation frame, honest carry): the distinct half is slightly NEGATIVE - L12 0.0777 (plane-half only) -> 0.0791, mode coef -0.2177 -> -0.2209 - consistent with kappa3^2 fresh emission scaling ~1/width (4x smaller relative at 1024). CONCLUSION: the w1024 deep k3^2/v-row deficit is NOT primarily fresh-emission kappa3^2; candidate carriers sharing that row direction: F3 transport error of carried distinct content (weights concentrate on the most non-Gaussian rows), or higher-bond emission truncation. Validated w1024 config = SECOND_ORDER=1 (plane half) only: deep planes 0.0802->0.0777/0.0935->0.0901.

Recorded measurement:

    KAPPA3^2 STEP 2 (distinct-tensor pattern pairs, exact coefficients, broadcast form crosschecked to 1e-12 against the GH-validated scalar reference): WIDTH-DEPENDENT VERDICT. w256 gate: deep tensor IMPROVES (K21_rel l31 0.500->0.474, l24 0.456->0.433; shallow slightly worse at rho_absmax 0.75 = bond_cap-3 truncation; free-run FINAL 2.41e-5->2.50e-5 = the known co-adaptation pattern of that frame). w1024 (validation frame, honest carry): the distinct half is slightly NEGATIVE - L12 0.0777 (plane-half only) -> 0.0791, mode coef -0.2177 -> -0.2209 - consistent with kappa3^2 fresh emission scaling ~1/width (4x smaller relative at 1024). CONCLUSION: the w1024 deep k3^2/v-row deficit is NOT primarily fresh-emission kappa3^2; candidate carriers sharing that row direction: F3 transport error of carried distinct content (weights concentrate on the most non-Gaussian rows), or higher-bond emission truncation. Validated w1024 config = SECOND_ORDER=1 (plane half) only: deep planes 0.0802->0.0777/0.0935->0.0901. Final honest chain on that supply queued.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/p2_width1024_depth16/public_g19_attribution_v1/run_attribution.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_attribution.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py)
- [code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py)
- [code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py](code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py)

## Evidence

- [tensor_v4_so2full_r35.log#L14](tensor_v4_so2full_r35.log#L14)

<details>
<summary>All result files and records (1)</summary>

- [tensor_v4_so2full_r35.log](tensor_v4_so2full_r35.log) · log

</details>

[Research graph](../../../graph/index.html#node=2026-08-30-squared-third-cumulant-distinct-tensor-terms-were-slightly-negative) · [Copy provenance](../../../PROVENANCE.json)

