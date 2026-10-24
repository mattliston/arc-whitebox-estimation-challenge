# Collective-skewness correction inside recursion reduced deep plane error

2026-08-30 · experiment · informational · measured

IN-RECURSION COLLECTIVE FIX = THE CAMPAIGN'S LARGEST SUPPLY GAIN (20:30). Post-hoc plane patching failed (-10.1/10.1% held-out vs 14.5/11.7 baseline) exactly per the co-adaptation basin law (inconsistent supply state: pre/k3u corrected, fourth-cumulant two-pair plane (K22)/fourth-cumulant triple-single plane (K31)/k4u not). The JOINT version (K3FIX=1: Tt += (K3S_l - implied_l) w x w x w inside the recursion at l>=5, all downstream objects inherit coherently, correction transports forward so deltas self-consistently shrink): deep planes L9 0.0620->0.0558, L12 0.0777->0.0644, L13 0.0826->0.0687, L14 0.0901->0.0711 (17-21% relative), corr 0.9982-0.9988.

Recorded measurement:

    IN-RECURSION COLLECTIVE FIX = THE CAMPAIGN'S LARGEST SUPPLY GAIN (20:30). Post-hoc plane patching failed (-10.1/10.1% held-out vs 14.5/11.7 baseline) exactly per the co-adaptation basin law (inconsistent supply state: pre/k3u corrected, K22/K31/k4u not). The JOINT version (K3FIX=1: Tt += (K3S_l - implied_l) w x w x w inside the recursion at l>=5, all downstream objects inherit coherently, correction transports forward so deltas self-consistently shrink): deep planes L9 0.0620->0.0558, L12 0.0777->0.0644, L13 0.0826->0.0687, L14 0.0901->0.0711 (17-21% relative), corr 0.9982-0.9988. Cost of the fix ~1e5 flops/layer + one rank-1 tensor add (~1e9 offline; the deploy form needs no tensor - the correction enters the carried planes/univariates directly). Full held-out chain running (r100 + tangents + consumption + transfer).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/p2_width1024_depth16/public_g19_attribution_v1/run_attribution.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_attribution.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py)
- [code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py)
- [code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py](code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py)

## Evidence

- [tensor_v4_k3fix_r35.log#L16](tensor_v4_k3fix_r35.log#L16)

<details>
<summary>All result files and records (1)</summary>

- [tensor_v4_k3fix_r35.log](tensor_v4_k3fix_r35.log) · log

</details>

[Research graph](../../../graph/index.html#node=2026-08-30-collective-skewness-correction-inside-recursion-reduced-deep-plane-error) · [Copy provenance](../../../PROVENANCE.json)

