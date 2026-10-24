# Deployment-input gate exposed third-cumulant truth leakage in the carried state

2026-08-30 · experiment · informational · measured

THREE CLOSING MEASUREMENTS. (1) DEPLOY-RAW GATE: v4 fed the closure's own m/v/C keeps plane accuracy (L12 rel 0.0812 vs 0.0802 validation-frame - the tensor recursion tolerates closure inputs), but consumption collapses to -15.9%/-27.1% at fixed betas; held-out beta transfer recovers only +5-6% (2.5-2.6e-8). (2) ROOT CAUSE FOUND - k3u ORACLE LEAK: preK21_diag_prev was only ever assigned None, so validation-frame runs silently fed TRUE k3 into completion columns and saved supplies (the estimated tables' k3 channel +6.9/+2.0% was oracle-fed), and deploy-frame fed host k3 (dead channel, degenerate betas). FIXED: carry = diag of the pushed tensor plane. Honest deployable-class state tonight = the deploy-frame +5-6% held-out. (3) GC-COLUMNS NULL: the derived Stein mean-shift marginal correction (a_GC[t] = a[t] + (k3u/6)a[t+3]/sd^3 + (k4u/24)a[t+4]/sd^4, exact to 1st GC order inside the connected Kibble series) changes deep planes by <0.5% relative (L12 0.0806 vs 0.0802) - first-GC-order stars are ~100x too small for the k4xm deficit.

Recorded measurement:

    THREE CLOSING MEASUREMENTS. (1) DEPLOY-RAW GATE: v4 fed the closure's own m/v/C keeps plane accuracy (L12 rel 0.0812 vs 0.0802 validation-frame - the tensor recursion tolerates closure inputs), but consumption collapses to -15.9%/-27.1% at fixed betas; held-out beta transfer recovers only +5-6% (2.5-2.6e-8). (2) ROOT CAUSE FOUND - k3u ORACLE LEAK: preK21_diag_prev was only ever assigned None, so validation-frame runs silently fed TRUE k3 into completion columns and saved supplies (the estimated tables' k3 channel +6.9/+2.0% was oracle-fed), and deploy-frame fed host k3 (dead channel, degenerate betas). FIXED: carry = diag of the pushed tensor plane. Honest deployable-class state tonight = the deploy-frame +5-6% held-out. (3) GC-COLUMNS NULL: the derived Stein mean-shift marginal correction (a_GC[t] = a[t] + (k3u/6)a[t+3]/sd^3 + (k4u/24)a[t+4]/sd^4, exact to 1st GC order inside the connected Kibble series) changes deep planes by <0.5% relative (L12 0.0806 vs 0.0802) - first-GC-order stars are ~100x too small for the k4xm deficit. The mode's identity is OPEN (u1 correlates -0.82 k4 but also -0.72 v, -0.66 ones - possibly a broad-row x m-column structure from a transport-error interaction); the EMPIRICAL frozen profile correction (held-out plane gains L14 0.103->0.067) stands as the working lever. Machine-care: Matt asked for gentler operation at 02:00; one-job/nice-15 regime from then.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/p2_width1024_depth16/public_g19_attribution_v1/run_attribution.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_attribution.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py)
- [code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py)
- [code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py](code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py)

## Evidence

- [tensor_v4_gc_r35.log#L14](tensor_v4_gc_r35.log#L14)
- [consumption_dep.log#L2](consumption_dep.log#L2)

<details>
<summary>All result files and records (2)</summary>

- [tensor_v4_gc_r35.log](tensor_v4_gc_r35.log) · log
- [consumption_dep.log](consumption_dep.log) · log

</details>

[Research graph](../../../graph/index.html#node=2026-08-30-deployment-input-gate-exposed-third-cumulant-truth-leakage-carried) · [Copy provenance](../../../PROVENANCE.json)

