# Self-carried fourth-order planes integrated into the width-1024 third-cumulant tensor recursion

2026-08-29 · experiment · informational · measured

PHASE-2 (1024x16) v4 ENGINE: the fourth-order machinery (self-carried k4/fourth-cumulant two-pair plane (K22)/fourth-cumulant triple-single plane (K31) planes via fourth_emission + fourth_push + k211 semi-distinct completion) ported from the day's derivations and wired into the w1024 tensor engine. RESULTS (rank 35, true mean/cov frame): deep third-cumulant repeated-index plane (K21) plane error 8.0 pct at L12 vs closure 12.5 (v2b was 11.9); carried K22 27 pct vs closure 55. SUPPLY QUALITY q = corr(delta vs true delta): L8 0.53->0.79, L12 0.47->0.78, L14 0.40->0.74 - nearly doubled deep.

Recorded measurement:

    PHASE-2 (1024x16) v4 ENGINE: the fourth-order machinery (self-carried k4/K22/K31 planes via fourth_emission + fourth_push + k211 semi-distinct completion) ported from the day's derivations and wired into the w1024 tensor engine. RESULTS (rank 35, true mean/cov frame): deep K21 plane error 8.0 pct at L12 vs closure 12.5 (v2b was 11.9); carried K22 27 pct vs closure 55. SUPPLY QUALITY q = corr(delta vs true delta): L8 0.53->0.79, L12 0.47->0.78, L14 0.40->0.74 - nearly doubled deep. CHAIN CONVERSION (run_tangent_v2 tensorK21): dv_corr 0.81/0.82/0.83 at L12/14/15 (was 0.42/0.39/0.26) = the truth-fed chain ceiling - supply is no longer the chain's bottleneck. Endpoint arithmetic: v-channel capture ~0.69 of the 1.73e-8 pool -> panel est ~2.0e-8 from 3.197e-8 before k3/k4 channels. DEPLOYMENT CONSTRAINT NOTED: the full-tensor 3rd-order push is ~6.6e12 flops/layer (45x budget) - NOT billable; the 4th-order machinery IS matmul-grade (~e10/layer, under budget). Decisive next: v5-deployable (planes-only, no tensor: V-completion 3rd-order push + full-channel emission + carried 4th) - if its q lands near v4's, a billable supply exists and the submission pipeline opens.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/p2_width1024_depth16/public_g19_attribution_v1/run_attribution.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_attribution.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_q.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_q.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py)
- [code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py)
- [code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py](code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py)

## Evidence

- [TENSOR_Q_rank0035.json#L1](TENSOR_Q_rank0035.json#L1)

<details>
<summary>All result files and records (1)</summary>

- [TENSOR_Q_rank0035.json](TENSOR_Q_rank0035.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-29-self-carried-fourth-order-planes-integrated-into-width-1024) · [Copy provenance](../../../PROVENANCE.json)

