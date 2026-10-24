# Uncompressed hub tensor cost corrected to about 7e12 FLOPs over 15 layers

2026-08-30 · experiment · informational · measured

CORRECTION to the hub-carrier deploy bill: uncompressed carrier is ~4.6e11 flops/LAYER (~7e12 over 15 layers, 3x OVER the 2^41 budget), not the ~1e11 class stated earlier. Deployability rides on CP-rank compression (truth distinct-content rank is 1-6 per r0b-emission-rank-not-scale; incremental warm-started ALS at rank ~24 = matmul-grade ~e9/layer). ALSO: kappa3u/kappa4u star centers (Kibble hub expansion, (k3u/6)a[p+q+3]/sd^3 shift) measured NULL at w1024 - q 0.606 vs 0.605 (r35 L13), marginals too mild.

Recorded measurement:

    CORRECTION to the hub-carrier deploy bill: uncompressed carrier is ~4.6e11 flops/LAYER (~7e12 over 15 layers, 3x OVER the 2^41 budget), not the ~1e11 class stated earlier. Deployability rides on CP-rank compression (truth distinct-content rank is 1-6 per r0b-emission-rank-not-scale; incremental warm-started ALS at rank ~24 = matmul-grade ~e9/layer). ALSO: kappa3u/kappa4u star centers (Kibble hub expansion, (k3u/6)a[p+q+3]/sd^3 shift) measured NULL at w1024 - q 0.606 vs 0.605 (r35 L13), marginals too mild. Hub supply is NOT consumption-ready: rel ~0.5 magnitude inflation makes tangent dv_beta 0.07-0.09 (10x oversized) -> consumption -2214%. Hub value = q 0.59-0.63 deep cross-net at billable ops; next lever is magnitude honesty (collision-class completion) not more sources.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/p2_width1024_depth16/public_g19_attribution_v1/hub_carrier_v0.py](code/p2_width1024_depth16/public_g19_attribution_v1/hub_carrier_v0.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_attribution.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_attribution.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v2b.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v2b.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py)
- [code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py)
- [code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py](code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py)

## Evidence

- [hub_carrier_v0.py#L1](code/p2_width1024_depth16/public_g19_attribution_v1/hub_carrier_v0.py#L1)

[Research graph](../../../graph/index.html#node=2026-08-30-uncompressed-hub-tensor-cost-corrected-about-7e12-flops-over) · [Copy provenance](../../../PROVENANCE.json)

