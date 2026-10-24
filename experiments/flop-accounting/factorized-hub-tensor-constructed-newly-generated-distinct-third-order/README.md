# Factorized hub tensor constructed for newly generated distinct third-order content

2026-08-30 · experiment · informational · measured

NEW CONSTRUCTION (hub_carrier_v0.py): the budget-class carrier for distinct third-order content. Idea: each layer's fresh V-emission is a HUB tensor sum_h c_h e_h x x_h x y_h, carried as factor matrices; the class is CLOSED under the layer push (3 matmuls) and the leading ReLU transport (Hadamard by the gate vector); plane reads are one matmul; sources decay and are dropped beyond cap 4. Per-layer cost matmul-grade (~1e11 flop class - DEPLOYABLE under the 2^41 budget); 10 s/layer offline on MPS.

Recorded measurement:

    NEW CONSTRUCTION (hub_carrier_v0.py): the budget-class carrier for distinct third-order content. Idea: each layer's fresh V-emission is a HUB tensor sum_h c_h e_h x x_h x y_h, carried as factor matrices; the class is CLOSED under the layer push (3 matmuls) and the leading ReLU transport (Hadamard by the gate vector); plane reads are one matmul; sources decay and are dropped beyond cap 4. Per-layer cost matmul-grade (~1e11 flop class - DEPLOYABLE under the 2^41 budget); 10 s/layer offline on MPS. RESULTS (rank 35): q climbs with accumulated history to 0.61 deep (L13) - beating the original full-tensor engine (0.45) with billable ops; the unaffordable v4 reference is 0.74-0.78. The j=k collision content (plane-class mass inside hub sources) was the magnitude inflator - exact first-read subtraction (2A+B enumeration) fixed rel 0.70->0.52. Enrichment paths to close 0.60->0.74: F3 correction terms (hub-closed under Hadamard with coupling matrices), richer (p,q) set, older-source retention, k211-into-4th (running separately).

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
- [hub_full3.log#L1](hub_full3.log#L1)

<details>
<summary>All result files and records (1)</summary>

- [hub_full3.log](hub_full3.log) · log

</details>

[Research graph](../../../graph/index.html#node=2026-08-30-factorized-hub-tensor-constructed-newly-generated-distinct-third-order) · [Copy provenance](../../../PROVENANCE.json)

