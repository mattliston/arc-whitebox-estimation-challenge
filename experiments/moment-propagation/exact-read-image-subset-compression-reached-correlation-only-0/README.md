# Exact read-image subset compression reached correlation only 0.10–0.29 at rank 96

2026-08-30 · experiment · informational · measured

EXACT-READ subset compression (closed-form read-image Grams via factor Grams, no sketching): q 0.10/0.18/0.26/0.29 (l5/9/13/14) at R=96 - better than sketched (0.08-0.17; aliasing was real) but far below v0 cap-4 (0.62): SUBSET selection has a hard ceiling ~0.3 at practical R because transported content spreads over thousands of columns. Compression campaign CLOSED with 2+ datapoints per arm: subsets (sketched 0.17, exact 0.29), Frobenius mixtures (worse with rank), Tucker (w256 partial, w1024 dead - read-alignment law), plane routing (zero), window truncation (memory-dominated). The supply road now runs through: SOL's parametrization sweep under the read-alignment constraint, the two sign-off instruments, or a design not yet conceived.

Recorded measurement:

    EXACT-READ subset compression (closed-form read-image Grams via factor Grams, no sketching): q 0.10/0.18/0.26/0.29 (l5/9/13/14) at R=96 - better than sketched (0.08-0.17; aliasing was real) but far below v0 cap-4 (0.62): SUBSET selection has a hard ceiling ~0.3 at practical R because transported content spreads over thousands of columns. Compression campaign CLOSED with 2+ datapoints per arm: subsets (sketched 0.17, exact 0.29), Frobenius mixtures (worse with rank), Tucker (w256 partial, w1024 dead - read-alignment law), plane routing (zero), window truncation (memory-dominated). The supply road now runs through: SOL's parametrization sweep under the read-alignment constraint, the two sign-off instruments, or a design not yet conceived. The chain side (deploy-gated 14.3/11.8% held-out = 2.354/2.359e-8) stands ready to convert whatever supply clears q ~0.75.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/p2_width1024_depth16/public_g19_attribution_v1/hub_carrier_v0.py](code/p2_width1024_depth16/public_g19_attribution_v1/hub_carrier_v0.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/hub_carrier_v2.py](code/p2_width1024_depth16/public_g19_attribution_v1/hub_carrier_v2.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_attribution.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_attribution.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v2b.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v2b.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py)
- [code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py)
- [code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py](code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py)

## Evidence

- [hub_v2_exact.log#L1](hub_v2_exact.log#L1)

<details>
<summary>All result files and records (1)</summary>

- [hub_v2_exact.log](hub_v2_exact.log) · log

</details>

[Research graph](../../../graph/index.html#node=2026-08-30-exact-read-image-subset-compression-reached-correlation-only-0) · [Copy provenance](../../../PROVENANCE.json)

