# Fourth-order plane-only transport missed 25–32% of the two-pair and about 95% of triple-single content

2026-09-04 · experiment · informational · measured

From truth post planes, the plane-supported push (fourth_push_1024; the same class of pushes the card's quartic star uses) reproduces the next pre-planes with fourth-cumulant two-pair plane (K22) rel 25 / 32%, fourth-cumulant triple-single plane (K31) 94 / 96%, k4 28 / 37% (L8->9 / L12->13). The missing part is uncorrelated with the pushed planes (not a rescale), 52-65% rank-8, and comes from the post tensor's distinct fourth-order classes = the born 4-path words (T-49: +121-134% of kappa4; block patterns matrix-matrix multiplication (GEMM)-class carry 33% / 49%; interleaved patterns 1024^4). The correlation matrices are not low-rank (rank 32 ~ 80% of energy) and sketches need ~10^3 probes, so the interleaved words stay 1024^4: the same wall the card's fourth-cumulant canonical-polyadic carrier (CP4) (2048 symmetric terms for a ~10^6-term pair-centered object) loses half of.

Recorded measurement:

    From truth post planes, the plane-supported push (fourth_push_1024; the same class of pushes the card's quartic star uses) reproduces the next pre-planes with K22 rel 25 / 32%, K31 94 / 96%, k4 28 / 37% (L8->9 / L12->13). The missing part is uncorrelated with the pushed planes (not a rescale), 52-65% rank-8, and comes from the post tensor's distinct fourth-order classes = the born 4-path words (T-49: +121-134% of kappa4; block patterns GEMM-class carry 33% / 49%; interleaved patterns 1024^4). The correlation matrices are not low-rank (rank 32 ~ 80% of energy) and sketches need ~10^3 probes, so the interleaved words stay 1024^4: the same wall the card's CP4 (2048 symmetric terms for a ~10^6-term pair-centered object) loses half of. Deployable planes top out near engine-K22 (emission + block push) + the card's K31: host ~3.1e-8 (r35) F2J-off vs truth planes 2.08e-8. T-83/T-84/T-85b: the GEMM-class block-pattern born-word feed (certified vs a dense sum, 2e-14) is ~1% of the missing fourth-order content at 1024 with corr 0.07-0.09 (rho^3 scaling at rho 0.05-0.1) - the born-word feed and the learned-multiplier idea are dead; CP4 rank 4096/8192 leaves the planes unchanged (0.56/0.43 of truth); the card's fourth-order planes are alpha x (v v^T + 2 C∘C) with a least-squares alpha that converges at 40-60% of truth (alpha := 1 explodes, alpha x0.5 -> 0.15). The deployable fix is the engine plane chain (K22 0.92-0.98), in-span. T-90..T-97b (cost): the CP3 arena push is 0.27 of the budget and is load-bearing (cap 4096 explodes the raw 10x with the sealed arena and with a plain push); f16 is billed as f32; the cohort spectra decay (99% energy at rank 300 -> 105 with age) but a billed low-rank bank costs more than it saves (0.84 util at the accurate schedule; birth-only truncation loses the plane). Cost floor of this architecture ~0.5 at raw ~2.9e-8.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/p2_width1024_depth16/public_g19_attribution_v1/run_attribution.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_attribution.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py)
- [code/p2_width1024_depth16/public_g29_bond_kernel_v1/fourth_split_cert.py](code/p2_width1024_depth16/public_g29_bond_kernel_v1/fourth_split_cert.py)
- [code/p2_width1024_depth16/public_g29_bond_kernel_v1/mc_post_planes_mps.py](code/p2_width1024_depth16/public_g29_bond_kernel_v1/mc_post_planes_mps.py)
- [code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py)
- [code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py](code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py)

## Evidence

- [LANE_g3_tracker.md#L702](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L702)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-fourth-order-plane-only-transport-missed-2532-two-pair) · [Copy provenance](../../../PROVENANCE.json)

