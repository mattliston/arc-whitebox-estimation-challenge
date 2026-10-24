# Tucker truncation failed at width 1024 because consumed coordinates were not spectrally aligned

2026-08-30 · experiment · rejected · measured

TUCKER WIDTH-TRANSFER FAILS + THE READ-ALIGNMENT LAW (15:40). w1024 Tucker truncation: B=64 destroys content (fourth-cumulant Edgeworth operator (L4) rel 0.449 vs 0.058), B=256 (width-fraction matched to the w256 success) fails identically after ONE truncation (L2 rel 0.352, corr 0.936) - killed fail-fast at third-cumulant Edgeworth operator (L3) both runs. DIAGNOSIS (unifies every failed compression arm today): the consumed functional - the third-cumulant repeated-index plane (K21) plane read T[a,a,b] - is COORDINATE-ALIGNED (near-diagonal entries), while spectral compressions (HOSVD bases, Frobenius CP-ALS, read-sketch OMP subsets) are ROTATION-ALIGNED; at small basis fraction the read-relevant near-diagonal content is the first thing lost. w256's apparent success was the 25% basis fraction retaining enough coordinate alignment by accident. DESIGN REQUIREMENT (final form, for SOL and tomorrow): any bounded carrier must make the coordinate-diagonal read structure EXPLICIT in its parametrization - e.g. carriers indexed by (diagonal position, offset) rather than rotation-invariant factors, or compression objectives containing the read functional exactly, not sketched.

Recorded measurement:

    TUCKER WIDTH-TRANSFER FAILS + THE READ-ALIGNMENT LAW (15:40). w1024 Tucker truncation: B=64 destroys content (L4 rel 0.449 vs 0.058), B=256 (width-fraction matched to the w256 success) fails identically after ONE truncation (L2 rel 0.352, corr 0.936) - killed fail-fast at L3 both runs. DIAGNOSIS (unifies every failed compression arm today): the consumed functional - the K21 plane read T[a,a,b] - is COORDINATE-ALIGNED (near-diagonal entries), while spectral compressions (HOSVD bases, Frobenius CP-ALS, read-sketch OMP subsets) are ROTATION-ALIGNED; at small basis fraction the read-relevant near-diagonal content is the first thing lost. w256's apparent success was the 25% basis fraction retaining enough coordinate alignment by accident. DESIGN REQUIREMENT (final form, for SOL and tomorrow): any bounded carrier must make the coordinate-diagonal read structure EXPLICIT in its parametrization - e.g. carriers indexed by (diagonal position, offset) rather than rotation-invariant factors, or compression objectives containing the read functional exactly, not sketched. All compression arms are now closed with receipts; supply state stands at v4-offline q 0.78 (converts, unbillable) vs hub q 0.62 (billable, below threshold).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/p2_width1024_depth16/public_g19_attribution_v1/run_attribution.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_attribution.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py)
- [code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py)
- [code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py](code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py)

## Evidence

- [tensor_v4_tuck256_r35.log#L6](tensor_v4_tuck256_r35.log#L6)

<details>
<summary>All result files and records (1)</summary>

- [tensor_v4_tuck256_r35.log](tensor_v4_tuck256_r35.log) · log

</details>

[Research graph](../../../graph/index.html#node=2026-08-30-tucker-truncation-failed-width-1024-because-consumed-coordinates-were) · [Copy provenance](../../../PROVENANCE.json)

