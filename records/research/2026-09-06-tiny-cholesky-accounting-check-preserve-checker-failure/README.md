# Tiny Cholesky accounting check: preserve the checker failure

2026-09-06T21:51Z · finding · informational · measured

Phase2, September6,2026. Native installed FlopScope0.12.0, no network arrays, candidate, truth, MPS or wide execution. This supports availability and kernel-only accounting, not whole-card native metering, deployment legality, positive-definite wide state, wall-time safety or accuracy.

Recorded measurement:

    Phase2, September6,2026. Native installed FlopScope0.12.0, no network arrays, candidate, truth, MPS or wide execution. The original source and receipt remain unchanged; no numerical rerun is needed to resolve this accounting issue. certify_gmm_cholesky_meter_v1.py successfully executed FP64 Cholesky at n16/n32. Relative reconstruction errors were2.19728e-16/2.36116e-16. Operation bills were2730/21844 FLOPs. The receipt's overall passed field is FALSE because root's checker incorrectly expected rounding within1 FLOP of 2*n^3/3; n32 differs by1.3333. This is a failed checker assertion, not a failed Cholesky or an excess computation bill. Read-only inspection of installed flopscope/numpy/linalg/_decompositions.py lines22–60 confirms the base cost is max(n**3//3,1), THEN dtype weighting is applied. Thus both observed bills exactly equal2*(n**3//3). The proposed conservative full-width FP64 envelope2*n^3/3 remains an upper bound;59 such factorizations are about42.234B FLOPs before any separately charged operations.

## Programs

- [certify_gmm_cholesky_meter_v1.py](certify_gmm_cholesky_meter_v1.py)

## Evidence

- [GMM_CHOLESKY_METER_V1_ROUNDING_NOTE.md#L1](GMM_CHOLESKY_METER_V1_ROUNDING_NOTE.md#L1)
- [GMM_CHOLESKY_METER_V1_ROUNDING_NOTE.md#L8](GMM_CHOLESKY_METER_V1_ROUNDING_NOTE.md#L8)

<details>
<summary>All result files and records (1)</summary>

- [GMM_CHOLESKY_METER_V1_ROUNDING_NOTE.md](GMM_CHOLESKY_METER_V1_ROUNDING_NOTE.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-tiny-cholesky-accounting-check-preserve-checker-failure) · [Copy provenance](../../../PROVENANCE.json)

