# Sampled local Gaussian three-residual third-cumulant source: 512-pair variance 4.5806e-10 fails the 1e-10 gate

2026-09-19 · experiment · rejected · measured

A research-only constructor samples Z ~ N(0, Cpre) at the sealed rank-35 layer-8 packet, forms the nonlinear residual R = alive * (ReLU(mu+Z) - E ReLU(mu+Z) - Phi(mu/sigma) Z) and the all-distinct third-ray contribution (qR)^3 - 3(qR)(q^2R^2) + 2(q^3R^3) weighted by the frozen consumer's third-cumulant mean derivative. The captured covariance failed an ordinary Cholesky (33 negative eigenvalues carrying 2.9423e-13 of the energy), and a positive active-only reference gave next-mean variances about five times the declared 1e-10 gate for two antithetic 512-pair batches, so the sampled constructor was closed.

Recorded measurement:

    negative energy fraction 2.9423e-13; active reference differs from captured C by relative RMS 9.053e-5; variances 4.5806e-10 and 4.8625e-10 vs gate 1e-10; independent mean cross-product -1.8020e-11; optimistic matrix-only acquisition/read bill 9.266BF per cell

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/gaussian_rrr_precision.py](code/gaussian_rrr_precision.py)
- [code/gaussian_rrr_active_precision.py](code/gaussian_rrr_active_precision.py)
- [code/gaussian_domain_diagnostic.py](code/gaussian_domain_diagnostic.py)

## Evidence

- [RECEIPTS.md#L217](../../birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md#L217)
- [RECEIPTS.md#L223](../../birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md#L223)
- [RECEIPTS.md#L224](../../birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md#L224)
- [GAUSSIAN_RRR_PRECISION_PROTOCOL.md#L9](GAUSSIAN_RRR_PRECISION_PROTOCOL.md#L9)
- [GAUSSIAN_RRR_PRECISION_PROTOCOL.md#L26](GAUSSIAN_RRR_PRECISION_PROTOCOL.md#L26)

<details>
<summary>All result files and records (6)</summary>

- [GAUSSIAN_RRR_PRECISION_PROTOCOL.md](GAUSSIAN_RRR_PRECISION_PROTOCOL.md) · record
- [GAUSSIAN_RRR_ACTIVE_PROTOCOL.md](GAUSSIAN_RRR_ACTIVE_PROTOCOL.md) · record
- [gaussian_rrr_precision.json](gaussian_rrr_precision.json) · result
- [gaussian_rrr_active_precision.json](gaussian_rrr_active_precision.json) · result
- [gaussian_domain_diagnostic.json](gaussian_domain_diagnostic.json) · result
- [RECEIPTS.md](../../birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-19-sampled-local-gaussian-three-residual-third-cumulant-source-512) · [Copy provenance](../../../PROVENANCE.json)

