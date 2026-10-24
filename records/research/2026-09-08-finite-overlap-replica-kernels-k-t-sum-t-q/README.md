# Finite-overlap replica kernels K(t) = sum t^q B_q bound m m^T but cannot supply the connected gate remainder Xi

2026-09-08T10:58Z · finding · rejected · derived

For Gaussian replicas with cross-covariance t I the kernel K(t) = E[h(X) h(Y_t)^T] has nonnegative Hermite coefficients B_q, giving K(0) = m m^T, K(1) - K(0) = C and the Loewner bound (K(t) - t K(1))/(1-t) <= m m^T <= K(t); but the across-depth sample Jacobian cross-product obeys E[P_l] = Q_l (W^T E[P_{l-1}] W) + Xi_l with a connected remainder Xi that exact pair-gate probabilities do not acquire, and setting it to zero recovers the known Gaussian factorization. No overlap census, node sweep or fit was launched.

Recorded measurement:

    q covariance-like trajectories with a shared first Gram cost 30q square products = 64.42450944 q GF of FP32 multiply-adds at n1024 before gates, Xi, integration and consumers; the executed fixed Dirichlet observer had source/parent raw ratio 1.000934916

## Evidence

- [OVERLAP_KERNEL_AND_TYPED_RESIDUAL_SOURCE_AUDIT_2026_09_08_V1.md#L19](OVERLAP_KERNEL_AND_TYPED_RESIDUAL_SOURCE_AUDIT_2026_09_08_V1.md#L19)
- [OVERLAP_KERNEL_AND_TYPED_RESIDUAL_SOURCE_AUDIT_2026_09_08_V1.md#L31](OVERLAP_KERNEL_AND_TYPED_RESIDUAL_SOURCE_AUDIT_2026_09_08_V1.md#L31)
- [OVERLAP_KERNEL_AND_TYPED_RESIDUAL_SOURCE_AUDIT_2026_09_08_V1.md#L46](OVERLAP_KERNEL_AND_TYPED_RESIDUAL_SOURCE_AUDIT_2026_09_08_V1.md#L46)
- [OVERLAP_KERNEL_AND_TYPED_RESIDUAL_SOURCE_AUDIT_2026_09_08_V1.md#L57](OVERLAP_KERNEL_AND_TYPED_RESIDUAL_SOURCE_AUDIT_2026_09_08_V1.md#L57)
- [OVERLAP_KERNEL_AND_TYPED_RESIDUAL_SOURCE_AUDIT_2026_09_08_V1.md#L75](OVERLAP_KERNEL_AND_TYPED_RESIDUAL_SOURCE_AUDIT_2026_09_08_V1.md#L75)

<details>
<summary>All result files and records (1)</summary>

- [OVERLAP_KERNEL_AND_TYPED_RESIDUAL_SOURCE_AUDIT_2026_09_08_V1.md](OVERLAP_KERNEL_AND_TYPED_RESIDUAL_SOURCE_AUDIT_2026_09_08_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-08-finite-overlap-replica-kernels-k-t-sum-t-q) · [Copy provenance](../../../PROVENANCE.json)

