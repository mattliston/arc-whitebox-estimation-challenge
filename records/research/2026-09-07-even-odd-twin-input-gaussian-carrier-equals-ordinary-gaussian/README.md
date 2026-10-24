# Even/odd twin-input Gaussian carrier equals ordinary Gaussian propagation: 58 vs 30 square products, no new information

2026-09-07T00:46Z · finding · rejected · derived

A source-only audit proved that a jointly Gaussian even/odd (antithetic) twin-input carrier with state (m_e, C_e, C_o) is algebraically the ordinary Gaussian mean/covariance recursion: the plus preactivation is Gaussian with S = W.T (C_e+C_o) W, so single-branch means are identical at every depth while the extra cross covariance T = W.T (C_e-C_o) W costs two extra square products per transition. It also showed that exact input orthogonal symmetry is already quotiented by the bank (the first preactivation law depends on W0 only through G = W0.T W0) and that an absorbable hidden basis must be a positive monomial map, so no dense rotation can lower the tensor dimension. The mixed even/odd third moments E[(e_i-E e_i) o_j o_k] = sqrt(2/pi) G_ij G_ik / (8 sqrt(G_ii)) survive after layer one.

Recorded measurement:

    Twin schedule 58 square GEMM equivalents vs 30 for ordinary Gaussian propagation at n=1024: 124,554,051,584 vs 64,424,509,440 conventional FP32 FLOPs before maps.

## Evidence

- [GAUSSIAN_RELU_SYMMETRY_SOURCE_AUDIT_V1.md#L13](GAUSSIAN_RELU_SYMMETRY_SOURCE_AUDIT_V1.md#L13)
- [GAUSSIAN_RELU_SYMMETRY_SOURCE_AUDIT_V1.md#L128](GAUSSIAN_RELU_SYMMETRY_SOURCE_AUDIT_V1.md#L128)
- [CONTINUATION_2026_09_06_1746_PDT.md#L26](CONTINUATION_2026_09_06_1746_PDT.md#L26)

<details>
<summary>All result files and records (2)</summary>

- [GAUSSIAN_RELU_SYMMETRY_SOURCE_AUDIT_V1.md](GAUSSIAN_RELU_SYMMETRY_SOURCE_AUDIT_V1.md) · record
- [CONTINUATION_2026_09_06_1746_PDT.md](CONTINUATION_2026_09_06_1746_PDT.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-07-even-odd-twin-input-gaussian-carrier-equals-ordinary-gaussian) · [Copy provenance](../../../PROVENANCE.json)

