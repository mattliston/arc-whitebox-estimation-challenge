# Terminal sign-crossing decomposition: a boundary-only correction misses the first-order gated residual E[1[A>0] e]

2026-09-07T01:51Z · finding · rejected · derived

With a fixed affine reference A and e = a - A, ReLU(a) = ReLU(A) + 1[A>0] e + B(A,e) exactly, and the middle term is signed and first order in e even when E e = E[A e] = 0; a smooth homogeneous example and an exact bias-free ReLU example (A = X1, q = sign(X1) min(|X1|,|X2|), e = eps (q - A/2)) have no sign crossings (B = 0) yet their ReLU mean moves linearly in eps while the moment-matched Gaussian mean moves only quadratically. A curvature/sign-crossing-only terminal correction therefore cannot replace a supplier of the conditional expectation E[e | A]; the audit priced the dense affine-query path at c = 34 n^2 FLOPs per sample (residual variance below 6.168094117647e-5 needed) and launched nothing.

Recorded measurement:

    ReLU example: Var(e) = eps^2 (3/4 - 2/pi), E[e 1(A>0)] = eps phi0 (3/2 - sqrt 2), E ReLU(a) = phi0 [1 + eps (3/2 - sqrt 2)] for 0 < eps < 2; smooth example E ReLU(a) = phi0 [1 - eps (n-1)/((n+1)(n+2))].

## Evidence

- [TERMINAL_BOUNDARY_RESIDUAL_SOURCE_AUDIT_V1.md#L34](TERMINAL_BOUNDARY_RESIDUAL_SOURCE_AUDIT_V1.md#L34)
- [TERMINAL_BOUNDARY_EXACT_RELU_ADDENDUM_V1.md#L59](TERMINAL_BOUNDARY_EXACT_RELU_ADDENDUM_V1.md#L59)
- [CONTINUATION_2026_09_06_1852_PDT.md#L66](../../../experiments/learned-correction/input-gradient-gram-observer-fails-all-three-gates-pooled/CONTINUATION_2026_09_06_1852_PDT.md#L66)

<details>
<summary>All result files and records (3)</summary>

- [TERMINAL_BOUNDARY_RESIDUAL_SOURCE_AUDIT_V1.md](TERMINAL_BOUNDARY_RESIDUAL_SOURCE_AUDIT_V1.md) · record
- [TERMINAL_BOUNDARY_EXACT_RELU_ADDENDUM_V1.md](TERMINAL_BOUNDARY_EXACT_RELU_ADDENDUM_V1.md) · record
- [CONTINUATION_2026_09_06_1852_PDT.md](../../../experiments/learned-correction/input-gradient-gram-observer-fails-all-three-gates-pooled/CONTINUATION_2026_09_06_1852_PDT.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-07-terminal-sign-crossing-decomposition-boundary-only-correction-misses-first) · [Copy provenance](../../../PROVENANCE.json)

