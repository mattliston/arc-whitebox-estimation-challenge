# Full-matrix fourth-cumulant modulation modes sum_r lambda_r S(V_r): 64.42450944B transport per mode, no tested source

2026-09-07 · finding · informational · derived

A prior-work scope audit of the tensor form T4 = sum_r lambda_r S(V_r), S(V)_ijkl = V_ij V_kl + V_ik V_jl + V_il V_jk, showed the archive's radial (alpha S(C)/3), transported Sym(C,D) = [S(C+A)-S(C-A)]/12, SSC fixed-mode and GMM4 work are exact subclasses, but none tested a carrier of independently born full symmetric modes with a nonlinear reset and a complete fourth-to-third source. Each full mode costs two dense products per transition (4 n^3), and the missing piece is the ReLU source of the modes and their odd descendants, not the affine push.

Recorded measurement:

    64.42450944B per mode over 15 width-1024 transitions; prior anisotropic fourth source: raw35 2.4910582e-8 vs 2.4641269e-8 and raw100 2.8609074e-8 vs 2.6988241e-8 (+1.09%/+6.01%, failing the 15% gate); fourth-trace oracle next-mean cross-energy ratios .6112197142/.6108077062 (fail .50); SSC nine-mode held-out R2 .60794/.63687/.68075/.69298 at post 4/8/12/14.

## Evidence

- [FULL_MATRIX_FOURTH_MODULATION_PRIOR_SCOPE_V1.md#L9](FULL_MATRIX_FOURTH_MODULATION_PRIOR_SCOPE_V1.md#L9)
- [FULL_MATRIX_FOURTH_MODULATION_PRIOR_SCOPE_V1.md#L47](FULL_MATRIX_FOURTH_MODULATION_PRIOR_SCOPE_V1.md#L47)
- [FULL_MATRIX_FOURTH_MODULATION_PRIOR_SCOPE_V1.md#L90](FULL_MATRIX_FOURTH_MODULATION_PRIOR_SCOPE_V1.md#L90)

<details>
<summary>All result files and records (1)</summary>

- [FULL_MATRIX_FOURTH_MODULATION_PRIOR_SCOPE_V1.md](FULL_MATRIX_FOURTH_MODULATION_PRIOR_SCOPE_V1.md) · record

</details>

## Related work

- ← supersedes: [One Gaussian fourth-cumulant mode after layer 12 fails the fixed .5 next-mean response gate](../../../experiments/fourth-order-terms/one-gaussian-fourth-cumulant-mode-after-layer-12-fails/README.md)

[Research graph](../../../graph/index.html#node=2026-09-07-full-matrix-fourth-cumulant-modulation-modes-sum-r-lambda) · [Copy provenance](../../../PROVENANCE.json)

