# Homogeneous boundary (Ward) identity: a gradient-energy Gaussian scalar closure returns the Gaussian ReLU mean exactly

2026-09-07 · finding · rejected · derived

For a degree-one homogeneous prefix a(X), Gaussian integration by parts gives E b = mu and c + E t = v with b = Laplacian a, t = |grad a|^2 and c = E[(a-mu) b]; the boundary form E relu(a) = E[1[a>0] b] + p(0) E[t | a=0] was then closed with an affine regression for beta(s) and a constant conditional energy, and the two terms summed to mu Phi(alpha) + sigma phi(alpha) for every value of the supplied gradient energy. Propagating a per-neuron gradient energy therefore cannot improve the mean under that closure; the addendum supplied a regular nonlinear example a(x) = x1 + eps r with finite boundary density where tau(0) = 1-eps^2 differs from tbar = 1+eps^2, showing the conditional field is the missing information.

Recorded measurement:

    Toy a = (x1^2-x2^2)/|x|: v = 4/(n+2), E t = 4(2n+1)/[n(n+2)], E[a b] = -4(n+1)/[n(n+2)]; E relu(a) = 2 E[chi_n]/(pi n) vs scalar closure 2/[sqrt(n+2) sqrt(2 pi)].

## Evidence

- [HOMOGENEOUS_BOUNDARY_MEAN_WARD_AUDIT_V1.md#L15](HOMOGENEOUS_BOUNDARY_MEAN_WARD_AUDIT_V1.md#L15)
- [HOMOGENEOUS_BOUNDARY_MEAN_WARD_AUDIT_V1.md#L115](HOMOGENEOUS_BOUNDARY_MEAN_WARD_AUDIT_V1.md#L115)
- [HOMOGENEOUS_BOUNDARY_REGULAR_TOY_ADDENDUM_V1.md#L114](HOMOGENEOUS_BOUNDARY_REGULAR_TOY_ADDENDUM_V1.md#L114)

<details>
<summary>All result files and records (2)</summary>

- [HOMOGENEOUS_BOUNDARY_MEAN_WARD_AUDIT_V1.md](HOMOGENEOUS_BOUNDARY_MEAN_WARD_AUDIT_V1.md) · record
- [HOMOGENEOUS_BOUNDARY_REGULAR_TOY_ADDENDUM_V1.md](HOMOGENEOUS_BOUNDARY_REGULAR_TOY_ADDENDUM_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-07-homogeneous-boundary-ward-identity-gradient-energy-gaussian-scalar-closure) · [Copy provenance](../../../PROVENANCE.json)

