# Spherical Stein control with one Gaussian Hessian probe: variance x cost at least 1.9374963009 x plain Monte Carlo

2026-09-07T00:46Z · finding · rejected · derived

An exactly centered spherical Stein control S(x,z) = (r^2/m) v.T Hess(u) v - x.T grad(u), computable for all outputs with a 46+31q matrix-vector directional recurrence through a smoothed near-network, was derived and then closed by a Bochner identity: with one Gaussian probe the trace-free Hessian noise retains at least rho_l/(q+rho_l) of every spherical-harmonic component l >= 2 (rho_2 = .9999980907869 at n=1024, so at least .4999995226963 of the l=2 energy), and the variance-times-cost multiplier over plain Monte Carlo has infimum (31/16) rho_2. A standard-library exact-rational certificate checked the quadratic-harmonic constants at n = 3, 4, 8, 16, 64, 1024.

Recorded measurement:

    Infimum 523775/270336 = 1.937496300899621 (Haar-orientation Rademacher 1.9337195049914349); literal depth-16 dense subtotal 46+31q matrix-vector products (77 for q=1 = 161,401,856 FLOPs); necessary nonlinear variance for adjusted 1e-9: 3.382509683738246e-05.

## Programs

- [certify_spherical_stein_constants_v1.py](certify_spherical_stein_constants_v1.py)

## Evidence

- [SPHERICAL_STEIN_ONE_POTENTIAL_VARIANCE_AUDIT_V1.md#L9](SPHERICAL_STEIN_ONE_POTENTIAL_VARIANCE_AUDIT_V1.md#L9)
- [SPHERICAL_STEIN_ONE_POTENTIAL_VARIANCE_AUDIT_V1.md#L190](SPHERICAL_STEIN_ONE_POTENTIAL_VARIANCE_AUDIT_V1.md#L190)
- [CONTINUATION_2026_09_06_1746_PDT.md#L81](../2026-09-07-even-odd-twin-input-gaussian-carrier-equals-ordinary-gaussian/CONTINUATION_2026_09_06_1746_PDT.md#L81)

<details>
<summary>All result files and records (3)</summary>

- [SPHERICAL_STEIN_ONE_POTENTIAL_VARIANCE_AUDIT_V1.md](SPHERICAL_STEIN_ONE_POTENTIAL_VARIANCE_AUDIT_V1.md) · record
- [spherical_stein_constants_certificate_v1.json](spherical_stein_constants_certificate_v1.json) · result
- [CONTINUATION_2026_09_06_1746_PDT.md](../2026-09-07-even-odd-twin-input-gaussian-carrier-equals-ordinary-gaussian/CONTINUATION_2026_09_06_1746_PDT.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-07-spherical-stein-control-one-gaussian-hessian-probe-variance-x) · [Copy provenance](../../../PROVENANCE.json)

