# Independent post-q V/K3/K4 fallback cross-audit

Date: 2026-08-31

Status: **STATIC DERIVATION ONLY.**  This audit opened no target or derived
feature cache, ran no training/scoring/estimator, and grants no execution,
integration, packaging, or submission authority.  It does not modify or
supersede the frozen `DESIGN.md` / `DESIGN_FREEZE.json` bytes.

## 1. Why a physical-state fallback is still a distinct question

The ten-network frozen perfect-state receipt gives noise-corrected mean MSE

```text
deployed              3.197225394770670e-8
exact v               1.465583780991439e-8
exact v+k3            1.028608386484068e-8
exact v+k3+k4         9.254687418209847e-9.
```

Thus the sequential values are `17.316e-9`, `4.36975e-9`, and
`1.031396e-9`.  Variance alone cannot reach `1e-8`; even exact variance plus
cubic misses by `2.86084e-10`.  Conditional on exact v and k3, the quartic
channel must recover at least

```text
(1.028608386484068e-8 - 1e-8) / 1.031396e-9 = 0.2773
```

of its sequential value.  A useful fallback therefore has to emit all three
physical gap fields, not just another cubic feature.

The direct start-5 exact-q oracle passed with MSE
`4.698527912579728e-10`, but its first legal target-free provider failed:
source-on was worse than its paired source-zero control.  The subsequent
owned four-field q provider also failed its one-shot gate: its frozen
parent-to-oracle gap recovery was only `0.0009354760270741128` versus the
required `0.08`, despite beating source-zero locally and at the endpoint.
Those results establish abundant *action capacity* but failed q-provider
observability.  They do not measure the separately constrained map from D11
physical provider decompositions to `(delta v, delta k3, delta k4)`.

## 2. Binding interlock correction

The existing freeze says this branch is reachable only if the exact start-5
q oracle is **not** below `1e-8`.  That premise is false: the oracle passed.
Therefore the frozen v1 design is unreachable and must not be executed.

A lawful post-q successor must be a new v2 freeze.  Its activation premise is
not oracle failure; it is:

1. start-5 exact-q capacity passed;
2. the direct transpose provider closed at result
   `e1ed27d0882dddb09698da78d167f7e34096ce44742b210b64792a3d20babd7f`;
3. the owned L3/L3-squared four-field provider closed at result
   `d84c5831d3392961d04fcf23323794ca75858190924672aa21c4e496731b98f5`;
4. no other already-preregistered q successor remains live; and
5. a separate exact-byte release is issued.

This is not a semantic loophole.  The oracle says a true q source can solve
the suffix; the two terminal results say the tested target-blind sources did
not reveal it.  The physical route starts at layer 1, is supervised only on
moment-state gaps, and restricts its action to the analytic
`F(m,v,k3,k4)` substitutions used by the perfect-state receipt.

## 3. Minimum target-free provider split

Write the three D11 physical provider fibers per unit as

```text
authoritative variance  versus pair-Q24 diagonal variance,
CP3 cubic               versus star/repeated-index cubic,
CP4 quartic             versus star/pairing quartic.
```

At fixed provider total, each fiber has one independent scalar direction.
Suppose an inference interface adds fewer than three scalar coordinates per
unit.  Its differential restricted to the direct sum of these three fibers
has rank at most two, hence a nonzero provider redistribution lies in its
kernel.  Two laws can then have identical inference features and different
physical gap labels in that direction.  Therefore one scalar split coordinate
per degree is information-minimal for this exact V/K3/K4 mechanism.  A free
scalar q source is not a counterexample: it abandons the physical mechanism
being tested.

The concrete minimal split is:

```text
e2_j = raw2_uni,j - diag(M11_pair,j)                  # before overwrite
uV_l = (W_l**2).T @ e2_(l-1) / sigma_l**2
u3_l = diag(cp_pre_k21_l) / sigma_l**3
u4_l = cp4_pre_provider_planes[0] / sigma_l**4.
```

`u3` and `u4` are views/reductions of already-live D11 operands.  `e2` must be
captured before `_copy_with_diagonal`; `W**2` is already constructed by D11.
No new dense plane is required.

There is one interface-counting nuance.  Native20 already exposes total
standardized k3, but not total standardized k4.  A predictor consuming only
the public native20 tape therefore appends **four** full-width fields:

```text
k4/sigma**4, uV, u3, u4.
```

Only `uV,u3,u4` are new provider-split coordinates.  Inside a fused D11
implementation the total k4 is already live, so this is three new ephemeral
reads plus one zero-product exposure.  Calling the external input widening
"three fields" is inaccurate.

These three splits are minimal, not universally sufficient.  In particular
`e2` is a diagonal consistency residual, not a theorem that missing covariance
is diagonal, and equality-sector splits do not identify an arbitrary
all-distinct dark fiber.  The held response gate below is therefore binding.

## 4. Exact code-ready liveness schedule

At reset `j`, compute `e2_j` from the still-live pair raw-moment diagonal and
the authoritative univariate raw2 immediately before the diagonal overwrite.
After the next weight push, reuse the live `weight2` buffer to form `uV`.
Read `diag(cp_pre_k21)` after `_cp_read`, and retain the CP4 marginal vector
immediately after `_cp4_read_planes` (or its final diagonal specialization).
Expose total `pre_k4/sigma**4` beside them.  Feed these values directly to the
shared pointwise head and release them at the end of the layer.

The first reset is `_exact_first_relu`.  Its pair raw-second-moment diagonal
equals the exact univariate raw2, so `e2_0=0` identically.  The layer-1 `uV`
field is therefore a literal zero and must not trigger a matvec.  Nonzero
variance-defect pushes are only `j=1,...,14 -> l=2,...,15`: fourteen actions.

Likewise the response transport at `l=1` multiplies the zero initial response
and can be elided.  The fused physical response uses fourteen signed-W
matvecs for `l=2,...,15`.  Hence the exact minimum dense-action census is

```text
14 squared-W skinny actions + 14 signed-W skinny actions = 28 matvecs
28 * 1024 * (2*1024-1) = 58,691,584 FLOPs
28/1024 = 0.02734375 dense-GEMM equivalents.
```

The frozen v1 design's 29 actions / `60,787,712` FLOPs are conservative, not
minimal: they charge the identically zero first variance push.  The
24->32->3 biased pointwise head costs `26,542,080` affine operations and
`491,520` ReLU comparisons over 15x1024 nodes.  Keeping its conservative
`4,915,200` scalar-response reserve gives

```text
total upper arithmetic = 90,148,864
                       = 0.04199934355153884 dense-GEMM equivalents,
```

using `1024^2*(2*1024-1)` as one dense-GEMM reference.  The 899 float32
weights occupy 3,596 bytes.  A 256 KiB incremental streaming reserve is safe:
the only large new object is the 32x1024 float32 hidden block (128 KiB), while
all width vectors can be reused and no reverse tape or dense plane survives a
layer.

The resource claim is void if an implementation recomputes CP3/CP4, forms a
new `W**2`, stores all-layer features, or misses the pre-overwrite M11 seam.

## 5. Single target-free estimator and one-network falsifier

The shortest falsifiable estimator is the frozen shared pointwise map

```text
input  = native20 + [k4/sigma**4, uV, u3, u4]   # width 24
head   = affine(24,32), ReLU, affine(32,3)       # 899 parameters
output = [delta v/sigma**2, delta k3/sigma**3, delta k4/sigma**4].
```

It is target-free at inference.  Training uses only physical state-gap labels
on panel networks `100,150,200,250,300,400,500,600,650`; it uses no q,
layer-mean, endpoint, observer-selection, network-ID, or board target.  The
sole held network is rank 35.  Architecture, normalization, last epoch,
features, layers, and the existing rank-35 LOO coefficients

```text
betaV=0.8273280548817595,
beta3=0.32346157239952833,
beta4=0.15489222117700333
```

must all be frozen before any held state or endpoint is opened.

For predicted physical gaps, form the three component-pure sources around the
same host base with the exact frozen `edg2_relu_mean` map, then fuse

```text
r_0 = 0
r_l = phi_l * (r_(l-1) @ W_l)
      + betaV*sV_l + beta3*s3_l + beta4*s4_l.
```

If the three gaps are exact, this reproduces the rank-35 perfect-state column;
its certified held oracle floor is `7.153517343378912e-9`.

The one-shot rank-35 sentinel passes only if all of the following hold:

1. with all three predicted gaps set to zero, the gates-off D11 lineage
   reproduces the matched `deployed_rank0035` parent byte-for-byte;
2. all states, sources, and responses are finite;
3. letting `z_hat` and `z_star` denote the combined predicted and exact
   rank-35 V/K3/K4 endpoint correction vectors,
   `cos(z_hat,z_star) >= 0.983459`;
4. `||z_hat-z_star||_2/sqrt(1024) <= 3.79871e-6`;
5. the noise-corrected held endpoint MSE is strictly below `1e-8`; and
6. each predicted component response has nonnegative inner product with its
   exact held component response.

Any failure closes this exact split/head/split/network spelling.  It does not
authorize a larger head, provider subset, layer window, altered beta, free-q
loss, retry, or a second held network.  A pass merely requests authority for
the remaining nine unchanged LOO evaluations.

## 6. Final verdict

The frozen v1 design is **not executable** because its q-oracle-failure
interlock is contradicted by the measured oracle pass.  As a post-q fallback,
the mathematically shortest lawful successor is a new v2 freeze containing
exactly the four external fields above (three minimal split coordinates), the
899-parameter state-gap head, and the rank-35 one-shot response gate.  Its
minimum incremental arithmetic is `90,148,864` operations under the stated
convention, with no new dense state.  There is no analytic alignment guarantee;
the proposed sentinel is intentionally a fast falsifier of the missing
provider-observability implication.

## 7. Static bindings

```text
frozen design
  e6cae82977371d31e9e32a89d9eeaf59d30fcfd03494142f6772a5ac025a5aab
frozen design receipt
  855629cbc27649394fe9ca9f4076e50719f5231a0a9d47da2e91075524b2cbe3
perfect-state ceiling
  4a476a61364476a3eb9d625c0831aeb580ae4575893a7abe8291aa2884de50e5
perfect-state source
  63cf25f83aab4d2dbd5c808bcebf0c57980e278a75fdbbbf9c2cd07e1ea08cb1
literal D11 source
  ace56bf60b267666e4f2a6d8ae772a089f234ebc725f34a4798b2f9e8284cbc2
direct provider terminal result
  e1ed27d0882dddb09698da78d167f7e34096ce44742b210b64792a3d20babd7f
owned four-field terminal result
  d84c5831d3392961d04fcf23323794ca75858190924672aa21c4e496731b98f5
```

