# V34 physical-state response sentinel v2

Status: **STATIC FREEZE PENDING TWO INDEPENDENT AUDITS.**  These bytes grant no
target, cache, estimator, training, held-network, execution, integration,
meter, package, board, or submission authority.

## 1. Why this is the sole post-q successor

The exact start-5 q oracle passed, but both frozen target-blind q providers
closed:

```text
direct transpose provider result     e1ed27d0882dddb09698da78d167f7e34096ce44742b210b64792a3d20babd7f
owned L3/L3² four-field result       d84c5831d3392961d04fcf23323794ca75858190924672aa21c4e496731b98f5
four-field gap recovery              0.0935476% versus 8%
```

The frozen v1 V34 design is unreachable because it required the exact-q
oracle to fail.  This v2 contract supersedes only that activation premise.  It
tests a different map: target-free D11 provider decompositions to the three
physical preactivation gaps `(v,k3,k4)`, beginning at layer 1 and supervised
only on those state gaps.

The ten-network perfect-state capacity ladder is:

```text
deployed              3.197225394770670e-8
exact v               1.465583780991439e-8
exact v+k3            1.028608386484068e-8
exact v+k3+k4         9.254687418209847e-9
rank35 v+k3+k4        7.153517343378912e-9
```

This is capacity evidence only.  The rank-35 sentinel below falsifies the
missing observability claim.

## 2. Exact minimal provider interface

For target preactivation `l=1,...,15`, use D11's host scale
`sigma_l=sqrt(max(v_l,1e-30))`.  The four fields appended beyond public
native20 are

```text
total4_l = pre_k4_l / sigma_l^4
e2_j     = raw2_univariate_j - diag(M11_pair_j)       # before overwrite
uV_l     = (W_l**2).T @ e2_(l-1) / sigma_l^2
u3_l     = diag(cp_pre_k21_l) / sigma_l^3
u4_l     = cp4_pre_provider_planes_l[0] / sigma_l^4.
```

There are three genuinely new provider-split coordinates `(uV,u3,u4)`;
`total4` is a zero-product exposure required only because native20 already
contains total standardized k3 but not total standardized k4.

The identities are literal in the pinned D11 source:

```text
pre_k21 = cp_pre_k21 + cubic_pre_k21
(pre_k4,pre_k31,pre_k22)
          = host_quartic_planes + cp4_pre_provider_planes.
```

`u3` is read after `_cp_read` (or from `_cp_k3` at the final diagonal-only
layer); `u4` is read after `_cp4_read_planes` (or
`_cp4_read_diagonal` at the final layer).  `e2` is captured from the still-live
pair raw moment immediately before `_copy_with_diagonal`.  Missing that seam,
recomputing CP3/CP4, or forming a new squared-weight matrix voids the cost
claim.

At the exact first reset, pair and univariate raw-second diagonals coincide,
so `e2_0=0` identically and `uV_1=0` without a matrix-vector action.  The
variance diagnostic is not a claim that missing covariance is diagonal, and
the equality splits do not identify arbitrary all-distinct dark fibers.

One scalar split per degree is information-minimal for this mechanism: at
fixed degree totals, variance consistency, CP3-versus-star, and
CP4-versus-star/pairing form three independent scalar fibers.  Fewer than
three added coordinates have differential rank at most two and leave a
nonzero redistribution invisible.  This dimension argument proves minimality,
not sufficiency.

The data-free reference algebra is `provider_fields.py`.  Its synthetic test
binds provider formulas, independent previous/target permutation
equivariance, the exact-first-reset elision, the 24-field interface, and the
fused response identity.

## 3. Frozen 899-parameter state-gap head

The single shared pointwise head is

```text
input  : native20 + [total4,uV,u3,u4]                 # 24
trunk  : affine(24,32), ReLU
output : affine(32,3)                                 # dV,d3,d4
params : (24+1)*32 + (32+1)*3 = 899
```

Its outputs have fixed physical meaning:

```text
dV_l = (truth_pre_v_l  - host_v_l ) / sigma_l^2
d3_l = (truth_pre_k3_l - host_k3_l) / sigma_l^3
d4_l = (truth_pre_k4_l - host_k4_l) / sigma_l^4.
```

Fit networks are exactly
`100,150,200,250,300,400,500,600,650`; rank 35 is the only holdout.  All
layers 1--15 and all 1024 units enter once.  No q, endpoint, post-mean,
residual, network-ID, observer-selection, board target, or held-rank statistic
may enter training or checkpoint selection.

Before any fit, compute population mean/std for each of the 24 inputs and RMS
for each of the three labels on the nine fit networks only.  Clamp only zero
scales to one.  Train standardized labels with equal component MSE using
float32 AdamW, batch size 4096, `lr=0.001`, `weight_decay=0.0001`, exactly 64
epochs, deterministic shuffle/initialization seed `0x56334`, and cosine decay
over exactly `64*ceil(138240/4096)=2176` optimizer steps.  Use PyTorch's
default seeded `Linear` initialization, no dropout, no early stopping, and the
fixed final epoch.  Freeze the model and normalization hashes before opening
any rank-35 state or endpoint target.  These choices are preregistered
protocol, not post-result tuning authority.

## 4. Exact L1-start physical response compiler

Let `F=edg2_relu_mean` from the perfect-state receipt.  Convert predicted
standardized gaps into

```text
v+  = max(v + sigma^2*dV, 1e-30)
k3+ = k3 + sigma^3*d3
k4+ = k4 + sigma^4*d4.
```

Around the same host base, form component-pure sources

```text
sV_l = F(m_l,v+_l,k3_l,k4_l) - F(m_l,v_l,k3_l,k4_l)
s3_l = F(m_l,v_l,k3+_l,k4_l) - F(m_l,v_l,k3_l,k4_l)
s4_l = F(m_l,v_l,k3_l,k4+_l) - F(m_l,v_l,k3_l,k4_l).
```

Use the already-frozen rank-35 leave-one-network-out coefficients

```text
betaV = 0.8273280548817595
beta3 = 0.32346157239952833
beta4 = 0.15489222117700333.
```

With D11's row-vector weight orientation, the sole overlay is

```text
r_0 = 0
r_l = phi_l * (r_(l-1) @ W_l)
      + betaV*sV_l + beta3*s3_l + beta4*s4_l,  l=1,...,15
prediction_15 = deployed_15 + r_15.
```

The `l=1` transport is exactly zero and is elided.  Linearity of transport
makes this fused recurrence identical to transporting the three component
columns separately and combining them at the endpoint.  If the predicted
gaps equal the true gaps, it reproduces the rank-35 V+K3+K4 column in
`PERFECT_STATE_CEILING.json` exactly.  This is a frozen-state response overlay,
not nonlinear feedback into D11's covariance/CP state.

## 5. One-shot 9/1 rank-35 sentinel

No rank-35 state or endpoint target may be opened until: (a) two independent
static audits bind every v2 byte; (b) a release grants only the nine fit rows;
(c) the fit completes once; (d) model/normalization hashes are frozen; and
(e) a second exclusive one-shot claim grants the single held row.  A failed
fit/preflight may not inspect rank35.  Once rank35 is claimed, any semantic
failure is terminal and nonretryable.

The source-zero control sets `(dV,d3,d4)=0` after the frozen head and must
reproduce the pinned instrumented D11 parent byte-for-byte.  The source-on arm
passes only if all gates hold:

1. every feature, state, source, response, and score is finite;
2. `cos(z_hat,z_star) >= 0.983459` for the combined predicted and exact
   rank-35 V+K3+K4 endpoint response vectors;
3. `||z_hat-z_star||_2/sqrt(1024) <= 3.79871e-6`;
4. held noise-corrected endpoint MSE is strictly below `1e-8`;
5. each predicted component response has nonnegative inner product with its
   exact held component response; and
6. source-on strictly beats source-zero in noise-corrected endpoint MSE.

Failure closes this exact provider split, head, training schedule, layer
window, and held network.  It grants no subset, larger head, altered beta,
free-q loss, retry, or second held network.  Passing requests, but does not
grant, nine unchanged leave-one-network-out evaluations.

## 6. Exact incremental resource shell

There are fourteen nonzero squared-weight variance pushes (`l=2..15`) and
fourteen nonzero fused signed-weight response pushes.  At width 1024:

```text
dense action FLOPs          28*1024*(2*1024-1) = 58,691,584
dense GEMM equivalents      28/1024            = 0.02734375
head affine operations                           26,542,080
ReLU comparisons                                    491,520
scalar physical reserve                           4,915,200
arithmetic upper (comparisons separate)          90,148,864
dense-equivalent arithmetic upper          0.04199934355153884
all operations including comparisons             90,640,384
```

The 899 float32 weights occupy 3,596 bytes.  The frozen incremental streaming
workspace upper bound is 256 KiB; no all-layer tape, new dense plane, or
reverse state is allowed.

## 7. Binding disposition

- `V34_V1_ACTIVATION_INTERLOCK`: **unreachable; never execute v1**.
- `V34_V2_POST_Q_ACTIVATION`: **satisfied by the two terminal q-provider
  results, subject to independent audits and explicit releases**.
- `V34_PROVIDER_INTERFACE`: **four external fields / three minimal split
  coordinates**.
- `V34_L1_RESPONSE_RECURRENCE`: **exact relative to the frozen EDG2
  perfect-state column**.
- `V34_VALUE`: **unknown; one rank-35 held falsifier only**.
- `V34_CURRENT_AUTHORITY`: **static audit only; no data or execution**.
