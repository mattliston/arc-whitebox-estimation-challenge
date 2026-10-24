# Q24 reverse-action K21 source audit

Status: **HOLD — live source exists, but the smallest rank-0 capacity screen is
negative.  No observer-fit host panel was regenerated.**

This audit is an observer-fit/source-information experiment.  It makes no
submission and opens no observer-selection or board target.  Its one target
read is the already-public observer-fit rank-0 `q_truth-q_bank` diagnostic.

## Live source seam

The current Q24 loop forms, at every zero-based nonlinear layer 1 through 14,

```text
cp_pre_k21    = _cp_read(...)
cubic_pre_k21 = _push_cubic(...)
pre_k21       = cp_pre_k21 + cubic_pre_k21
```

before `_full_relu_reset`.  Thus the complete directed plane

```text
K_l[i,j] = cum(Y_l[i], Y_l[i], Y_l[j])
```

is already live.  Reading a thin action from it does not reconstruct a tensor
and does not repeat the closed scalar CP3-provider split.  The all-688 native
tapes persist only `host.npy`, `raw20.npy`, and `sigma.npy`; none preserves an
off-diagonal plane, factor bank, or action sketch.

## Frozen smallest bank

Let `H_r` be the first `r` columns of the normalized 1024 Walsh-Hadamard
matrix.  Let `Phi_l` be the target-blind analytic Gaussian gate already stored
as native field 5.  Construct a weights-only reverse endpoint sketch by

```text
D_15 = qf(Phi_15[:,None] * H_r)
D_l  = qf(Phi_l[:,None] * (W_(l+1) @ D_(l+1))),  l=14,...,10,
```

where `qf` is deterministic modified Gram-Schmidt with positive diagonal.
For `l=10,...,14`, define the standardized off-diagonal directed plane

```text
T_l[i,j] = 0                                      if i == j
           K_l[i,j] / (sigma_l[i]^2 sigma_l[j])  otherwise.
```

The smallest new per-coordinate feature is

```text
F_l = T_l @ D_l,             l=10,...,14,
F_15 = 0.
```

The orientation-complete arm appends `T_l.T @ D_l`.  A Gaussian control arm
appends the off-diagonal correlation action

```text
G_l = (diag(sigma_l)^-1 C_l diag(sigma_l)^-1 - I) @ D_l.
```

These features preserve a signed receiver direction that raw20 row
mean/RMS/absolute summaries destroy.  No endpoint target, learned direction,
or external estimator defines `D_l`.

## Exact incremental arithmetic

For `n=1024`, `r=8`, five exposed K21 layers, six reverse-basis layers, and the
existing six-layer 64-wide q-head, define

```text
M(n,r)  = (2n-1) n r
GS(n,r) = r(r-1)/2 (4n-1) + 3nr
```

under the frozen dot/multiply/subtract/sqrt/divide circuit.  The reverse basis
cost is

```text
5 M + 6nr + 6 GS = 84,729,688 FLOPs.
```

One standardized off-diagonal K21 action costs

```text
5 (M + 4nr) = 84,008,960 FLOPs.
```

Adding eight inputs to the existing q-head first layer costs

```text
6 n (2r) 64 = 6,291,456 FLOPs.
```

Therefore the exact arithmetic increments are:

| arm | added FLOPs |
|---|---:|
| reverse `T@D`, r=8 | 175,030,104 |
| reverse `T@D` and `T.T@D`, r=8 | 265,330,520 |
| both K21 orientations plus `G@D`, r=8 | 355,589,976 |

The smallest row-action tape is 163,840 bytes per row for layers 10:14, or
112,721,920 bytes for all 688 observer-fit rows.  A literal meter would still
be required before deployment, because casts, clipping, array setup, and
payload reads are outside this arithmetic identity.

## Existing-data availability

- The 688 current native rows have no recoverable off-diagonal action.
- The 35-row rich tape contains only scalar row mean/RMS/absolute summaries;
  it also cannot reconstruct `K@D`.
- The quarantined rank-0 CP3 snapshot contains full source-only planes and is
  adequate for an algebra/capacity check.  It failed host byte parity by an
  RMS of `2.21e-8`, so it is not licensed as a current-Q24 scientific score.
- The rank-0 chronological-carrier archive contains the matching public
  `q_truth` and `q_bank` arrays.  It is observer-fit evidence only.

## Rank-0 capacity result

The diagnostic used zero-based q layers 10:15, deterministic four-fold
coordinate holdout (`index mod 4`), separate affine ridge per layer, and one
fixed pooled ridge selected from `[1e-4,1e-3,1e-2,1e-1]`.  The baseline was
raw20 plus `q_bank`.  Because this is one network with coordinate rather than
network holdout, it can reject a weak source but cannot license promotion.

| features | held local-q-gap MSE | ratio to baseline |
|---|---:|---:|
| raw20 + q_bank | 5.498163582850841e-9 | 1.000000 |
| + r4 K21 row action | 5.528478655850982e-9 | 1.005514 |
| + r8 K21 row action | 5.557668143282336e-9 | 1.010823 |
| + r16 K21 row action | 5.608408803450642e-9 | 1.020051 |
| + r4 row+transpose+covariance | 5.611443045137274e-9 | 1.020603 |
| + r8 row+transpose+covariance | 5.700824787622343e-9 | 1.036860 |
| + r16 row+transpose+covariance | 5.936146983618058e-9 | 1.079660 |

The negative trend with rank is evidence against paying for a 128/32 panel
under this fixed reverse-probe spelling.  It does not close a receiver-local
or nonlinear carrier-action statistic.

## Bounded successor gate

Do not regenerate 200 hosts.  A distinct successor must first identify a
receiver-local contraction not reducible to the frozen global `D_l` bank.  It
then gets exactly 12 observer-fit networks: eight fixed development ranks and
four fixed validation ranks, captured target-blind before their q targets are
opened.  Promotion to the existing 128/32 split requires at least a 1% mean
validation improvement over the frozen alpha=.75 J plus q-head, with the same
ridge/head and no rank selection on validation.  The protected 40 stays
sealed until that network-held-out gate passes.

## Bindings

- Current instrumented Q24 source SHA-256:
  `f539889324fb82cc1ded35a41c8b5a359014d9ddac52eab62949257e8e58a0ef`
- Quarantined snapshot manifest SHA-256:
  `ef187af733a2f8ae46b8ad3d85e3b17d8a41722063588f0d51e0fab46074ae3c`
- Rank-0 q diagnostic archive SHA-256:
  `07306f9d012fbb89c43360259126bfd703904623051635a1124d7e35f5a6724e`
- Rank-0 weights SHA-256:
  `399488100ef907d6a76fa27e911285809bf5d53405bc59290c84396a7e4c8c97`

