# P2R0C1D5 CP3 depth-4 Winograd hybrid preregistration

Status: **FROZEN, UNEXECUTED, TARGET-FREE GATES REQUIRED**  
Date: 2026-08-26  
Parent: `P2R0C1D`  
Scope: one accuracy/compute-hybrid candidate derived from the exact estimator
that produced public submission `#328439`. This document authorizes no target
read, training, submission, canonical-graph edit, or network action.

## 1. Frozen hypothesis

Replace only the 58 float32 dense contractions in the full, non-truncating
CP3 carrier with a depth-4 batched Strassen--Winograd kernel. Preserve every
rank, birth, coefficient, residual star, CP4 path, Q24 path, transport
correction, and final marginal read. The candidate is exact over real
arithmetic but changes float32 association, so it is **not** an output-parity
claim. It is the compute leg of a hybrid: even at unchanged raw error it does
not reach adjusted `1e-8`.

The kernel must use only ordinary `flopscope.numpy` operations:

- leaf products: `fnp.matmul`;
- arithmetic: explicit `fnp.add` / `fnp.subtract` (ordinary `+`/`-` on
  `FlopscopeArray` is equivalent);
- seven-way operand assembly: `fnp.stack`;
- quadrant assembly: `fnp.block`, or top-level `fnp.concatenate(..., out=)`
  with exactly the same output-numel bill.

Forbidden: `tensordot`, TD52 or high-rank singleton padding, custom meter
weights, plain-NumPy arithmetic inside the budget, off-meter callbacks,
symmetry discounts, lower precision, rank truncation, randomized products,
or changing the carrier state.

## 2. Exact replacement roster

The standing source initializes rank 1,024 after h1 and adds 1,024 births at
each nonfinal reset. Entering ranks for loop layers 1--15 are therefore
`r_l = 1024*l`; the cap 16,384 never truncates them.

At each layer 1--15, `_cp_push` has two products
`(1024,1024) @ (1024,r_l) -> (1024,r_l)`. At each nonfinal layer 1--14,
`_cp_read` has two `einsum("ir,jr->ij", ...)` contractions, equivalently
`(1024,r_l) @ (r_l,1024) -> (1024,1024)`. The layer-15 `_cp_k3`
matrix-vector read is excluded.

| layer | rank | push shape, calls | read shape, calls |
|---:|---:|---|---|
| 1 | 1,024 | `1024x1024 @ 1024x1024`, 2 | `1024x1024 @ 1024x1024`, 2 |
| 2 | 2,048 | `1024x1024 @ 1024x2048`, 2 | `1024x2048 @ 2048x1024`, 2 |
| 3 | 3,072 | `1024x1024 @ 1024x3072`, 2 | `1024x3072 @ 3072x1024`, 2 |
| 4 | 4,096 | `1024x1024 @ 1024x4096`, 2 | `1024x4096 @ 4096x1024`, 2 |
| 5 | 5,120 | `1024x1024 @ 1024x5120`, 2 | `1024x5120 @ 5120x1024`, 2 |
| 6 | 6,144 | `1024x1024 @ 1024x6144`, 2 | `1024x6144 @ 6144x1024`, 2 |
| 7 | 7,168 | `1024x1024 @ 1024x7168`, 2 | `1024x7168 @ 7168x1024`, 2 |
| 8 | 8,192 | `1024x1024 @ 1024x8192`, 2 | `1024x8192 @ 8192x1024`, 2 |
| 9 | 9,216 | `1024x1024 @ 1024x9216`, 2 | `1024x9216 @ 9216x1024`, 2 |
| 10 | 10,240 | `1024x1024 @ 1024x10240`, 2 | `1024x10240 @ 10240x1024`, 2 |
| 11 | 11,264 | `1024x1024 @ 1024x11264`, 2 | `1024x11264 @ 11264x1024`, 2 |
| 12 | 12,288 | `1024x1024 @ 1024x12288`, 2 | `1024x12288 @ 12288x1024`, 2 |
| 13 | 13,312 | `1024x1024 @ 1024x13312`, 2 | `1024x13312 @ 13312x1024`, 2 |
| 14 | 14,336 | `1024x1024 @ 1024x14336`, 2 | `1024x14336 @ 14336x1024`, 2 |
| 15 | 15,360 | `1024x1024 @ 1024x15360`, 2 | excluded final K3 vector read |

Totals: 30 push GEMMs plus 28 read GEMM-equivalent contractions = 58.

## 3. Frozen Winograd algebra and literal cost

For each even `m x k` times `k x n` product, split both operands into four
quadrants. Use this 7-product/15-add Winograd schedule:

```text
S1=A21+A22; S2=S1-A11; S3=A11-A21; S4=A12-S2
T1=B12-B11; T2=B22-T1; T3=B22-B12; T4=T2-B21
P1=A11 B11; P2=A12 B21; P3=S4 B22; P4=A22 T4
P5=S1 T1;  P6=S2 T2;  P7=S3 T3
U1=P1+P2; U2=P1+P6; U3=U2+P7; U4=U2+P5
U5=U4+P3; U6=U3-P4; U7=U3+P5
C=[[U1,U5],[U6,U7]]
```

At every level the bill is:

- 4 A-block plus 4 B-block adds/subtracts;
- 7 A-block plus 7 B-block elements copied by the two stacks;
- 7 output-block adds/subtracts;
- 4 output-block elements copied by block/concatenate.

Thus overhead per node is exactly
`11/4 * (m*k + k*n + m*n)`. The depth-`d` cost is

```text
7^d * (m/2^d)*(n/2^d)*(2*(k/2^d)-1)
+ sum(l=0..d-1) 7^l * 11*(m*k+k*n+m*n) / 4^(l+1).
```

All production dimensions are divisible by `2^7`; there is no padding.
Installed FlopScope prices each float32 leaf at the textbook scalar count
`m*n*(2*k-1)` and each explicit add/stack/block element at one. This is
ordinary scalar-operation reduction, not operator-price selection.

Across the 58 products:

| depth | honest kernel bill | saving versus ordinary contractions |
|---:|---:|---:|
| 1 | 847,842,377,728 | 118,244,245,504 |
| 2 | 746,611,408,896 | 219,475,214,336 |
| 3 | 661,941,616,640 | 304,145,006,592 |
| **4** | **594,693,332,992** | **371,393,290,240** |
| 5 | 547,817,207,808 | 418,269,415,424 |
| 6 | 527,741,313,536 | 438,345,309,696 |
| 7 | 546,821,157,760 | 419,265,465,472 |

The ordinary bill is 515,144,417,280 push + 450,942,205,952 read =
966,086,623,232. The fixed rank-gated batching below adds 20,971,520
stack-copy operations, so the preregistered net saving is exactly
**371,372,318,720** before the required literal common-driver meter.

## 4. Storage and batching schedule

Store P/Q as one `(bank=2, channel=2, 1024, 16384)` float32 allocation;
this has the same element count as the incumbent's two separate P/Q banks.
For `_cp_read`, use two `(channel=2,1024,16384)` product workspaces holding
`[P^2,PQ]` and `[Q*lambda,P*lambda]`; this is two width-by-cap buffers more
than the incumbent and is included in the bound below.

To reduce Python dispatch without exceeding the conservative RSS policy:

- batch the two push channels only for ranks <=10,240; duplicate the common
  left weight with one billed two-way `fnp.stack` (10 stacks total);
- batch the two read channels only for ranks <=11,264;
- run the two channels sequentially above those thresholds, freeing the
  recursion workspace between them;
- explicitly `del` S/T temporaries before recursion and child inputs before
  recombination; use the U schedule above with no more than five live output
  blocks.

The batching thresholds are part of the frozen candidate. Changing them is a
new preregistration.

## 5. Conservative memory bound

Let `q=7/4`. For one product, the conservative extra-element bound used here
is

```text
sum(j=1..d) q^j*(m*k+k*n)       retained recursive operand stacks
+ q^(d-1)*max(m*k,k*n)          deepest S/T dependency transient
+ q^d*m*n                       leaf result
+ 3*q^(d-1)*m*n                 conservative recombination allowance
+ 2*m*n.                        output-bank allowance
```

It intentionally adds the whole workspace to the incumbent maximum RSS even
where buffers replace incumbent storage.

- depth 4, sequential largest push rank 15,360: 844,165,120 extra float32
  elements = 3,376,660,480 bytes; plus incumbent 3,421,503,488 gives
  6,798,163,968 bytes = 6.3313 GiB;
- depth 4, batched push rank 10,240 plus the duplicated left input:
  7,902,887,936 bytes = **7.3601 GiB**, the candidate maximum, leaving
  687,046,656 bytes below 8 GiB;
- depth 4, batched read rank 11,264 including the two extra product buffers:
  7,838,334,976 bytes = 7.3000 GiB;
- depth 5, even sequentially at the largest push: 9,353,728,000 bytes =
  8.7113 GiB, 763,793,408 bytes over 8 GiB.

Depth 5 and deeper are therefore closed by this conservative schedule. The
depth-4 algebraic bound is not a runtime certificate; measured process RSS
must still pass strictly below 8 GiB on every gate call.

## 6. Frozen target-free execution gates

Implementation may begin only from the exact bound source and must first
produce its own source hash and a no-target roster manifest. Then run exactly
three cold common-driver calls in order `[35,279,35]`, matching the incumbent
meter. No targets or board joins may be opened.

Every call must satisfy all of:

1. finite float32 output of shape `(16,1024)`;
2. literal `flops_used < 2^41` and exact operation sum equals total;
3. candidate CP3 bill/delta agrees with this receipt after accounting for the
   frozen 20,971,520 batching-copy bill;
4. process RSS strictly below 8 GiB;
5. outer wall strictly below 120 seconds;
6. residual wall strictly below 0.4 seconds; for deployment readiness the
   maximum must be <=0.385 seconds to retain 15 ms of operational margin;
7. repeat rank-35 candidate outputs byte-identical.

The incumbent meter's maxima (24.95499 s outer wall, 0.35143497 s residual,
3,421,503,488 bytes RSS) provide context, not a candidate certificate. The
trim receipt's 18.8 ms median improvement for about 1,100 fewer dispatches
cannot certify this kernel. A depth-4 kernel has 73 counted wrapper operations
per sequential product; the frozen batching schedule is about 2,731 wrapper
operations, making the residual gate genuinely tight and empirical.

## 7. Floating-order discriminator

Compare candidate and exact-incumbent predictions on the same prebound
weights-only roster, with no targets. Require:

- identical shape and dtype; both finite;
- repeat candidate byte parity;
- pooled final-layer RMS delta <= `1.714909904361275e-6` (1% of the
  #328439 aggregate raw RMSE used only as a scale reference);
- maximum absolute final-layer delta <= `8.574549521806375e-6` (5% of that
  scale reference);
- report per-layer RMS/max deltas and the worst row; no silent clipping.

Passing bounds numerical perturbation only. It does **not** establish raw MSE
or authorize a submission. Any target-authorized score must be a later,
separately frozen experiment.

## 8. Terminal decision rule

- Any static roster/cost mismatch, nonfinite, nondeterminism, RSS/wall/residual
  failure, or drift-gate failure: close this implementation form.
- If all target-free gates pass: mark `PREREG_TARGET_FREE_PASS`; the route
  remains accuracy-unproven and requires separate target authority.
- Do not infer unchanged raw error. At the board aggregate ratio, unchanged
  raw gives only a planning projection near `1.788e-8`; adjusted `<1e-8`
  still needs raw below about `1.645e-8` (about 44.1% below #328439).

