# Source-gate accounting addendum for the difference-replica K6 carrier

Status: **STATIC ADDENDUM ONLY — HOLD SOURCE GATE, ROWS, TARGETS, AND RUNS**  
Date: 2026-08-27

This immutable addendum does not alter the frozen production package.  It
binds the distinct resource bill and normalization used by the future
target-blind 315-cell source diagnostic.

## Split custody and coefficients

For each pool, split the already frozen ordered replica indices into two
disjoint contiguous halves.  The two half-estimators use ranks

```text
                 [6]     [4,2]     [2,2,2]
half rank         256       96          64
coefficient   1/(2*256) -15/(2*96)     15/64
```

Within either half, `[4,2]` pairs the first 96 half-local A replicas with all
96 B replicas.  `[2,2,2]` pairs the first 64 half-local A and B replicas with
all 64 C replicas.  A, B, and C remain mutually independent; the two halves
are independent because their underlying Gaussian paths are disjoint.  Each
half is therefore separately population-unbiased for the same sector action.

No full-rank denominator is allowed in a half-estimator.  Concretely, the
three coefficients are `1/512`, `-5/64`, and `15/64`.

## Exact seven-sector diagnostic bill

The gate reuses the same 1664 paths, propagated states, mixed-factor rows,
and 90 float64 map GEMMs as production.  It replaces only the total-support
polynomial assembly and type-reduction lines by all-seven-sector assembly and
split reduction/statistic lines.

For one compressed polynomial term at one output, the conservative census is
at most five product multiplies, one coefficient multiply, and one
accumulation: seven float64 operations, billed as 14 official FLOPs.  Applying
the frozen term tables to the full ranks gives:

```text
sector      [6] terms  [4,2] terms  [2,2,2] terms  weighted units       FLOPs
411                 4             9               16            5,824   1,252,392,960
321                 5            13               29            8,768   1,885,470,720
222                 3             6               12            4,224     908,328,960
3111                7            19               42           12,608   2,711,224,320
2211                8            22               52           14,976   3,220,439,040
21111              10            28               65           18,816   4,046,192,640
111111             11            29               66           19,648   4,225,105,920
--------------------------------------------------------------------------------------
all sectors                                                     84,864  18,249,154,560
```

Here `weighted units = 512*t6 + 192*t42 + 128*t222`, and each row is
`15*1024*14*weighted_units`.

For each sector and half, three float64 rank reductions plus their scales and
two type additions cost at most

```text
(256-1) + (96-1) + (64-1) + 3 + 2 = 418
```

ordinary float64 operations per output.  An additional eight-operation
per-output reserve covers the pointwise products, differences, squares,
reductions, and scalar guard/ratio work for `S` and `N`.  With two halves,
seven sectors, double billing for float64, 15 layers, and width 1024, the
fixed line is

```text
15*1024*2*7*(2*418 + 8) = 181,493,760 FLOPs.
```

The exact diagnostic reconciliation is:

```text
production enumerated core                         625,985,232,896
- production total-Q3+ assembly                     -1,411,645,440
- production type reductions/scales                    -25,620,480
+ seven-sector assembly                              18,249,154,560
+ split reductions and S/N reserve                      181,493,760
-------------------------------------------------------------------
diagnostic enumerated core                          642,978,615,296
+ unchanged conversion/copy/call padding              2,786,811,904
-------------------------------------------------------------------
diagnostic admission cap                            645,765,427,200
```

Against the banked D4 maximum `1,356,595,288,584`, one role/network gate call
has conservative combined cap `2,002,360,715,784`, utilization
`0.9105682311`, and margin `196,662,539,768` below `2^41`.  The three frozen
role ranks are three sequential calls; costs are not combined into one
estimator invocation.

Sector assembly is streamed in the already reserved arena.  The projected
combined peak therefore remains `5,679,808,512` bytes, leaving
`2,910,126,080` bytes below 8 GiB.  Official FLOP and RSS receipts remain
mandatory if a gate is later authorized.

## Empirical action-energy gate

In each of the 315 independently declared cells (15 layers, three role ranks,
seven sectors), let the two disjoint-half action vectors be `y1` and `y2` and
compute

```text
S = <y1,y2>
N = ||y1-y2||^2/2.
```

The frozen pass rule remains `S>0` and `N/S<=1/9` in every cell, with one
failure closing the spelling.  Independence and unbiasedness imply
`E[S]=||mu||^2` and `E[N]=tr(Sigma_half)`, so their population ratio motivates
the 90% expected-action-energy threshold.  A single realized ratio is an
**empirical gate statistic**, not a finite-sample confidence bound and not a
proof that the latent ratio is at most `1/9`.

No pooling, retry, alternate split, rank change, seed change, layer/sector
omission, or coefficient adjustment is permitted.  This addendum authorizes
no source call, D4 integration, residual measurement, target access, release,
or submission.

