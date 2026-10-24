# Additive preregistration correction: fp64 advanced-gather billing

Status:
`SEALED_BEFORE_SOURCE__SUPERSEDES_CS3B_LOWRANK_COST_AND_SELECTOR_LAW`.

This correction binds immutable CS3B preregistration SHA-256
`dde30939ef271523a7bda593903d9377da8372ffd56d0fa1684c479d9a9c9e5f`,
null-division correction SHA-256
`4f95cdc324e31459e7e700c54c68f47d8164eca261286a93521f0df9880de253`,
and pinned flopscope `0.12.0` ndarray source SHA-256
`fa5bbecb48b123ab77f31ced3da877d2e333aa5b403a324c33da591b967d3e74`.
It was sealed before any CS3B candidate source or execution.

## Correction

The sealed preregistration applied the advanced-gather base charge of four per
materialized output element but omitted the fp64 dtype rate of two.  Each fp64
advanced `getitem` therefore bills eight weighted FLOPs per gathered output
element.  The four frozen gathers

```text
U     = W[idx, :]       # r*w outputs
Crows = C[idx, :]       # r*w outputs
A     = left[:, idx]    # w*r outputs
G     = Crows[:, idx]   # r*r outputs
```

cost exactly

```text
24*r*w + 8*r^2,
```

not `12*r*w+4*r^2`.

The five fp64 matmuls plus three fp64 combines remain
`16*r*w^2+4*r^2*w-4*r*w`; the fp64 current-reset mask remains `8*w^2`; and
the invert plus `flatnonzero` scans remain `2*w`.  The corrected complete
low-rank branch law is consequently

```text
Lr_star(r,w)
= 16*r*w^2 + 4*r^2*w + 8*w^2 + 20*r*w + 8*r^2 + 2*w.
```

The corrected and only authorized selector is

```text
LOWRANK64 iff Lr_star(r,w) <= D0(w),
D0(w) = 4*w^3 + 8*w^2;
otherwise DENSE32.
```

No old formula or hard-coded crossover may survive in candidate source,
observer code, tests, receipts, or projections.  The number of calls is
unchanged: 18 on a low-rank activation, 10 on an executed dense activation
including selector scans, and the corrected global delta remains

```text
Delta_calls = 18*a_l + 10*a_d - 14.
```

## Exact crossover replay

The corrected integer crossovers include:

| Width | Largest selected `r` |
|---:|---:|
| 4 | 0 |
| 5 | 0 |
| 6 | 1 |
| 8 | 1 |
| 12 | 2 |
| 16 | 3 |
| 32 | 7 |
| 64 | 14 |
| 128 | 29 |
| 256 | 60 |
| 512 | 120 |
| 1024 | 241 |

At width 1024,

```text
Lr_star(241,1024) = 4,294,999,816 <= 4,303,355,904
Lr_star(242,1024) = 4,313,779,744 >  4,303,355,904.
```

The width-1024 crossover happens not to change, but width 5 and width 128
demonstrate that the correction is operational and must be exercised by the
small-width source probe.

## Worst-case projection

The executed dense fallback remains

```text
Dh(w)=D0(w)+2*w
```

because it owes the same invert and `flatnonzero` selector scans.  Every
selected low-rank branch is at most `D0`, so the all-fourteen worst case is
still all dense.  The corrected worst-case width-1024 total is unchanged:

```text
2,131,652,533,303 + 14*(4,303,355,904 + 2,048) - 112
= 2,191,899,544,519 FLOPs,
```

or `99.67605112792626%` of `2^41`, with `7,123,711,033` projected headroom.
This remains a projection, not a literal candidate meter.

## Gate correction

Every future static checker, source observer, small-width delta fit, crossover
fixture, source result, and design receipt must bind `Lr_star`.  In particular,
the frozen width-5 rank-one case must select `DENSE32`, and the width-128
rank-30 case must select `DENSE32`; either selection under the old law is a
fatal source failure.  All algebra, state, certificate-equivalence, null-face,
target-free, and authorization boundaries in the original preregistration and
null correction remain unchanged.

