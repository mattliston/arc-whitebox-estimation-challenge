# PC-TT5 v1 terminal design audit and graph-ready repair split

Status: **CLOSE THIS FORM BEFORE D0**  
Date: 2026-08-26  
Scope: design/algebra audit only; no target access, experiment launch, estimator
build, submission, network action, or canonical-graph edit

## Decision

The exact `PC-TT5` spelling in `DESIGN_RECEIPT.md` is not D0-ready and should be
closed as `closed_this_form`. Changing the dense fixture from `n=4` to `n=5`
repairs only one contradiction. It does not supply the missing 3--5-variate
ReLU reset oracle, a certified finite correlation expansion, an entry circuit
under 8,192 counted FLOPs, or a symmetry-preserving and sector-balanced rank-16
carrier.

The broader `P2R0F3B2B` question should remain open, but its only active child
should first be a target-free **multivariate linked-cluster reset-oracle
derivation**. Rank, autonomous dense stability, compression, and production
economics are blocked dependents until that derivation exists.

There is deliberately no D0 implementation or run command in this receipt.
Running a dense or TT discriminator now would test an unspecified approximation
and could not produce interpretable evidence.

## Frozen evidence read

| artifact | SHA-256 | authority actually supplied |
|---|---|---|
| `DESIGN_RECEIPT.md` | `eee173f564b655ff17aec9a8962ba490ec3f2b5013cebbf34f7ee1203b81e6ff` | proposed augmented TT5 state and conditional cost envelope |
| `public_exact_ledger_v1/ledger_rig.py` | `c74c92862a3afad7e44a077c80f87fdd09c66be4f02718db24a21ea7b5f4a581` | univariate and bivariate `c[p,q]`, Edgeworth blocks, connected pair resets, and exact affine chains for `[5]`, `[41]`, `[32]` |
| `public_exact_ledger_v1/ledger_verify.py` | `51e4d8aa9ead031871f2fb7c7913d7c165e03f7346eb408b76ebcf58a921e0cd` | scalar Hermite columns and B2A verification rig |
| `public_exact_ledger_v1/LEDGER_VERIFY_RESULTS.json` | `f14b4964ab6d7da4917d4c4a77c08099477a840849819cc61ee46da0d842c938` | measured B2A partial-carry instability and algebra checks |
| `public_trivariate_derivation_v1/DERIVATION.md` | `5fec258709489fc5c36361bb6bb5fb3e0b8f780502d4ef9b48132bf040bb9444` | exact Gaussian Mehler master series; worked order-four `kappa211` and `kappa1111` formulas |
| `public_trivariate_derivation_v1/verify.py` | `bd58d0f0e3c5d6bd89765aa42a3f1d2121282ba5ce9b07b2d852cc99ab42a8c4` | deterministic evaluators for those order-four formulas only |
| `public_trivariate_derivation_v1/L5_DERIVATION.md` | `6c0dfdc022799953f8bd0f0372a7dbfdf8dbf9eeaf8649cb03f03600c49fee92` | complete **bivariate** third tilt shell `Lambda5 + Lambda3 Lambda4 + Lambda3^3/6` |
| `public_k5_probe_v1/estimator.py` | `5790af0b6f807807098cb47d3f64ba87869c6be049419615549f8827335574f6` | diagonal Gaussian `kappa5` emission and `W^5` diagonal push only |

No read artifact contains a general correlated 3-, 4-, or 5-variate ReLU
moment evaluator or reset formulas for fifth-order collision classes `[311]`,
`[221]`, `[2111]`, and `[11111]`. The trivariate derivation is a valid starting
identity, not an implementation authority for those missing classes or for the
non-Gaussian `E5` shell.

## Defects that make the original D0 non-executable

### 1. `n=4` cannot test `[11111]`

The exhaustive ordered fifth-tuple census is:

| class | `n=4` | `n=5` |
|---|---:|---:|
| `[5]` | 4 | 5 |
| `[41]` | 60 | 100 |
| `[32]` | 120 | 200 |
| `[311]` | 240 | 600 |
| `[221]` | 360 | 900 |
| `[2111]` | 240 | 1,200 |
| `[11111]` | **0** | 120 |
| total | 1,024 | 3,125 |

Thus `n=5` is the minimum dense dimension. A symbolic label at `n=4` is not a
numerical all-distinct test.

### 2. The 3--5D reset entry oracle is undefined

For a slot tuple grouped into `k <= 5` distinct coordinates with multiplicities
`p=(p_1,...,p_k)`, the missing authority would have to define

```text
f_p(h) = product_u relu(h_u)^p_u
Lambda_r = (1/r!) kappa_r : partial^r
E4 = I + Lambda3 + Lambda4 + (1/2)Lambda3^2
S3 = Lambda5 + Lambda3 Lambda4 + (1/6)Lambda3^3
E5 = E4 + S3.
```

If `G ~ N(mu,Sigma)`, the unambiguous expectation-side reference is

```text
M_p = E_G[(E5 f_p)(G)].
```

Grouping a derivative multi-index as `q=(q_1,...,q_k)`, its Gaussian kernel has
the exact infinite multivariate Mehler form

```text
K_{p,q}(R)
 = sum_{t_uv >= 0}
     product_{u<v} rho_uv^t_uv / t_uv!
     product_u c[p_u, q_u + s_u(t)] / sigma_u^q_u,

s_u(t) = sum_{v != u} t_uv.
```

The coefficient of each derivative channel is obtained from the symmetric
cumulant blocks

```text
L_r(q) = kappa_r[1 repeated q_1, ..., k repeated q_k]
         / product_u q_u!,     sum q_u = r,

D = 1 + L3 + L4 + (1/2)(L3 * L3)
      + L5 + (L3 * L4) + (1/6)(L3 * L3 * L3),
M_p = sum_q D(q) K_{p,q}(R).
```

Finally, for the original five labeled slots, the connected output is

```text
kappa(i_1,...,i_5)
 = sum_{pi in Partitions({1,...,5})}
     (|pi|-1)! (-1)^(|pi|-1) product_{B in pi} M_{i_B}.
```

These equations are a specification of the missing object, not a finite
algorithm. For `k>=3`, the Gaussian kernel is an infinite correlation series.
The repository has neither a frozen truncation law nor a certified tail bound
for the reached correlation domain. Calling the unbounded series "analytic" or
reusing a bivariate `tmax=40` loop does not resolve that gap.

### 3. No current 8,192-FLOP entry proof is plausible by enumeration

For `k=5`, the full `E5` shell has the following numbers of derivative
multi-indices:

```text
|q| = 3,4,6,5,7,9:  35 + 70 + 210 + 126 + 330 + 715 = 1,486
plus the identity:                                      1,487 channels.
```

There are ten Gaussian correlation edges. A total-bond cutoff `T` has
`C(T+10,10)` bond assignments: 1,001 at `T=4`, 3,003 at `T=5`, and 8,008 at
`T=6`. The `[11111]` Gaussian connected term first appears at `T=4` (the 125
labeled spanning trees), so `T<=3` omits the leading all-distinct sector.

The naive channel-by-bond work is already millions of terms. Factorization may
exist, but none is derived or counted. At `T=6`, enumerating bond assignments
alone consumes almost the entire 8,192 allowance before one cumulant query,
Hermite lookup, channel contraction, Mobius conversion, TT read, or symmetry
operation. Therefore the design's conditional cost table is not an executable
bound.

A legitimate finite oracle must freeze `T_k` (or another deterministic method)
for every distinct-coordinate count, prove a uniform raw-moment and connected-
cumulant error bound on a source-certified correlation/cumulant domain, and
present a literal arithmetic circuit whose count includes all work above. An
unpriced multivariate-normal CDF/orthant routine is not a substitute.

### 4. The original ablation gate removes the wrong object

Setting only `Lambda5=0` leaves `Lambda3 Lambda4 + Lambda3^3/6`, so it cannot
reproduce the K4-level operator. Exact K4 parity requires removing the **whole
third tilt shell** `S3`. `Lambda5=0` may remain a separate cancellation
diagnostic, and on a nonzero `kappa3/kappa4` fixture it should generally differ
from `E4`.

### 5. The stated B2A projection is not B2A

Zeroing only the three-or-more-distinct fifth-order entries leaves the complete
three- and four-coordinate sectors of `kappa3` and `kappa4`, which B2A never
carries. Define the exact projection

```text
P2(kappa_r)[i_1,...,i_r] = kappa_r[i_1,...,i_r]
                           if number of distinct indices <= 2,
                           else 0,       for r=3,4,5.
```

B2A parity is the operator identity obtained by applying `P2` to **all three
orders after every push and reset**, with the same standardization and
bivariate Mehler cutoff as `ledger_rig.py`:

```text
P2 Reset_E5(P2 Push(P2 state)) == ledger level-D step.
```

The comparison must be fieldwise on `k3d/K21`, `k4d/K31/K22`, and
`k5d/K41/K32`, not only on a mean or aggregate norm.

### 6. Aggregate sentinel rank is blind to lower-order sectors

The sentinel tensor embeds order `r` with `C(5,r)` copies and `n^r` entries.
An ordinary Frobenius or future-action score over the augmented order-five
tensor can pass while discarding `kappa3` or `kappa4`; those are precisely the
feedback states that B2A showed cannot be perturbed casually. A repaired rank
gate must report, and pass, every order separately:

```text
e_r = ||Jhat_r - J_r||_F / max(||J_r||_F, tau_r),  r in {3,4,5},
```

with frozen zero-sector floors `tau_r`, plus per-order suffix-action errors and
per-order unfolding spectra. No weighted aggregate may override a failed
sector. The cross objective must state how it enforces those simultaneous
constraints with shared rank 16; independent rank-16 fits would have different
storage and affine costs and must be remetered.

### 7. Ordinary TT-cross does not preserve cumulant symmetry

Every `kappa_r` is invariant under all `r!` slot permutations. Modewise affine
push preserves symmetry for an exact tensor, but an unconstrained rank-truncated
TT-cross need not. Canonicalizing an input tuple does not symmetrize its affine
contraction. Averaging all permutations costs up to 120 TT evaluations per
fifth-order read and invalidates the present cost envelope.

A reopened carrier must be permutation-equivariant by construction or provide
an independently audited symmetrization circuit inside the same 8,192-FLOP
entry cap. D0 must exhaustively check every permutation at `n=5`; later rank and
action gates must be per-order and symmetry-aware.

## Minimum repaired discriminator specification (conditional, not licensed)

The following would be a coherent replacement only after the reset-oracle and
cost theorem above are sealed.

1. **Dimension and fixtures.** Use `n=5`, seeds `51031` and `51047`, two
   deterministic SPD correlation matrices with every off-diagonal nonzero, and
   symmetric synthetic `kappa3..kappa5` with every collision class nonzero.
   Freeze matrix/state bytes and hashes before evaluation.
2. **Dense affine identity.** Compare the augmented modewise push to explicit
   dense contractions separately for orders 3, 4, and 5, relative error
   `<=1e-12` in every sector.
3. **Exhaustive reset identity.** Evaluate all 3,125 ordered fifth tuples and all
   lower raw moments needed by Mobius conversion. Compare the finite oracle to
   an independently implemented dense reference using the same frozen
   `T_k`; numerical equivalence `<=1e-10` is separate from the proved
   finite-versus-infinite tail bound.
4. **Correct shell ablations.** Require `E4` parity only when all of `S3` is
   removed. Report `Lambda5=0` as a distinct arm and require it to be nontrivial
   on the positive-control fixture.
5. **Exact B2A exposure.** Apply `P2` to all carried orders at each boundary and
   require fieldwise parity with `ledger_rig.py` at its exact cutoff and
   normalization.
6. **Non-equivalence.** A fixture with nonzero `[11111]` must differ from both
   the B2A projection and the mean-tangent/Duhamel operator.
7. **Symmetry.** Exhaustive maximum permutation defect `<=1e-12` before and
   after affine push, reset, and any rank operation, separately for each order.
8. **Sector-balanced rank gates.** No joint pass unless every `e_3,e_4,e_5` and
   every per-order future-action residual passes its frozen threshold. Report
   all four TT unfolding ranks for the joint object and for each extracted
   order sector.
9. **Cost.** Static source audit and literal instrumentation must both show
   `C_entry<=8192`, including kernel tail method, `E5`, connected conversion,
   incoming-state reads, and symmetry. Recompute the full `2^41` table if any
   representation, sweep, rank, or symmetry mechanism changes.

Until all nine items are instantiated in a hash-bound preregistration, there is
no honest command to run.

## Reopen criteria

Reopen this representation family only if a new receipt proves all of:

1. a complete 3--5D linked-cluster/Edgeworth reset formula for every collision
   class and every lower raw moment used by connected conversion;
2. a frozen finite Gaussian-correlation evaluation with a uniform tail bound on
   a source-certified reachable domain (including a specified out-of-domain
   law);
3. a literal `<=8192`-FLOP arithmetic circuit for the worst five-distinct entry;
4. exact whole-shell and `P2`/B2A identities;
5. a symmetry-preserving carrier with simultaneous per-order error/rank/action
   gates; and
6. a recomputed total production bound strictly below `2^41`.

Absent those proofs, another TT implementation would repeat B2A's central
mistake in a larger basis: autonomous feedback through an incompletely defined
state.

## Graph-ready decomposition (suggestion only)

No canonical graph file was edited. The following split keeps the broader
scientific question alive without showing an implementation node as active.

| suggested ID | label | status | terminal / unblock criterion | evidence |
|---|---|---|---|---|
| `P2R0F3B2B1` | Exact PC-TT5 v1 spelling | `closed_this_form` | Close pre-D0: `n=4` misses `[11111]`; 3--5D oracle, finite tail, symmetry, per-sector rank law, and executable cost proof are absent. Reopen only through a materially new sealed spelling satisfying all reopen criteria above. | `DESIGN_RECEIPT.md`; this audit |
| `P2R0F3B2B2` | Multivariate linked-cluster ReLU reset oracle | `open` | Produce a target-free theorem and independent verifier for `k<=5`: complete `E5`, all collision classes, frozen `T_k`/tail and out-of-domain law, whole-shell ablation, exact `P2` B2A parity, and worst-entry arithmetic count `<=8192`. | `public_exact_ledger_v1/ledger_rig.py`; `public_trivariate_derivation_v1/DERIVATION.md`; `L5_DERIVATION.md`; this audit |
| `P2R0F3B2B3` | Sector-balanced symmetric low-rank carrier | `blocked` | Unblock only after `B2B2`. Require permutation-equivariant construction, separate `kappa3/kappa4/kappa5` error and future-action spectra, joint and per-order rank gates at widths 5/8/16/32, and a remetered memory/FLOP bound. | future preregistration plus `B2B2` receipt |
| `P2R0F3B2B4` | Dense partition-complete autonomous stability | `blocked` | Unblock only after `B2B2`. With no compression, pass the frozen `n=8,d=4` and `n=16,d=5` multi-seed stability/MSE gates against level C; every layer finite and no hidden sector ablation. | `LEDGER_VERIFY_RESULTS.json` baseline plus future dense receipt |
| `P2R0F3B2B5` | Compressed endpoint and production economics | `blocked` | Unblock only after both `B2B3` and `B2B4`. Retain the preregistered fraction of dense gain, project raw `<1e-8`, and pass independently audited total FLOPs `<2^41` with literal meter parity. | future rank/stability receipts and frozen no-eigh host-cost authority |

Suggested gates are `B2B2 -> {B2B3,B2B4}` and
`{B2B3,B2B4} -> B2B5`. Parent `P2R0F3B2B` remains `open`; only `B2B2` is active.

