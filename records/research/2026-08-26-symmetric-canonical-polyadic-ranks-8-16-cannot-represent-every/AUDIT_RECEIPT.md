# B2B3 symmetric-CP carrier prior-art, rank, and cost audit

Status: **CLOSE whole-tensor CP8/CP16; preserve only collision-sidecar
residual CP8/CP16 as a blocked, low-prior conditional form**  
Date: 2026-08-26  
Scope: target-free repository audit, exact rank proof, and static integer
arithmetic only. No target, teacher, network, sample, estimator, submission,
canonical graph, HTML, or external state was opened or changed.

## Decision

| proposed form | disposition | exact reason |
|---|---|---|
| Three complete symmetric CP8/CP16 tensors for `kappa3`, `kappa4`, and `kappa5` | **closed by proof** | The exact reset domain includes independent-coordinate laws whose complete connected tensors are coordinate-diagonal and have mode-1 unfolding rank 1024 at width 1024. CP rank 8 or 16 cannot represent that valid reset exactly. |
| Independent CP8/CP16 blocks for only the three-or-more-distinct residual, with exact collision sidecars and correction seams | **conditional, low prior, blocked** | This avoids the diagonal-rank contradiction and is cheaper than generic TT16, but no sealed covariance-resummed reset/compiler, source-only CP constructor, per-order capacity witness, recurrence, or integrated cost proof exists. |
| One factor bank shared across `kappa3`, `kappa4`, and `kappa5` | **blocked stricter ablation** | It additionally forces all three mode-1 column spaces into one rank-8/16 span. Test it only if independent per-order factors first pass. |

The representation family is not new. Symmetric CP carriers for third and fourth
cumulants have been implemented and tested extensively. The materially new
piece would be a partition-complete fifth-order residual carrier reset from a
future covariance-resummed compiler. No repository implementation of a true
`kappa5` CP carrier was found. That negative search is scoped to this repository;
it is not a claim about literature outside it.

The current `B2B2` exact-oracle wording is stale. `IG-LC3` has now been closed
as an exact or production evaluator: its finite algebra is valid at declared
grade three, but it misses the leading Gaussian `[11111]` covariance shell and
is inaccurate at the reached `rho=0.54` anchor. A carrier may consume a future,
independently sealed covariance-resummed or tail-certified reset/compiler. It
must not use the carrier's own `<=8192` query premise to establish that compiler;
that would be circular.

## Exact whole-tensor rank obstruction

For order `r` in `{3,4,5}`, consider a valid independent-coordinate input law
whose marginal cumulants of order `r` are all nonzero. Coordinatewise ReLU
preserves independence, so its complete connected order-`r` reset is the
coordinate-diagonal tensor

```text
D_r = sum_(i=1)^n d_i e_i^tensor(r),    every d_i != 0.
```

Its mode-1 unfolding has row `i` supported at the unique column indexed by
`(i,...,i)`, with value `d_i`. The `n` rows therefore have distinct pivots and

```text
rank(unfold_1(D_r)) = n.
```

For any CP representation

```text
T_r = sum_(a=1)^R lambda_a u_a^tensor(r),
```

the same unfolding is a sum of `R` rank-one matrices, so its matrix rank is at
most `R`. Exact representation of a generic diagonal tensor thus requires
`R >= n`. At `n=1024`, both `R=8` and `R=16` fail before any mixed-index entry is
considered.

This is a counterexample to a universal exact complete-state CP8/CP16 reset
operator. It does not prove that every particular reached correlated tensor has
unfolding rank above 16, nor that low CP rank cannot approximate one in a chosen
weighted action norm. It does prove that the unqualified complete-state spelling
is not partition-complete and cannot claim exact collision replay over the
declared reset domain.

`verify_audit.py` checks the mode-1 rank by exact rational elimination at
`n=5` for orders 3--5 and by the unique sparse-pivot identity at `n=1024`.

## Only coherent residual partition

Let `P2` retain entries with at most two distinct coordinate indices and let
`Q2=I-P2`. A coherent candidate may store a symmetric CP tensor `G_r` plus an
exact collision correction `S_r`:

```text
S_r = P2(kappa_r) - P2(G_r),
kappa_r_hat = G_r + S_r,
P2(kappa_r_hat) = P2(kappa_r),
Q2(kappa_r_hat) = Q2(G_r).
```

The visible collision fields are:

```text
kappa3: diagonal and K21
kappa4: diagonal, K31, and K22
kappa5: diagonal, K41, and K32.
```

This identity is not by itself a deployable sidecar. After a generic affine
map, a collision-supported tensor generates all-distinct action. Each `S_r`
therefore needs an exact represented affine-action law, analogous to the current
H3/H4 residual seams, and must be added and subtracted exactly once at each
consumer/reset boundary. The CP-only arithmetic below excludes the storage,
transport, reset, and stability cost of those collision corrections.

The nonlinear reset/compiler is a logically prior dependency. It must define
the requested post-ReLU cumulant entries and error law independently of the CP
fit. Only then may a deterministic CP constructor query it, fit `Q2(G_r)`, and
form the collision correction above.

## Repository prior art and measured boundaries

### Current Phase-2 carriers

The current full-budget architecture already stores:

- a third-order `sym3(P,P,Q)` carrier capped at 16,384, reaching 15,360 births;
  and
- a signed pure symmetric CP4 carrier capped at rank 2,048.

Every `sym3(P,P,Q)` atom converts algebraically into three pure cubes. The
15,360-birth state therefore has a pure-cube representation with at most 46,080
atoms. Rank 8 or 16 is not a mild reduction of the incumbent third-order state.

The measured CP3 resource ladder costs 229.196B, 336.972B, 500.555B, and
620.091B FLOPs at caps 2,048, 4,096, 8,192, and 16,384. The final rung is exact
no-truncation for the fixed depth. The CP4 cap-2,048 carrier has a 234.508B
audited add-on and 8,396,800 persistent bytes; its top-energy rule is explicitly
not a best-rank theorem.

At layer 6, expanding CP4 to cap 8,192 so no truncation occurred through that
point recovered `-0.0631691` of the observed quartic deficit. A separate rank-4
K22 residual carrier recovered only `0.178543`. More CP4 history alone did not
identify the missing quartic action.

### Third-order capacity versus source supply

At width 256:

- an exact rank-192 pure-cube bridge narrowly missed its action gate at 11.2831%
  pooled and 17.6417% worst error;
- adding target-guided residual columns to rank 384 passed at 0.00563% pooled
  and 0.01257% worst action error; but
- the legal source-only rank-384 constructor missed actual action by 30.4828%,
  and even its untruncated rank-6,912 legal source ceiling missed by 30.0029%.

Thus CP can have oracle capacity at hundreds of atoms while the causal
construction and update supply the wrong information. A width-16 target-guided
rank-12 control and a small local rank-8 residual witness do not establish
width-1024 recurrence.

The closest low-rank action test is the frozen source-blind fused CP/TT16
construction. It fit its meter at 172,799,232 operations per layer but failed
all eight action rows, with errors approximately 0.575--0.947 against a 0.15
gate while every incumbent row passed. Exact K21 overlay still agreed to
`4.16e-17`. This closes that physical 28-atom action-fiber regression, not all
possible residual CP16 constructions, but it is strong negative evidence
against CP16 as a generic action-preserving escape.

### Fourth-order low-rank sequence

- Diagonal CP4 retained 77.89% of the oracle mean gain but produced raw
  covariance minimum eigenvalue `-0.008273` because its mixed planes were wrong.
- One full-support rank-one CP4 had relative `B/d/A` errors
  `0.6822/0.9531/1.3420` and raw MSE `7.6665e-6`, worse than its predecessor.
- A source-derived rank-24 CP4 halved selected deep A-plane errors, then regressed
  the sealed width-256 endpoint by 0.0800% and became nearly axes-only at
  deployment width.
- Pair-local, leverage, dyadic, layer-cake, action-residual, and transition
  pullback rank-24 successors produced source-local gains but no deployable
  endpoint gain. The final DCT-Ritz24 successor ended 0.00816% worse.
- A separate sparse symmetric CP24 discriminator failed both TT4 action rows at
  about 1.08 against a 0.20 gate.

### Shared-factor adjacent evidence

A joint rank-8 positive-state fit was authority-inconclusive: the added
amplitudes collapsed or the solver exhausted its cap. Common-support positive
cubature is more constrained than signed per-order CP, but its width-256 M285
form lacks dimension for generic K21, and a much larger action-adaptive cubature
passed reset 1 then failed reset 2. No artifact uses one signed factor bank with
separate coefficients for all of `kappa3`, `kappa4`, and `kappa5`.

## Conservative CP-only width-1024 arithmetic

Assumptions:

```text
n                         1024
affine pushes             15
nonlinear resets          14
float32 factor storage    4 bytes/scalar
old entry-cost comparison 8192 FLOPs/query
```

For rank `R`, one factor push `W^T U` costs
`n R (2n-1)`. Independent order blocks require three pushes; a shared factor
bank requires one. Exact visible CP collision reads require five rank-`R` plane
GEMMs per reset: one for K21, two for K31/K22, and two for K41/K32. Their exact
dot count is `5 n^2 (2R-1)` per reset.

| item, all 15 pushes / 14 resets | R=8 | R=16 |
|---|---:|---:|
| one factor push | 16,769,024 | 33,538,048 |
| independent three-order pushes | 754,606,080 | 1,509,212,160 |
| shared-factor pushes | 251,535,360 | 503,070,720 |
| five visible plane reads | 1,101,004,800 | 2,275,409,920 |
| independent carrier subtotal | 1,855,610,880 | 3,784,622,080 |
| shared carrier subtotal | 1,352,540,160 | 2,778,480,640 |
| independent persistent bytes | 98,400 | 196,800 |
| shared persistent bytes | 32,864 | 65,728 |

For local identifiability, independent symmetric blocks have approximately
`3nR` continuous degrees of freedom per reset. A shared unit-direction block
with three amplitudes per atom has `R(n+2)`. Pricing one oracle query per degree
of freedom at the old 8,192-FLOP cap gives this optimistic comparison:

| proxy, all 14 resets | R=8 | R=16 |
|---|---:|---:|
| independent queries/reset | 24,576 | 49,152 |
| independent query work | 2,818,572,288 | 5,637,144,576 |
| shared queries/reset | 8,208 | 16,416 |
| shared query work | 941,359,104 | 1,882,718,208 |
| independent carrier + query proxy | 4,674,183,168 | 9,421,766,656 |
| shared carrier + query proxy | 2,293,899,264 | 4,661,198,848 |

These are not executable production bounds. They omit the CP fitting solver,
Jacobian/ALS/QR work, restarts, error certification, collision-correction state
and transport, reset compiler, stability machinery, casts, dispatch, and literal
FlopScope behavior. The former 8,192 figure is used only to compare arithmetic
scales; the closed IG-LC3 receipt means it is not a live oracle gate.

For context, the now-closed generic TT16 design priced 25.203B affine transport,
322.437B capped entry work, 10.076B cross/QR, and 377.716B total increment. The
CP route has a real cost ceiling if low residual ranks and a sparse deterministic
constructor exist. Neither premise is established.

## Conditional reopen sequence

No implementation or experiment is licensed by this receipt. Reconsider only
in this order:

1. Seal a materially different covariance-resummed or independently
   tail-certified reset/compiler with no carrier-dependent proof loop.
2. At small dense width, verify the `G_r + S_r` partition, permutation symmetry,
   collision replay, and affine action separately for orders 3, 4, and 5.
3. Run synthetic/dense-reference capacity gates for independent ranks 8 and 16,
   reporting per-order tensor error, per-order suffix-action error, and collision
   leakage. No aggregate score may hide a failed order.
4. Only after capacity passes, test a deterministic source-only constructor and
   autonomous multi-reset stability.
5. Test shared factors only after the corresponding independent blocks pass.
6. Recompute the integrated production ledger, including collision sidecars and
   the actual reset/compiler, and require literal meter parity below `2^41`.

Given rank-384 third-order oracle evidence, the fused CP16 action failure, and
the rank-24 fourth-order endpoint failures, the residual CP8/16 hypothesis has a
low prior. Its value is as a cheap falsifier after the reset dependency exists,
not as a current production candidate.

## Bound evidence

| artifact | SHA-256 | authority used |
|---|---|---|
| `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v1/DESIGN.md` | `5c3ae5b91a32f4216b8739e2d2013eb2cadbae64a0430c0aa7e71b3f15e2046f` | active CP3/CP4 state, update, and read laws |
| `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/FULL_BUDGET_CP4_CAP2048_RESIDUAL_CARRIER_AUDIT.md` | `9f044c2665c2143413e5b7da398487c38545994b20ff177f3494c004b0811588` | CP4 algebra, exact ledger, and seam limitations |
| `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/cp3_full_birth_cap_ladder/SYNTHETIC_RESOURCE_LADDER.json` | `9fa0588a008f1a9934dc440cff7afb3e023bf6ff28c1fb72c84bae9e1f1126e6` | measured CP3 cap costs |
| `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_oracle_kappa_probe_v1/ORACLE_KAPPA_VARIANTS.json` | `48cab175c5f06ac3024c070e7cf116455666f86da9c18deafbdb28b9fe7b4d73` | cap-8192 and rank-4 K22 recovery controls |
| `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0f3b2b_partition_complete_tt5_design_v1/TERMINAL_DESIGN_AUDIT.md` | `ddfdfc64dec12efb5dbc7b96384e6087bc0eaad19a28a44e9d2eccd2839c16e6` | missing reset, per-order carrier, and symmetry requirements |
| `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0f3b2b2_ig_lc3_reset_derivation_v1/DERIVATION_RECEIPT.json` | `41a9045df1105671fd45645142db2721cec13fba3e3e35c941c5c7bdabd50703` | formal IG-LC3 algebra and closed generic-TT disposition |
| `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0f3b2b2_independent_gaussian_linked_cluster_grade3_v1/CONSTRAINT_RECEIPT.md` | `d6275c2ace74294c4d9c6fcc6b5126dbbb9725e4dac51aeca1affba0e98c7cba` | independent IG-LC3 accuracy, collision, and cost obstruction |
| `pure_closure_writeup/problem_graph/experiments/x3_all_distinct_cp_width_scaling/README.md` | `643ecaa2e37118f5cb82914112b94a038ce63e49ca8bfbb6d45803367c29dfc1` | width-256 rank-192 action bridge |
| `pure_closure_writeup/problem_graph/experiments/x3_all_distinct_cp_residual_capacity/README.md` | `489fac12e2097889469937d95bda3256f3b76b628f3e3dc84aaa5685d62e719c` | width-256 rank-384 oracle capacity |
| `pure_closure_writeup/problem_graph/experiments/x3_all_distinct_cp_source_constructor/README.md` | `6eaa2f1deb5aedc1f300ae481d993a2cd6edb668841fb6b789cce56ce699d03f` | source-only supply and economics failure |
| `pure_closure_writeup/problem_graph/experiments/x3_rank12_cp_capacity/README.md` | `483d0b54df0ffa73d75a04107dedac49c31cd71908efca39904b4ccf2f2854c4` | local unrestricted rank-12 capacity control |
| `pure_closure_writeup/problem_graph/experiments/x1_residual_cp_dimension/README.md` | `cb6e5ac94fc8593433fa46ee4f0c68217836adb850733c2339bd28f364c904dc` | local residual rank-8 witness |
| `pure_closure_writeup/problem_graph/experiments/x1_joint_rank8_positive_capacity/r2/README.md` | `d98f2e4aeadd8676f7b618c8dfaa63a715cb4b620c24f0a8eae7de87e09d1ad4` | shared/common-direction adjacent evidence |
| `pure_closure_writeup/problem_graph/experiments/x3_r0c1_kappa3_ceiling/fused_two_channel_h_action_cp_rank16/RESULTS.json` | `22d11a68c143f84f53b52fb2efff899f03e4613f8ed0f4c5db561f5b395ab44d` | fused CP16 action failure |
| `pure_closure_writeup/problem_graph/experiments/x3_r0c1_kappa3_ceiling/fused_two_channel_h_action_cp_rank16/METER.json` | `32eef43223f4fdc11786369699de2e9fee924e4387feb11fc6759852f9a1ff98` | fused CP16 complete meter |
| `pure_closure_writeup/problem_graph/experiments/x3_r0c1_kappa3_ceiling/diagonal_cp4_rolling_action/DERIVATION.md` | `cc09105940a42a7f1ee917da6045beb108f9bdfae7c2865ad6457332f27573dc` | diagonal CP4 action and physicality failure |
| `pure_closure_writeup/problem_graph/experiments/x3_r0c1_kappa3_ceiling/rank1_pairing_cp4_escape/RESULTS.json` | `17e92d18132a376a21c562a258d49e897e4f90292805acdb47e1708cb023a332` | rank-one CP4 capacity failure |
| `pure_closure_writeup/problem_graph/experiments/x3_r0c1_kappa3_ceiling/coherent_rank24_carrier_cp4_k31_provider/FINDING.md` | `7dd37b112f7655dba6df4e1fd801da02506acfa60facc8dacc582139b44df148` | rank-24 source gain and endpoint failure |
| `pure_closure_writeup/problem_graph/experiments/x3_r0c1_kappa3_ceiling/sparse_polynomial_semiring_tt4_rank24/RESULTS.json` | `e85cd3d4200c9ee6e79c64bfd02b8241c49edaedffd57d84352fa21c7bbf00bd` | symmetric CP24 action failure |

## Verification

Verifier: `verify_audit.py`  
Verifier SHA-256:
`9a63de324815654f5757ace4de60434ce674f1549915c57c0188863a15fa38ae`

From the repository root:

```text
python3 pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0f3b2b3_symmetric_cp_residual_prior_audit_v1/verify_audit.py
```

Expected terminal status:

```text
status: PASS
source_hash_count: 19
source_hashes_match: true
universal_exact_whole_tensor_cp8_cp16_possible: false
failures: []
```
