# P2R0F3B2B successor design receipt: partition-complete TT5 cumulant jet

Status: **DESIGN ONLY — NO IMPLEMENTATION OR TARGET AUTHORITY**  
Date: 2026-08-26  
Selected successor count: **exactly one**

## Decision

The one surviving high-ceiling representation break is a **partition-complete,
rank-16 tensor-train cumulant jet through order five** (`PC-TT5`). It carries
the complete connected third-, fourth-, and fifth-order state in one augmented
tensor train, applies the affine push exactly, computes every lower-state and
fifth-order collision class from one common nonlinear reset, and truncates the
whole jet once rather than inserting selected K5 planes into an otherwise
incomplete trajectory.

This is a conditional research candidate, not a claim that rank 16 is enough.
Its rank premise is the first fail-fast gate. A generic fifth-order tensor does
not have low TT rank.

No development target, `observer_fit`, `observer_selection`, identity, public
name, board row, estimator, or submission surface was opened while producing
this receipt. No canonical graph file was edited.

## Why this is the only successor retained

The current bank is submission `#328439`, raw
`2.9409159800763974e-8` and adjusted `2.2842888692472472e-8`. Raw `<1e-8`
therefore needs a `2.9409159800763974x` MSE reduction, or removal of
`65.99698846296103%` of endpoint SSE.

The post-Fable evidence gives a narrow specification:

1. True-state teacher forcing reaches `8.207963477637236e-9`, so the existing
   local mean consumer has enough capacity when its hidden trajectory state is
   correct. The missing ceiling is autonomous state reconstruction.
2. The final-site sensitivity receipt assigns the dominant error to the
   propagated mean trajectory, not a last-layer missing column: its signed mean
   share is `1.9627`, while k3 and k4 last-site shares are only `0.1096` and
   `0.0501`.
3. Direct K5/K7/K9 marginal columns are 25--50 times too small and have
   `|corr| <= 0.114`. A local scalar source is not the state.
4. Complete K22 supply was the large host improvement, whereas all remaining
   order-`<=4` supply, the crossed `2F` route, kappa211 planes, a corrected-frame
   microhead, and scalar/observable correctors are submaterial.
5. B2A proves that **partial** autonomous fifth-order carry is actively unsafe:
   on its two frozen n=16,d=5 cases, level-C MSEs
   `0.0019371902666675653` and `0.0889081513905304` become
   `0.11720815944570878` and `644.8317168674707` with diag/K41/K32 carry, and
   `92.51253102041694` and `73436.32971540386` at the next partial rung.
   One-step improvement alongside autonomous explosion is evidence for missing
   coupled sectors/cancellations, not evidence for another isolated plane.
6. B1A proves reverse transport is the same first-order Duhamel operator to
   `1.9244557334329387e-16`; its K12 ladder is catastrophic, and a literal
   degree-5 scalar basis has `9,474,438,804,480` coefficients. Direction or
   basis relabeling supplies no information.
7. Deterministic quadrature/sampling hybrids, fixed-tape observers, low-rank
   covariance features, scalar gains, and output ensembles have measured
   ceilings far below the required 66% SSE removal.

`PC-TT5` is materially different from every item above: it changes the carried
state, includes all fifth-order index partitions, updates k3/k4/k5 together,
and feeds the jointly reset state forward. It is not a mean-only correction,
not an observer tape, and not a backward spelling.

## Exact state definition

Let `n` be network width and introduce a sentinel coordinate `0`, so
`n_bar = n+1`. Define a symmetric augmented order-five tensor
`J_l in R^(n_bar^5)`:

- a slice with exactly three non-sentinel indices stores the complete
  connected `kappa_3` entry;
- a slice with exactly four non-sentinel indices stores the complete
  connected `kappa_4` entry;
- a slice with five non-sentinel indices stores the complete connected
  `kappa_5` entry;
- slices with fewer than three active indices are zero; mean and covariance
  remain explicit dense state.

All sentinel permutations store the same symmetric cumulant entry. `J_l` is
represented by five TT cores with ranks `(1,R,R,R,R,1)`, frozen at `R=16` for
the production hypothesis. The complete state is

```text
S_l = (m_l, Sigma_l, TT16[J_l]).
```

There is no learned amplitude, scalar damping, target-fitted basis, or
per-network rank selection. The TT gauge is fixed by deterministic left/right
orthogonalization and positive-pivot sign conventions.

### Affine operator

Extend a layer matrix `W_l` by a fixed sentinel:

```text
W_bar = [[1, 0],
         [0, W_l]].
```

Apply `W_bar` to the physical index of each of the five TT cores. This pushes
every active order by the correct number of copies of `W_l`; the sentinel
copies remain unchanged. Mean and covariance use their ordinary exact affine
pushes.

### Nonlinear reset operator

At one ReLU reset, define cumulant differential operators `Lambda_3`,
`Lambda_4`, and `Lambda_5` from the **whole** incoming jet. The declared
order-five closure measure is

```text
E5 = I + Lambda_3 + Lambda_4 + (1/2)Lambda_3^2
       + Lambda_5 + Lambda_3 Lambda_4 + (1/6)Lambda_3^3.
```

For every requested raw post-ReLU moment of total coordinate multiplicity at
most five, one analytic entry evaluator applies `E5` to the Gaussian reference
using the existing Hermite `c[p,q]` identities, then converts all raw moments
to connected cumulants by the ordinary set-partition/Mobius formula. It must
enumerate every fifth-order collision class

```text
[5], [41], [32], [311], [221], [2111], [11111]
```

and update `m`, `Sigma`, `kappa_3`, `kappa_4`, and `kappa_5` from that same
operator. Omitting any class reduces this to B2A and fails equivalence.

A deterministic two-sweep TT-cross queries this analytic tensor-entry map and
then performs one rank-16 TT round on the augmented jet. The pivots and tie law
must be frozen before any reference comparison. TT-cross here interpolates an
analytic state tensor; it does not evaluate network-input samples and is not a
quadrature estimator of the endpoint.

The entry evaluator must first be algebraically reduced to at most `8,192`
counted arithmetic operations per queried entry. Failure to prove that bound
closes the design before a width-1024 implementation. This receipt does not
claim that reduction has already been implemented.

## What is proved, and what is conjecture

### Proof-level statements

1. **Affine rank preservation.** Modewise multiplication maps a TT to a TT of
   the same ranks. For invertible `W`, every TT unfolding rank is preserved,
   because its unfolding is multiplied on the left and right by invertible
   Kronecker factors. Dense mixing neither causes rank growth nor magically
   creates compression.
2. **Partition coverage.** An augmented jet containing all entries of
   `kappa_3`, `kappa_4`, and `kappa_5` contains every repeated- and
   all-distinct index sector. B2A's diag/K41/K32 state is a strict projection
   of it.
3. **Trajectory non-equivalence.** A nonlinear reset of the entire jet followed
   by forward feedback is not a Duhamel correction and cannot be obtained by
   transposing B1A's frozen mean tangent.
4. **Conditional compression.** If all four TT unfolding ranks are at most 16,
   the augmented tensor has at most
   `(n+1)*(3*R^2+2*R)` stored scalars. At `n=1024,R=16`, this is `820,000`
   float32 scalars (`3,280,000` bytes), versus
   `9,474,438,804,480` homogeneous degree-5 coefficients. Thus a certified
   TT16 structure evades the literal degree-5 storage/count bound by about
   `11.55 million x`.
5. **No generic escape.** A generic order-five tensor has middle TT unfolding
   rank `n^2`. TT16 is a measure-zero structured family, not a universal
   representation. The B1 rank bound is not disproved; it becomes a rank
   falsifier for the actual reached states.

### Conjectures requiring the discriminator

1. The quenched ReLU trajectories reached by this problem have action-relevant
   augmented-jet TT ranks at most 16, even though generic tensors do not.
2. Joint projection preserves the cancellations whose absence destabilized
   B2A.
3. Complete fifth-order state improves the autonomous k3/k4 trajectory enough
   to inherit the teacher-forced `<1e-8` capacity.
4. The analytic entry evaluator can be reduced below the frozen per-query cost
   without a forbidden high-dimensional quadrature or hidden dense tensor.
5. Small-n rank and accuracy transfer to width 1024. No annealed or `1/n`
   transfer is assumed.

## Width, depth, memory, and strict FLOP envelope

For fixed TT rank `R` and order `d_j=5`:

```text
state                 O(n R^2)
affine push           O(D n^2 R^2)
two-sweep TT-cross    O(D d_j n R^2 * C_entry)
cross orthogonalize   O(D d_j n R^3)
```

At the frozen production shape `n=1024`, `n_bar=1025`, `D-1=15`, `R=16`:

| item | strict design count |
|---|---:|
| TT core scalar slots, `(n_bar)*(3R^2+2R)` | `820,000` |
| five-mode affine transport | `25,202,700,000` |
| analytic entry queries, `2*15*5*n_bar*R^2` | `39,360,000` queries |
| query work at the hard `8,192`-FLOP cap | `322,437,120,000` |
| two-sweep TT-cross/QR upper, `2*15*16*5*n_bar*R^3` | `10,076,160,000` |
| collision reinstall, connected conversion, checking, output allowance | `20,000,000,000` |
| **maximum PC-TT5 increment** | **`377,715,980,000`** |

The largest trim-card literal path used `2,023,256,192,375` FLOPs. The no-eigh
card removes exactly the unconditional 14-call eigendecomposition block billed
at `270,582,939,648`, yielding a conservative frozen-host bound of
`1,752,673,252,727`. Therefore

```text
host + PC-TT5 <= 2,130,389,232,727
2^41            = 2,199,023,255,552
integer margin  =    68,634,022,825
```

This is a hard design envelope, not permission to spend it. Before any
production-shape science run, an independent static audit must prove the
`8,192` entry bound and the no-eigh host subtraction, and a literal meter must
remain below the same total. Any extra eigensolve, third cross sweep, rank
increase, hidden dense `n^3+` tensor, or unpriced special function is forbidden.
The added persistent TT state is 3.28 MB; total temporary workspace is capped
at 512 MiB, retaining ample room under the 8 GiB process limit.

## Smallest leakage-safe fail-fast discriminator

The discriminator is sequential. A failed stage stops all later stages. It
uses only synthetic Gaussian-input networks and exact/synthetic references;
it never opens development rows.

### D0 — algebra and non-equivalence, n=4

- Frozen weight seeds: `51031, 51047`.
- Materialize dense `kappa_3..kappa_5` and compare the augmented affine push
  with five explicit tensor contractions, relative error `<=1e-12`.
- Compare the analytic reset entry oracle against dense enumeration for all
  multiplicity classes `[5]` through `[11111]`, relative error `<=1e-10`.
- Require exact K4-level reproduction when `Lambda_5=0` and exact exposure of
  B2A when all three-or-more-distinct fifth-order sectors are zeroed.
- Verify that the candidate differs from both the mean-only Duhamel output and
  B2A on a nonzero all-distinct fixture.

### D1 — does complete order five cure B2A before compression?

- Reuse the exact-ledger network law at `n=8,d=4` and `n=16,d=5` with new
  frozen family seeds `51059, 51071, 51089`; antithetic reference seeds are
  `151059, 151071, 151089`.
- Run the dense, untruncated partition-complete order-five jet. Rank compression
  is forbidden in this stage.
- Required on **every** seed: finite state, no covariance-cone failure, final
  level-C MSE divided by dense-jet MSE `>=4.0`, and no intermediate-layer MSE
  more than `2x` level C.
- If dense complete carry fails, classify
  `CLOSE_ORDER5_HIERARCHY_NOT_SUFFICIENT`; do not blame TT rank and do not open
  targets.

### D2 — rank and action spectrum

- Widths `n={8,16,32}`, matched depth 4, three coupled weight families above.
- Compute exact dense unfoldings at n=8/16. At n=32, use a deterministic
  matrix-free unfolding SVD whose products are checked against sampled exact
  entries; no endpoint target is used.
- Report ordinary Frobenius spectra and **future-action spectra** after each
  remaining affine suffix. Let `r_99` be the smallest rank retaining 99% of
  suffix-action energy.
- Gate: `r_99 <=16` at every layer/width/family; the median log-slope of `r_99`
  versus width must be `<=0.25`; TT16 cross residual must be `<=0.10` in
  suffix-action norm. Two widths are not accepted as an asymptotic law.
- If this fails, classify `CLOSE_FACTOR_RANK_DOES_NOT_EVADE_B1_BOUND`.

### D3 — compressed autonomous endpoint

- On the same frozen cases, TT16 must retain at least 85% of the dense jet's
  MSE reduction, improve level C by `>=3.0x` on every case and by `>=3.5x`
  geometrically, remain finite, and have no layer with correction/state norm
  over `4x` the dense-jet counterpart.
- The conditional Phase-2 projection is then
  `2.9409159800763974e-8 / 3.5 = 8.402617085932564e-9`. This is only a
  plausibility projection, never a result.
- A pass licenses only a new production-shape preregistration and independent
  cost audit. It does not license development-target access or submission.

## Explicit audit of the suggested intermediate-layer pathwise control variate

The proposed form

```text
B*mhat_k + mean_s[x_16(s) - B*x_k(s)]
```

is **rejected as this successor**.

Its expectation is

```text
E[estimate] - E[x_16] = B*(mhat_k - E[x_k]).
```

For `k>1`, the analytic intermediate mean has exactly the compounded trajectory
bias this program is trying to remove; a suffix CV does not cancel it. For
`k=1`, exact ReLU-Gaussian means and selected quadratic moments can remove that
bias, but the method is then a sampling/control-variate estimator, explicitly
outside the requested representation class. It also lacks the necessary
variance ceiling: at the full sampling budget the quadrature receipt gives
plain MC about `1.019e-6` on the reference row, any degree-2 oracle
`6.452e-7`, a perfect 512-dimensional active subspace `3.763e-7`, and even a
free degree-64-exact surrogate `2.850e-8`. Raw `1e-8` would require at least
about `99.0%` capture. If composed with the no-eigh host, only about 446B FLOPs
remain—roughly 13.3k full forwards rather than 65.5k—so the variance condition
is stricter still. A future exact intermediate-mean identity plus independently
measured `>99%` suffix capture would be new evidence; neither exists now.

Low-rank quadratic x1 features are not proved impossible, but existing
measurements are too far from the required capture to make them the single
highest-ceiling next representation, and choosing them would repeat the closed
sampling/quadrature class.

## Deduplication and firewall

This receipt does not reopen or rename:

- fixed-tape J/P/Q/R/M/N observers or another target-fitted residual model;
- N's rank-16 covariance feature axis (TT16 here is a full dynamic cumulant-jet
  bond rank, not a rank-16 output feature);
- order-`<=4` supply, K22 `2F`, kappa211, discarded reset residuals, or scalar
  gains;
- deterministic quadrature, RQMC, pathwise CV, or a sampled suffix;
- reverse Hermite/Duhamel/Onsager notation;
- B2A's diag/K41/K32-only fifth-order state;
- a literal full degree-5 coefficient table.

The first forbidden access or failed gate terminates this design. There is no
retune, seed replacement, rank increase, target-based pivot selection, or
fallback arm.

## Bound evidence (SHA-256)

- `FABLE_HANDOFF_20260825.md` —
  `0e0eea364c1849a4739d8041f7fa9ef83684c1378f739981ddeacebb46497bf2`
- B1A `RESULT.json` —
  `785d789a9ce588989f6ab070d84c6dfdf8ec4faaacc3eabf1bbe791fb49b70e6`
- B1 independent `AUDIT_RECEIPT.json` —
  `450378882b0fe75964cc1378fdd17b52d320ef5f80aba3ddd9bf1bb83cb761cb`
- B2 `LEDGER_VERIFY_RESULTS.json` —
  `f14b4964ab6d7da4917d4c4a77c08099477a840849819cc61ee46da0d842c938`
- `LEDGER_STEP1_RESULT.json` —
  `22d6c18b7ee450bdc669f77fb9ba83ad6b60e48fa42365072623cf4e17647382`
- `FINAL_SENSITIVITY_RESULT.json` —
  `870bbb6731c0b8b5fbe5be26fd042d5c2a39e2c8b3b97d4b81a1b602a4fdc6c9`
- `MICROHEAD200.json` —
  `db614bc2053b62c07556b3989442f32b7e7a31efa636e03b371d5bf7f285af15`
- `MISSING_ROUTES_RESULT.json` —
  `ca559f11483c8a54e4bc5a43739eb56eef05d263ea4328271f043306c94a8d33`
- `QUADRATURE_VERDICT.json` —
  `fd99ba5597293a2e10c98602acc884c498bf761faf8c8100ad44ad2743dbd41f`
- teacher-forced `RESULT.json` —
  `fe2f2f3c8cfa890d416b520f443fbbe475ea8fabafbd16956006a9154b100440`
- trim-card `METER_RESULT.json` —
  `bd2b33754585a4c5ffcd6877d7f7956f135a2f743c5dd9c96593b3b19569b006`
- no-eigh `BUILD_RECEIPT.json` —
  `3fd515406b6eec954523e7d553aba769999998318fb29fd987ce4637e2bbb0d3`
- no-eigh terminal board log —
  `6f2339963c1272d9eee46e20897b06fc87f5b321126e2e1976a565df71f3c781`

## Terminal design verdict

`DESIGN_READY_FOR_TARGET_FREE_FAIL_FAST_ONLY`.

The next legal action, if separately authorized, is D0 only. No heavy run,
development-target access, graph mutation, or submission follows from this
receipt.
