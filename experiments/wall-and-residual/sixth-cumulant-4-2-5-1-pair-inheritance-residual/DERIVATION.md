# Raw oriented K42 + K51 pair-inheritance complement

Status: **STATIC REPAIR V3, TARGET-BLIND SOURCE/ACTION PACKAGE — HOLD ALL
RELEASES, NUMERICAL ROWS, AND ESTIMATOR CALLS**.

This package prepares two missing two-coordinate sectors of the marginal
sixth-cumulant push:

\[
K_{42,ij}=\operatorname{cum}(X_i,X_i,X_i,X_i,X_j,X_j),\qquad
K_{51,ij}=\operatorname{cum}(X_i,X_i,X_i,X_i,X_i,X_j),\quad i\ne j.
\]

They are an independent complement to K33.  They are not conditioned on a
K33 result, do not contain K33, and do not alter the K33 package.  Neither
plane is symmetrized: `K42.T` is the complementary K24 orientation and
`K51.T` is K15.

## 1. Frozen authorities and exact Q7 support

The sole host is the board-proven public D4 estimator

`p2r0c1d11_cp3_arena_depth4_full_estimator_v1/estimator.py`

with SHA-256
`6aedce54f2006efcb9d87eeb41aae6fd785d7344534634371c48a7979158342e`.
Its public path is `Estimator.predict -> _predict_corrected`; the public
answer includes the native-tape/J transport correction, not just the closure
host returned by `_predict`.

The independent algebra/resource authority is D29
`K42_K51_STATIC_DESIGN_PRICING_RECEIPT.json`, SHA-256
`bb8a55e3224da56139dc79fa2b65acc4cf8320f0991c7d77da5ec0892ad3b6cc`.
Its literal-cost fields are superseded by
`K42_K51_STATIC_ACCOUNTING_ERRATUM_V1.json`, SHA-256
`63fbf1da297b292b6340c688da3e20b16a850c296e165e05f111b0847534b8c3`,
grounded in the independent HOLD receipt SHA-256
`1d5786631f2923c606d4be53f428f9e6fe88f9418dbf15ecc6abb111a592f7ff`.

D4 has exactly seven Mehler terms `t=0,...,6` and these 17 derivative
addresses:

```text
(0,0),
(0,3),(1,2),(2,1),(3,0),
(0,4),(1,3),(2,2),(3,1),(4,0),
(0,6),(1,5),(2,4),(3,3),(4,2),(5,1),(6,0).
```

Thus every new gathered coefficient has `q=t+a<=12`.  The literal source
materializes only `q=0,...,12`; it introduces no new rho order, derivative
channel, lambda3 plane, or rank projection.

## 2. Centered coefficient sources

For the frozen Gaussian preactivation law, let

\[
c_{p,q,i}=E_G[X_{i,+}^{p}\,\mathrm{He}_q(Z_i)],\quad p=1,\ldots,5.
\]

D4 already supplies `c1,c2,c3` and raw moments through four.  The literal
package constructs `c4,c5` without a second probability primitive.  For
`q<=p`,

\[
c_{p,q}=\frac{p!}{(p-q)!}\sigma^q m_{p-q},
\]

where `m0` is the positive mass, and for `q>p`,

\[
c_{p,q}=p!\sigma^p\phi(\alpha)\mathrm{He}_{q-p-1}(-\alpha).
\]

The only new marginal moment is
`m5=pre_mean*m4+4*pre_variance*m3`.  Equivalently, the tail rows use
`c4[q]=4*sigma*c3[q-1]` for `q>=4` and
`c5[q]=5*sigma*c4[q-1]` for `q>=5`.

Crucially, centering uses D4's **corrected post-ReLU mean** `mu`, not the
Gaussian-base `m1`:

\[
\begin{aligned}
d_1&=c_1-\mu I_0,\\
d_2&=c_2-2\mu c_1+\mu^2I_0,\\
d_4&=c_4-4\mu c_3+6\mu^2c_2-4\mu^3c_1+\mu^4I_0,\\
d_5&=c_5-5\mu c_4+10\mu^2c_3-10\mu^3c_2
       +5\mu^4c_1-\mu^5I_0.
\end{aligned}
\]

Here `I0[q]=1[q=0]` is the full-Gaussian constant row.  It is not a ReLU
support indicator.  For derivative address `a` and Mehler term `t`, every
feature is

\[
D_p[a,t,i]=d_p[t+a,i]\sigma_i^{-a};
\]

the scale is never `sigma^(-(t+a))`.

Using the same 17 D4 weights `omega` for
`1+L3+L4+0.5*L3^2`, the two central moments are evaluated in one batched
kernel:

\[
\begin{aligned}
C_{42,ij}&=\sum_r\omega_{r,ij}\sum_{t=0}^{6}
 \rho_{ij}^t/t!\;D_4[a_r,t,i]D_2[b_r,t,j],\\
C_{51,ij}&=\sum_r\omega_{r,ij}\sum_{t=0}^{6}
 \rho_{ij}^t/t!\;D_5[a_r,t,i]D_1[b_r,t,j].
\end{aligned}
\]

At h1 the same batched call uses only Gaussian channel zero.  At h2 through
h15 it uses the already-populated `fusion_context[16]` after the original D4
reset; `_edgeworth_weights` is never called a second time.  There are one
initial and fourteen full-Q7 sources per sector.

## 3. Exact connected assemblies

Enumeration of all 203 set partitions of six labelled variables leaves 41
singleton-free partitions for either label count.  For K42 it gives

\[
\boxed{K_{42}=C_{42}-v_jK_{4,i}-3v_i^2v_j-8CK_{31}
-12v_iC^2-4K_{12}k_{3,i}-6v_iK_{22}-6K_{21}^2.}
\]

The lower-product coefficients are respectively
`1,3,8,12,4,6,6`; together with the one-block K42 term they account for all
41 labelled partitions.

For K51 the same enumeration gives

\[
\boxed{K_{51}=C_{51}-5CK_{4,i}-15Cv_i^2-10v_iK_{31}
-10K_{21}k_{3,i}.}
\]

Its lower-product coefficients `5,15,10,10`, plus the one-block term, also
sum to 41.  All displayed plane products are elementwise.  The implementation
uses D4's repaired variance, covariance, connected K21/K31/K22, and marginal
K3/K4 from the same reset association.  Both diagonals are set exactly to
zero with `fnp.fill_diagonal(plane, 0.0)` because marginal kappa6 is outside
this design.  FlopScope 0.12 returns a read-only `fnp.diagonal` view, so a
`copyto(diagonal(...))` spelling is invalid and forbidden.  Symmetrization
is forbidden.

## 4. Exact oriented affine actions

Let `W[i,a]` map source coordinate `i` to next-preactivation coordinate `a`.
D4 already holds `W2=W*W` and `W3=W2*W`.  The package forms `W4=W2*W2` and
`W5=W3*W2`, then dispatches both dense actions in one batched matmul:

\[
\begin{aligned}
r^{42}_a&=15\sum_i W_{ia}^4(K_{42}W^2)_{ia},\\
r^{51}_a&= 6\sum_i W_{ia}^5(K_{51}W)_{ia},\\
r^{\mathrm{pair}}_a&=r^{42}_a+r^{51}_a.
\end{aligned}
\]

The coefficients 15 and 6 are the ordered multinomial counts
`6!/(4!2!)` and `6!/(5!1!)`.  Each oriented plane already contains the
complementary role assignment in its transpose entries, so there is no
factor-of-two correction.

All carriers and actions remain raw.  No tau/B normalization and no second
consumer sigma normalization is present.  A future reset, if separately
licensed, would consume the **single** sum `r_pair`; this source package does
not execute that reset.

## 5. Coordinate-only output and no-feedback firewall

The only new scientific coordinate artifact permitted by this package is

```text
combined_raw_k6.npy, exact float64 C-order shape [15,1024]
```

ordered by source h1 through h15 / consumer h2 through h16.  Individual
K42/K51 planes and sector vectors may be inspected transiently by the sink
for fixed target-free checks, but they are not coordinate banks and must not
be retained.  No `raw_k6/720` endpoint, univariate L6 column, bivariate
`(0,6)/(6,0)` weight, mean tangent, training feature, target join, or
autonomous state is constructed.

The wrapper calls the original D4 functions first and reads their outputs.
It does not replace any host result with a recomputed approximation.  At the
full public entrypoint, a fresh-process instrumented call must be byte-equal
to a separate fresh-process frozen D4 call in all five checkpoints:

1. host layer-mean tape;
2. `native_raw` tape;
3. `native_sigma` tape;
4. native-J transport correction; and
5. final max-corrected public prediction.

The incremental bill is instrumented full-public-predict FLOPs minus frozen
full-public-predict FLOPs.  Any future autonomous recurrence must recompute
the native tape and J correction on its changed state; frozen-base J reuse is
forbidden.

## 6. Literal batched ledger

The literal q0..12 implementation removes the 29,420,544-F production
endpoint.  FlopScope 0.12 bills binary64 arithmetic at rate two, resolves the
mixed float64/float32 action matmul as float64, and bills the literal advanced
float64 gathers at eight units per gathered output element.  Isolated
multi-width fits give this conservative dense coordinate-only diagnostic:

```text
coefficient/context = 15*(3078*n) + 14*(3808*n)
sources/reductions  = 80*n^2 + 28*n + 14*(1506*n^2)
connected/diagonal = 15*(44*n^2 + 20*n)
actions/shared sum = 15*(4*n^2*(2*n-1) + 16*n^2 + 2*n)
```

| Component | Incremental FLOPs |
|---|---:|
| shared c4/c5, centered rows, grids/gathers | 101,869,568 |
| two-sector source kernels and reductions | 22,192,091,136 |
| connected assemblies and valid diagonal writes | 692,367,360 |
| two-sector affine actions plus one shared sum | 129,037,793,280 |
| **standalone dense diagnostic total** | **152,024,121,344** |

The independent integrated-association audit proves that official FlopScope
chooses the fused later kernel-plus-reduction path `1492*n^2 + 476*n` (the
explicit path is cheaper than dense `1506*n^2` at width 1024).  The integrated
public-predict increment is therefore predicted as `151,825,424,384` FLOPs,
exactly `198,696,960` below the standalone dense diagnostic.  Neither number
is a brittle runtime-equality gate.  Before any outcome, the runner freezes
the honest hard cap `152,200,000,000`: measured `B-A1` and `B-A2` must be
equal, nonnegative, and no larger than this cap.  Both predictions are
recorded diagnostically.

The operation inventory is 29 source/reduction einsum calls (one initial,
14 later kernels, 14 later reductions) and 15 batched dense-matmul calls
carrying 30 logical sector matmuls.  There are two legal syntactic
`fill_diagonal` sites and 30 runtime calls.  The exact official
instrumented-minus-base call delta for
`empty/add/concatenate/copy/getitem/multiply/reshape/stack/subtract/`
`einsum/diagonal/fill_diagonal/transpose/copyto/matmul/sum` is respectively
`16/165/30/15/43/1020/15/45/210/29/15/30/15/30/15/15`.  This is the
residual-friendly D29 spelling; a sequential 30-dispatch action is forbidden.

Against D4's three official base bills, the integrated predicted totals are

```text
1,471,697,134,967
1,508,420,712,968
1,469,774,383,616
```

so the minimum predicted margin below `2^41` is `690,602,542,584` FLOPs.

The residual-friendly D29 allocation ceiling is 309,092,352 incremental
bytes: a 34-plane fp64 source bank (285,212,672), two persistent oriented
planes (16,777,216), and coefficient/feature/derivative context (7,102,464).
The source bank and both persistent planes are float64.  The action scratch
is also float64, so its mixed input copies cannot silently claim a float32
matmul bill.  The compact q12 implementation is no larger than the stated
simultaneous-allocation ceiling.  The legacy D4-plus-remote-scientific-scratch
projection is 7,043,162,112 bytes, but it excludes the runner's target-free
client diagnostic fetch.  Conservatively allowing 25,165,824 more bytes for
one fetched plane and allocation-light metrics gives 7,068,327,936 bytes,
leaving 1,521,606,656 below 8 GiB; the largest new remote array is
285,212,672 bytes.  These are static projections, not a process RSS or
residual pass.  Every released A1/B/A2 worker must still measure RSS at most
8 GiB and residual below 0.4 seconds.

## 7. Governance and strict deduplication

- This leaf is exactly K42/K24 plus K51/K15.  It excludes K33, marginal
  kappa6, and every three-or-more-coordinate order-six sector.
- It shares only the frozen D4 host and eventual raw endpoint coordinate with
  K33.  There is no K33 source contraction, carrier alias, success
  prerequisite, result reuse, or release reuse.
- It is not a lambda3 plane, a rank/tensor compression, the mixed `(1,2,3)`
  Gaussian shell, or a fifth-order carrier retry.
- Source planes have one-layer lifetime.  The 34-plane bank is scratch; the
  two connected planes are disjoint persistent state until their one action.
- No release file exists.  No rank, row, weights, names, targets, truth,
  score, training, autonomous estimator, or submission is authorized by this
  package.

The V3 runner is the only future transaction authority.  A release must pin
the complete V3 freeze map, and the runner creates and fsyncs an exclusive
rank-specific consumption claim before any row loader; that claim survives
every failure, so retry is impossible.  Rank 643 additionally requires the
release-hash-pinned rank-201 result and atomic commit manifest and validates
its PASS semantics, complete conjunctive gates, release payload, and V3
source/preregistration/freeze association.

For either rank the runner internally launches three distinct official-
environment subprocesses in the fixed order base A1, instrumented B, base
A2.  Each consumes a distinct capability and binds its parent-observed PID,
nonce, role/rank, row, weights, release, runner, and source hashes before
opening the row.  Before any weights load, the row loader also requires the
sealed manifest's exact integer-zero sentinels for `scores`,
`target_values_decoded`, `target_columns_projected`, `fits`, and
`optimizers_constructed`; every worker persists the same sentinel payload and
canonical hash, and the final firewall gate joins that evidence.  Injected
callbacks are not authority.  The runner executes
every scientific proof and live finite/diagonal/nonzero gate, requires byte
parity at all five public checkpoints, exact official operation-call deltas,
equal nonnegative `B-A1` and `B-A2` bills at most 152.2 billion FLOPs, all
totals below `2^41`, all fresh-process RSS values at most 8 GiB, all residuals
below 0.4 seconds, and a zero target/training/selection firewall.

A completed transaction becomes visible only through one atomic directory
rename.  The directory contains exactly `combined_raw_k6.npy`, the terminal
result, and `COMMIT_MANIFEST.json`; the manifest binds result and coordinate
file hashes plus the coordinate dtype, shape, C-order content hash, release
bytes, row/weights, and frozen sources.  Precommit failure removes the staged
coordinate/result while the no-retry claim remains durable.

The accounting authority is sealed without a self-hash cycle.  First the
derivation, preregistration, literal source, runner, and verifier source are
made final.  The independent integrated-accounting receipt binds those exact
hashes and the frozen foundational receipts; the verifier source does not
embed the accounting-receipt hash.  A deterministic verifier replay then
records the receipt hash in `STATIC_VERIFICATION_V3.json`, and
`SOURCE_FREEZE_V3.json` pins both files.  Thus the verifier semantically joins
the receipt to every final input while the later freeze makes the whole
authority set immutable.

A future target-blind discriminator requires a new independent audit and a
hash-pinned singleton release.  Passing such a source/action screen would
license only a separately derived autonomous-reset stability design; it
would not license target access or deployment.
