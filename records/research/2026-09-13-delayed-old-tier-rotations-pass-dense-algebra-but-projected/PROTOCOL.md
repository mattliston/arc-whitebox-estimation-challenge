# DRAFT: fixed two-join lazy factor rotation on BOX

Source preparation only. No candidate has been materialized or imported, and no
model, array, target, fit, primitive test, profile or numerical stage has run.
All c2/BOX sources and receipts are preserved. Root must read these new owners
and complete the actual metadata/API preflight before any test is authorized.
This is a modest component, not evidence of45s, a raw improvement, or sub3.

## Fixed ownership and interface

The sole parent is BOX `rpc37_v1/estimator.py`, SHA
`378092f76102b83ddcff445adaf27309ae8136d1df6844599dad59b69cd75dfa`.
Its actual cold/warm37 F was407,741,701,711/404,454,713,141, with exact c2
all16 output bytes but a NEGATIVE runtime objective51.2290/50.6024s. That
negative45s result stays negative; no qualification is inherited by this source.

- `fnp.py` is a no-import numerical appendix, with no tensor work at module entry.
- `compose.py` exposes `authenticate(tick=None)` -> flat direct source pins and
  `derive_source(tick=None)` -> full source text plus JSON-native proof. It never
  writes, imports or executes that estimator. Root owns any future materialization.
- `price.py` exposes `calculate(tick=None)`. Only its explicit call/main reads
  current scalar JSON and prints a conditional price. It imports no estimator,
  NumPy, FNP or Torch and opens no array/target. It creates no files or stages.

The composer reuses the byte-pinned BOX composer's stdlib read/AST-span helpers,
not its candidate builder or ancestral admissions. Six reversible source splices
alter exactly three original definitions: state, push factor handling, and read
inner handling. The entire parent is recovered byte-for-byte by the positional
inverse after removing the one exact appendix. Original main/physical loop,
matrix kernels/dispatch, every source literal/getter/binding, shared commit and
finite helper remain byte-identical. Added names are private `_slr_`/`_SLR_`.

Fixed rank384, age4 and weighting lambda-squared do not change. The only schedule
is lazy joins L8/10/12/14, consolidation L9/11/13, final L14 map through L15.
L5..7 retain materialization. L15 still has the original full repeated-index
read; its join was already absent. No other schedule, rank or age is implemented.

## Real-arithmetic representation invariant

When materialized, `state.F` is the original r-by-R factor bank and lazy is None.
At a lazy join, let L=Qnew.T@Qold and let F0 denote the prior factor bank. Store
two ordered groups, `(L,F0)` and the newly computed Fj, with `state.F=None`.
The virtual canonical factor matrix is

    Fcan = concat(L F0, Fj).

The original Qnew QR, the original Fj projection, and the ORIGINAL canonical
core update `Snew=L(Sold L.T)+Fj D Fj.T` remain at every join. No QR is skipped;
no coordinate inverse, normalization, solve, rank deletion, lambda modification
or new clipping is added. In finite arithmetic S is the original separately
updated core; no exact FP32 Gram-recomputation identity is asserted.

For U=W.T@Qnew, the full1024-row old bank is reconstructed as

    concat((U L) F0, U Fj) = U Fcan.

The actual inner repeated-index read is

    [X0^2 (F0 Lambda0).T] L.T + Xj^2 (Fj Lambdaj).T
      = Xold^2 (Fcan Lambda).T.

The original ONE trailing multiplication by U_live.T remains outside this sum.
Both positive and negative lambda values and their original column order remain.
Original all-row diagonal cubes, omitted-radial terms, young/newborn paths,
star/trace/BD/NODE/P heads and all other consumers see the full reconstructed pp.

At the next joining layer with overlap N, materialize

    concat((N L) F0, N Fj, Fnew).

Thus a lazy chain never exceeds one retained map; at most two factor groups are
stored between joins. L14's map is consumed at14 and15 and dies with the model.
All original join-count/intact-cohort guards remain. The old single-factor shape
guard becomes an exact two-group ordered-column/dtype/history check when needed.

The actual representation is fully finite-checked at every join: BOTH F0 and Fj
are scanned, even though F0 was already checked; L and every newly composed map
are scanned too. The original Q/core and all-row old-bank finite checks remain.
Extra mapped-U and mapped-inner scans are added. There is no silent nonfinite
acceptance, clamp, pseudoinverse, or relaxed old output/domain/resource gate.

The virtual L@F0 is not materialized, so a prospective PAID overflow envelope
also guards the omitted intermediate. Compute maxabs(L) and maxabs(F0) with
keepdims, cast each to an FP64 one-element array, and use typed FP64 one-element
width/limit arrays. Require BOTH maxabs(L)<=2^80 and maxabs(F0)<=2^80, AND
384*maxabs(L)*maxabs(F0)<=2^80. Individual bounds are necessary: a huge finite
map and a tiny prefix can meet the product bound while an input transform
overflows. Each depth-d input
transform grows maxabs by at most4^d, each output recombination by at most4^d,
and a leaf has384/2^d summands. The conservative whole-tree factor is2^(5d);
d<=6 keeps input transforms<=2^92 and products/recombination<=2^110, with an
18-bit exponent gap to FP32 overflow.
Ordinary rounding cannot consume that margin. This is a stricter safety envelope,
not a clamp or a claim that the never-formed rounded array was scanned. No weak
scalar RPC arithmetic is used in its comparison. An envelope failure is fatal.

FP32 multiplication/addition association changes. The equations above prove
real-arithmetic equality, NOT output-byte, selector, complete-bill, raw or runtime
qualification. Any changed predictions require their own evidence.

## Code-derived complete tariff

`price.py` reuses only the pinned original `dense`, `packed`, `matrix_price`
definitions from SHARED/price_metadata.py and the last actual BOX square-depth
tables. Row/output/contracted tiling, all original D4/D6 scalar arithmetic,
concats and partial-sum additions are included. Only current native40_v2's
first40 scalar geometries are read, each diagnostic SHA matched to the actual
positive result; no old APUSH selectors or selected numerical subset substitutes.

Write M(m,k,n) for that tariff, n=1024,r=384,Nc=R+j, and h(x)=2x-1 for the
FP32 finite predicate plus Boolean all reduction. Relative to the parent:

At a lazy join `_slr_join_factors` removes M(r,r,R). `_slr_finish_factors`
changes scalar work by

    -r*Nc - h(r*Nc) + h(r*R) + h(r*j) + h(r*r)
    + h(r*R) + h(r*r) + 20.

This includes removing the original factor concat and replacing the original
one full-factor scan with scans of BOTH blocks plus the map. No prefix finite
scan is declared free. At consolidation, added matrix work is

    M(r,r,r)+M(r,r,R)+M(r,r,j)-M(r,r,Nc),

plus h(r*r). The new three-part factor concat and full scan have exactly the
same element tariff as the original two-part concat/full scan.
The second h(r*R)+h(r*r) pays both envelope abs/max scans. The20F pays two
FP64 casts, two typed FP64 full(1,1) constants, two multiplies, three comparisons
and two Boolean conjunctions. Boolean all over its one-element result costs0F.
This guard adds mean36,427,848F, including6F per join for both individual maxima.

At EACH lazy consumer, including15, added matrix work is

    M(n,r,r)+M(n,r,R)+M(n,r,j)-M(n,r,Nc)
    +M(s,R,r)+M(s,j,r)+M(s,r,r)-M(s,Nc,r),

plus n*Nc+s*r+h(n*r)+h(s*r). This pays the extra reconstructed-old concat,
inner sum, mapped-U scan and mapped-inner scan. The two weight/square operations
cover exactly the original total element counts. All-row old finite scanning,
the outer old+young concat and trailing old-plane read remain in the baseline.
Every remaining full-model operation stays in the measured complete baseline.

Source-only scalar reconciliation of these explicit terms gives current40 mean
delta -6,615,094,649.75F. The conditional BOX mean baseline is402,343,068,582.4F
(actual c2 geometry/bills plus BOX's retained first-call bindings); resulting
conditional mean is395,727,973,932.65F. Rank37 cold conditional price is
401,096,612,878F; same-geometry warm is397,809,624,308F. These are NOT actual
candidate bills. The new `price.py` itself remains unexecuted pending root review;
these figures were independently evaluated with source/JSON-scalar arithmetic.

No new persistent arena, reshape binding, getter or literal array is added.
Retain every original BOX binding and initialization3,286,988,570F; do not refund
the BOX1,342,018,272F addition or c2's nested startup. Basic slices/newaxis/T
are0F, not zero wall/handles. Any frame-cache miss/eviction still consumes real
runtime and must appear in actual execution. A changed trajectory can change
shapes/selectors and bills: the above prices are conditional, not unconditional
future-cost bounds or a banked adjusted-score benefit.

## Liveness and memory

Maps, factors and reconstructed products are independent outputs of the parent's
matrix helpers, never views into their aliased scratch arena. Every product is
serial. Only the original bounded caches retain original-arena shape views;
the new code stores no globals containing participant arrays. `state.lazy` is
strictly private to the existing per-model shared state and is cleared after
consolidation. The final map dies on prediction retirement. There is no context
or model-spanning numerical cache.

At a lazy reconstruction, prefix and recent outputs coexist with their concat;
that extra n*Nc copy is NOT omitted. The mapped basis and small map also remain.
Read temporaries are explicitly released. For the actual40 maximum Nc18176,
the conservative extra transient allowance4[n*Nc+2*n*r+r*r] is78,184,448 bytes.
This bounds the larger reconstruction phase as well as the smaller split-read
phase; the original participant input/output and original arena stay charged.
The envelope's one-at-a-time abs temporary is smaller than this same allowance;
it is not excluded from the liveness scope.
It is not an observed RSS result. If actual shapes exceed this geometry, report
the actual bound and enforce the original independent8GB limit, not this estimate.

## Prospective tiny supervised algebra/billing test (source draft, not run)

Reuse the existing admitted FNP primitive harness/real local-RPC transport and
owned-tree monitor. Do not call predict or materialize model/head parameters.
Extract only the byte-pinned original `_mm`, shared-matrix and finite helpers
plus this appendix. The private tiny namespace selects the original public dense
branch with `_WINOGRAD=False`; no arena is needed or billed. This is an
algebra/guard/dense-tariff unit check, NOT deployed packed-kernel, binding, card
or runtime qualification. The composed candidate keeps its original flag,
kernels and all bindings. All fixture/reference work is paid in one context.
Root must independently verify source/API admission and the exact dense prices.

Use ONE fresh real-RPC paid context containing two fixed sequences: L8 lazy
followed by L9 consolidation, and L14 lazy followed by a second L15 consumer.
There is no second local-mode context or full-network call. Use actual r384,
full n1024 reconstruction, s32 response rows, v64 live read rows, R64 prefix
and j32 join (including the next join's32 columns). Use bounded binary-
rational FP32 fixtures, signed opposite coefficient halves, nonzero factors and
noncommuting maps L=I/2+cyclic_permutation/8 and N=3I/8+adjacent_swap/16.
Their operator norms are at most5/8 and7/16, so no growing/ill-conditioned solve
is involved.
Each positive half-factor is a strictly positive binary-rational row pattern
plus a monotone column offset; its negative-coefficient partner is exactly half
that column. U/V entries are strictly positive, and L/N have nonnegative entries
and positive row sums. Every pair contributes(1-1/8)=7/8 of a strictly positive
cubic plane term. Nonzero reconstruction/plane norms follow before any run,
without a period-dividing-half cancellation or result-selected fixture rescue.
The fixture construction, copies/casts, scans, direct reference and comparisons
are ALL paid. No random seed search or result-dependent fixture/tolerance change.

Independent reference directly materializes Fcan in FP64 from the same stored
FP32 fixture values and uses ordinary public dense matmul, not these grouping
helpers, to form full X and the signed repeated-plane read. Check the transposed
map/column ordering and the second-step `(N L)` order. Separately reconcile the
component's exact FP32 bill to the same term list with the exact public dense
M(m,k,n)=m*n*(2*k-1), as implemented in `tiny_worker.py`;
charge reference/qualification F separately in the complete context, never hide it.

Proposed fixed algebra tolerance is maxabs<=2e-5+5e-5*maxabs(reference) and
relative Frobenius<=2e-4 with denominator floor2^-20; require a nonzero finite
reference norm>=2^-12. This allows ordinary 384-term FP32 dot/reassociation
rounding in the bounded fixture, not a claim of FP32 operation-order identity.
Reject invalid schedule, wrong group column roster/dtype, stale canonical F,
nonfinite factor/map and a missed next-join consolidation before further math.

Prospective primitive envelope: aggregate<=10BF INCLUDING setup/reference,
whole120s, combined8GB, zero added swap, completed observations<=1s, all groups
and numerical owners retired. Root must freeze the actual independently priced
fixture and resource wrapper before any test; this document does not authorize
its execution. No full-card test follows automatically. A later fixed37 own-bill
and output/domain/RPC test would still retain original gates, including the45s
component objective if root selects it. No QR runtime improvement is claimed:
actual PROFILE's ten QR calls totaled only0.177730168 seconds.
