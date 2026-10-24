# CP suffix/range products: exact identity, no whole-card saving established

Source/algebra audit only, Phase2 width1024/depth16. No estimator execution,
new targets, fitting, numerical experiment, or modification of frozen files.

## Verdict

A supplied fixed trajectory permits an exact O(depth) matrix-product schedule
for ONE terminal CP query instead of replaying every birth through every
intermediate state. Consequently the old T-97b wording that any factor/probe
view necessarily costs O(depth squared) is too broad.

This is not an exact cheaper implementation of the current whole card. It
still needs each intermediate third plane to generate its next covariance,
means, gates, masks, radial state, star residual, and birth factors. Accounting
for those reads and each cohort's distinct covariance base restores the same
triangular dense matrix work. The strongest newborn sharing exception is
already implemented in the official bank. No new accuracy/cost experiment
follows from this audit.

## Exact current-bank identity

Use the last `_predict_binary` definition and `_push_bank_shared_newborn` in
`candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py`, not an earlier
overridden implementation. For a birth at post-layer b, let E_b select its
birth coordinates, Q_b be its stored `newborn` covariance-base matrix, and
t_b/s_b its stored diagonal additions. Thus

    P_b+ = Q_b + E_b diag(t_b),
    P_b- = Q_b + E_b diag(s_b).

Q_b is the selected, alive-masked columns of `g[:,None]*Cpre` with the
diagonal removed. It is NOT shared across birth layers. The two signs have
weights +a_b/-a_b. Let D_j=diag(alive_j * gate_j). In exact real arithmetic,
at a later PRE layer l,

    M_b,l = W_l.T D_(l-1) W_(l-1).T ... D_(b+1) W_(b+1).T,
    P_b,l+/- = M_b,l Q_b + (M_b,l E_b) diag(t_b or s_b).

The empty-interval convention is M_(l-1),l = W_l.T. With a supplied
trajectory, the suffix recursion is

    M_b,l = M_(b+1),l D_(b+1) W_(b+1).T.

For ONE fixed l, build those maps backward once; each covariance-base
application then costs one rectangular multiplication, and M_b,l E_b is
only a column selection. There is genuine terminal-only sharing.

The current bank already exploits this at age one: it pushes Q_(l-1) once,
uses bare selected W_l columns for both diagonal additions, and pushes the
older signed factors together. There is no unclaimed factor-of-two newborn
optimization here.

## Complete dense scheduling census

Let one full-width GEMM equivalent mean 2*n^3 classical FLOPs; this is NOT
the actual Winograd/live-card meter. In the full-width/full-birth limit,
at pre-layer l=1..15:

- current shared-newborn push: 2*(l-1) older factor matrices plus one common
  newborn matrix = 2*l-1 GEMM equivalents;
- suffix reread: l-1 range-product multiplications plus l cohort-specific
  Q_b applications = the same 2*l-1 equivalents.

Both schedules therefore use 225 equivalents for all factor pushes/reads.
The active CP-plane contraction is still required at pre1..14: in this
full-width limit it costs another sum(2*l)=210 equivalents. The final
layer already uses only its scalar diagonal, not a full plane. Covariance,
coordinate-star, covariance-trace, radial/Mehler maps, learned heads, and
their state updates all remain additional and unchanged. Thus the suffix
identity offers no whole-card arithmetic reduction even before its storage,
dispatch, copies, and altered floating-point association are charged.

With fixed width n and b_b<=n selected births, suffix minus online classical
push cost at pre l is

    2*n * sum_(b=0..l-2) n*(n-b_b) >= 0.

For the actual compact card, denote post active count by a_b. Its exact
classical push subtotal, excluding elementwise reconstruction, is

    2*n * sum_(l=1..15) a_(l-1)
          * [2*sum_(b=0..l-2) b_b + b_(l-1)].

A backward schedule retaining each restricted range M_b,l requires

    2*n * sum_(l=1..15)
          [sum_(b=0..l-2) a_(b+1)*a_b
           + sum_(b=0..l-1) a_b*b_b],

before the same downstream CP-plane and scalar reads. These formulas are
complete for the specified factor transport operations, not a universal
lower bound on all possible tensor algorithms. No advantageous variable-
shape, Winograd-metered schedule was established. In particular one cannot
price Q_b applications as free merely because M_b,l was constructed.

The already archived exact-bank census (`bank_matmul_census_v1.json`, source
SHA 3b7f55f9a814789743467bb4bcda0837c3c1aad15d120e4cdab632594bf54757)
has full-card FLOPs490,466,018,197 on rank35 and489,374,517,341 on rank100,
with byte-identical bank predictions. These are existing measurements, not
new suffix bills; the classical equivalents above must not be substituted
for their actual compact/Winograd implementation costs.

## Why known W does not supply the missing trajectory

At layer l, K21 changes the nonlinear mean, covariance, post repeated-third
target, radial update, and learned correction. These in turn determine the
next pre m/C, hence its gates and active/birth masks. The current birth
diagonal parameters also depend on the current third residual. Computing a
bank teacher first already pays for this state acquisition. Reusing its
gates/covariances for a terminal suffix calculation does not remove that
first pass. A cheaper forecast is a different approximation, not this exact
identity or an authorized new arm.

A common-prefix inverse also supplies no demonstrated escape. Masks can
make the prefix singular; even when an inverse/right inverse is available,
rewriting births in a common basis leaves the large basis-times-all-factors
product at every plane read. Avoiding that read requires a different tensor
summary with its own complete contraction price. Grouping births before the
cubic read introduces cross terms: a sum of cubes is not the cube of a sum.

Reassociation is real-arithmetic exact, not floating-point byte parity.
The bank stores FP32 factors, performs intermediate gate multiplications,
and uses cancellation-sensitive signed pairs. A changed association would
require a new fidelity certificate before any claimed deployable equality.

## Archive scope

- T-95..T-97b in `public_g19_math_acceleration_v1/LANE_g3_tracker.md` already
  model separate transported covariance and coordinate factors. The billed
  low-rank reconstruction/truncation experiment invalidated its initially
  optimistic cost; those rank/cap arms remain closed. Its blanket asymptotic
  phrasing should not erase the fixed-terminal exception above.
- `radial_lookahead.py` actually builds backward gated products on a
  Gaussian forecast, then fits compressed CP weights to current/final cubic
  queries. Existing rank35 result:1.0432655976944163e-7 raw without query fit
  versus1.3874975689503222e-7 with it. This is an approximate compression
  negative, not a proof against algebraic scheduling of the exact bank.
- `LANE_direct_diagram_adjoint.md` checkpoint80 explicitly distinguishes a
  single fixed seam from multiple transported source metrics; checkpoint90
  explains why the covariance reset itself is not one separable suffix map.
- The generic backward-state CV and physical cubic-action transpose pilot
  were actually executed and failed their frozen value gates. Neither is an
  exact online-CP reassociation benchmark; neither is reopened here.
- No separately identifiable Phase2 receipt literally named block-retro was
  located in this bounded filename/text search. This is not an assertion
  that every historical archive spelling has been exhausted.

Conclusion: a legitimate terminal-query identity, but no complete exact
inference saving or new raw-information mechanism for the current bank.
