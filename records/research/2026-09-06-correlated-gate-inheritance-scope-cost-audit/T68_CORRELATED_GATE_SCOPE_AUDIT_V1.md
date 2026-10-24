# T-68 correlated-gate inheritance: scope and cost audit

Phase 2 only. Source/receipt audit, September 6, 2026. No new numerical
experiment, action/truth array, fit, feature, or candidate was evaluated.
Frozen sources and closeouts are unchanged.

## Decision

**Do not reopen this as an actionable accuracy lane.** T-68 is not an
empirical or mathematical closure of arbitrary accumulated correlated-gate
inheritance. Its broad language should not be repeated as one. However,
the later exact all-distinct one-edge source test and failed fixed q4 edge
approximation are directly relevant negative evidence; no complete affordable
representation of the unmeasured accumulated intervention was identified.
An unmeasured multi-layer effect, by itself, is not a new executable lead.

This is a research-budget decision, not an impossibility theorem for every
possible future tensor compression algorithm.

## What T-68 actually measured

`public_g29_bond_kernel_v1/bond_kernel_oracle.py` independently reloads each
old bank layer and computes its next-layer **scalar diagonal** third read.
It evaluates the old CP part only, T=sum lambda*Sym(u,u,w), with the card's
same-layer covariance/gates. D1/D2 are not injected into any later state.
There is no corrected covariance, following-mean consumer, closure refit,
accumulated bank trajectory, or final-mean score in that oracle.

The comparisons are against the accumulated old card's scalar k3 residual,
using archived 32M moments. This rejects the proposed one-step scalar
explanation. For example rank100 at read15 has D1 RMS3.0948e-5 versus
residual RMS9.6061e-4, correlation .057, and natural-strength explained
energy .0026. These are not current-bank final raw measurements.

The oracle also does not project its correction onto all-distinct tensor
entries. Setting rho_ii=0 does NOT remove all repeated triples: the bond
correction at (i,i,j) still contains

    2*g_i*phi_i*phi_j*rho_ij*T_CP[i,i,j].

Those repeated post moments have separate ownership in the card's pair map
and star restoration. Therefore the T-68 vector is not automatically the
correct intervention that preserves existing repeated moments while fixing
only omitted distinct inheritance. Its standalone dense certificate verifies
the declared full CP contraction, not that ownership distinction.

The phrase "an accumulated version cannot exceed ~10x" is not supported by
an operator-norm bound or an accumulated experiment in this source. A scalar
one-step norm/correlation cannot bound later covariance/mean consumers in
general. No contrary positive value follows from pointing this out.

## The engine is not an independent closure of that gap

The CURRENT `public_g19_attribution_v1/run_tensor_recursion_v4.py` constructs
`terms` with di,dj,dk all positive, then uses zero-bond plus those terms in
F3. Thus it excludes single-bond patterns. The T-68 standalone oracle and
certificate enumerate those patterns without this filter. Their statement
that they reproduce the engine's full bond0/1/2 loop is not source parity
with the current file.

File modification times, rather than tracker prose dates: oracle September2
04:10:24 PDT; its certificate04:08:32; current engine06:30:03. The exact engine
contents at the earlier oracle time were not established by this audit.
Accordingly T-73's engine/card agreement with supplied fourth planes is not
a controlled accumulated bond1 ablation. It changes other state supplies,
and the currently available engine also omits the leading single-bond class.

T-69/T-70 concern a DIFFERENT source: a repeated pre-plane feeding distinct
triples. Their accumulated gate-chain overshoot, engine NOFEED ablation and
in-card mean results cannot be transferred to correlated inheritance. They
do, appropriately, warn that an isolated read and a coherent bank intervention
are different experiments. T-73's supplied-fourth-plane/operator redundancy
is also not a theorem about the missing bond1 source.

## The later experiment that matters

`THIRD_COVARIANCE_EDGE_DESIGN_V1.md`, `third_covariance_edge_v1.py`, and
`check_third_covariance_edge_wide_v1.py` explicitly implement

    DeltaT_ijk = T_distinct_ijk *
       (C_ij*d_i*d_j*g_k + C_ik*d_i*d_k*g_j + C_jk*d_j*d_k*g_i),
    d_i = phi_i/sigma_i.

Since C_ij*d_i*d_j=rho_ij*phi_i*phi_j, this IS the leading correlated-gate
inheritance class. Unlike T-68, it includes CP, pushed coordinate-star and
covariance-trace contributions and removes every repeated triple. The
`ThirdCore.matrix/quadratic` contractions implement that removal explicitly.

The September5 fixed test was rank35 post12 -> pre13, 32 predetermined next
repeated indices with all1024 partners, using the current carrier family but
oracle previous means/covariances. It was NOT an unchanged deployed-bank
closed loop. Natural coefficient1 gave:

- Full-source unweighted plane-error MSE ratio1.00254042.
- Leading-consumer-weighted ratio1.01336446, cosine -.04847, source norm
  7.69% of the corresponding weighted baseline error.
- Fixed top4 covariance-edge approximation: weighted source relative error
  84.76%, unweighted83.30%, versus the prescribed15% limits.

The weighting is the directed r0 coefficient .5*d_next_i*g_next_j. It is
not the full symmetrized/r0..2 bank covariance action, next W covariance
push, or final-mean response. This result therefore closes its declared
source/representation test, not all downstream accumulated effects.
The closeout correctly stopped before rank100, damping, rank/basis variants,
fitting or rollout. None is justified by this audit.

Other later receipts are adjacent, not substitutes: `INHERITED_DISTINCT_GAIN_
V2_CLOSED.md` tested a genuinely accumulated per-layer scalar gain on inherited
distinct CP content and obtained pooled final raw ratio .99544356, failing
.90. It did not add correlation bonds. `PROJECTED_PLANE_FEED_ROUND_V1_CLOSED.md`
tested the different repeated-plane source projected into births; it failed
transfer across35/100. Neither should be mislabeled a full bond1 ablation.

## Complete representation price, not just a cheap local coefficient

For a current symmetric CP atom lambda*p^3, the full single-bond source has
the exact form lambda*Sym(h,A), where h=g*p and

    A = diag(phi*p) rho diag(phi*p).

After an arbitrary next W, each A needs its own congruence. The shared-rho
Hadamard form is not preserved. Storing full per-atom matrices at the recorded
R=20,352 and n=1024 requires R*n^2 float32 entries:79.5GiB, before anything
else. Factoring rho instead expands each atom across its covariance modes;
full-rank factors do not avoid that growth after transport. The tested q4
truncation is neither accurate nor free, and was explicitly closed.

The existing exact query code uses two dominant n-by-n by n-by-R products
per queried output row, about4*n^2*R ordinary FLOPs. Reading all1024 rows
at R=20,352 is approximately4*n^3*R, or39.75 budgets of2^41, for ONE source
cell, before remaining terms. The 32-row oracle is consequently an offline
information diagnostic, not a near-deployable plane generator.

A dense correction tensor is a different complete form:4GiB float32 state,
with three classical mode pushes costing6*n^4 =3 full budgets per transition,
before nonlinear source construction and the original card. These are costs
of the specified ordinary complete implementations, not lower bounds against
all fast algebra or hypothetical compression. Neither is a viable additive
carrier for an adjusted-score target near1e-9.

A cheaper model that drops the new distinct state after one read, projects it
into an old plane/trace span, or retains only a few covariance modes is a new
approximation, not accumulated exact inheritance. This audit supplies no
new capacity certificate for one. It therefore proposes **no new value
experiment**. A genuinely new, independently specified source compression
with complete cost and fidelity would be required before this gap becomes
an actionable mechanism; simply increasing q, changing a layer, or appealing
to possible accumulation does not meet that bar.
