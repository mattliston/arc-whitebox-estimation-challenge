# Score-aware future geometry for CP replacement: source-only audit

September 6, 2026 PDT. Phase 2 only, width1024/depth16. This bounded audit
read source and existing small JSON/Markdown receipts only. It ran no
estimator, derivative, numerical spectrum, training, or scoring job and
decoded no numerical array or target. Existing sources and closeouts remain
unchanged. Paths below are relative to the Phase2 experiment directory;
`NEW/` means `astra_phase2_reexamination_v1/`.

## Verdict

Approximate compression in a target-free, full-state endpoint-risk metric is
not experimentally closed by the identified receipts. It is also not an
established cheap new arm. The exact observability/Hankel question already
appears in the mathematical lanes; the executed lookahead and backward-M
experiments did not implement that metric. A new execution recommendation
would require both a fixed affordable metric acquisition and a complete
representation/consumer bill. Neither was established in this audit.

## Critical C1 ownership correction

`public_g19_attribution_v1/run_c1_adjoint_rank.py` explicitly loads
`mps_truth_rank0035_P8M.npz` and defines

    resid = truth_post15 - deployed15

before transporting this vector backward through the Phi/W consumption
chain. Its pair costate is assembled recursively as

    Mcore_l = A_l @ (Mcore_(l+1) + diag(Sv_(l+1) * g_(l+1))) @ A_l.T
    M_l = diag(a2_l) @ Mcore_l @ diag(e1_l).

Here `g` is descended from that target-dependent final residual. The
reported `r95_value` also uses the actual supplied tensor-plane difference
`dK`, through signed terms `S[r] * (U[:,r].T @ dK @ Vt[r])`.

Consequently the small value ranks cited in tracker T-21 are properties of
one target-conditioned signed scalar pairing. They are NOT weights-only
query directions, not an MSE-risk metric, and not a uniform projection
certificate. This corrects the tracker wording that those particular read
directions were already card-computable. The existing
`C1_ADJOINT_RANK_rank0035.json` reports ordinary costate Frobenius-energy
95% ranks105--197. Value-r95 is8 atL14 and11 atL13, but atL9 it was not
reached among the256 terms examined. Neither number supplies a deployable
target-free rank guarantee.

The C1 source mtime is August31,2026,16:53:50PDT. The H3 and H4 mathematical
lane mtimes checked in this audit were August30,21:07:24PDT and
August31,14:22:49PDT, respectively; their later section numbers do not make
them new September6 experiments.

## Theory versus executed experiments

Selected relevant sections were read, not the entire very long lanes:

- `public_g19_math_acceleration_v1/LANE_h3_reverse_adjoint.md` section21.5:
  the complete-depth block Gram includes cross-layer reinforcement and
  cancellation. Per-layer Frobenius tails do not certify endpoint loss.
- `public_g19_math_acceleration_v1/LANE_h4_reverse_adjoint.md` sections24,
  57 and58: the future-visible quotient and its weighted singular tail are
  the right theoretical objects. One Gaussian ReLU covariance reverse
  update can turn a rank-one costate into a full-rank matrix through the
  Hadamard multiplier `1/4 + arcsin(rho)/(2*pi)`. Exact full rank alone is
  not a lower bound on the approximate energy tail. The tight-frame tail
  bounds require their stated hypotheses; no such current-net premise was
  numerically established by this audit.

The following actually executed narrower experiments were checked against
their source and/or saved small result records:

1. `NEW/radial_lookahead.py`, `screen_radial_lookahead.py`, and
   `radial_lookahead_r35_v1.json`: at layers3/7/11, keep2048 energy-selected
   CP columns and optionally ridge-reweight them against256 next-layer plus
   256 terminal scalar-cubic rays. The suffix directions come from a
   separate Gaussian forecast. This is not the full covariance-response
   Jacobian. Fixed rank35 raw MSE was1.0432655976944163e-7 without query
   fitting and1.3874975689503222e-7 with fitting, despite in-sample query
   residuals around0.00065--0.00076. These are old unlearned/binary-frame
   results, not current-bank score measurements.
2. `public_full688_backward_adjoint_state_cv_v1/run_experiment.py`,
   `PREREGISTRATION.json`, and `CV_RESULTS.json`: backward M carries four
   learned vector sources with exact `W@(Phi*b)` transport and learned
   gain/mixing, then conditions a forward learned state. The688-net OOF
   endpoint MSE was3.233766009119748e-8 versus J's
   3.2334958229470984e-8. This is not an endpoint-risk Gram or physical
   full-state covariance/cumulant adjoint. Its reported small backward
   vector-transport cost cannot price a full-matrix reverse pass. The
   native transfer was not established by this failed capacity arm.
3. `NEW/GAUSSIAN_NEXT_MEAN_RISK_METRIC_V1_VERDICT.md` and
   `THIRD_ENDPOINT_SCORE_RISK_V1_CLOSED.md`: a distinct analytic
   next-mean risk metric was qualified on one fixed cell, but refitting
   the same plane dictionary gave residual risk0.772535, with a same-ridge
   optimum lower bound0.77252535 versus the required0.50. This closes that
   dictionary/objective arm, not task-weighted CP replacement.

The physical cubic-transpose feature pilot also executed and failed its
source-on versus source-zero controls; H4 section100 records its terminal
result. It must not be called an unexecuted successor merely because
`LANE_direct_diagram_adjoint.md` section121 contains an earlier design.

## The lawful metric and what it does not claim

Let `x_l` include every affected state: means, covariance, CP factors and
weights, coordinate star, trace vector/metric, radial state and learned
recurrent state. For one fixed membership cell, let `J_l` be the derivative
of the complete remaining bank's1024-vector final mean with respect to a
typed replacement error `e_l`. A target-free output-drift risk is

    R_l(e_l) = ||J_l e_l||^2 / 1024.

For multiple replacements the appropriate linearized quantity is

    ||sum_l J_l e_l||^2 / 1024,

not a sum of separate local losses. This retains the covariance and mean
feedback pathways omitted by scalar gated suffix rays. Exact current-plane
star restoration must be included in the replacement operator; it does
not make the future all-distinct error disappear.

Output drift is not the change in true MSE. The latter additionally has
the unknown signed cross term with the incumbent's true error. Also a
fixed-cell Jacobian does not certify a finite compression step that changes
membership, ordering, nonlinear state or subsequent births.

## Complete acquisition and representation obligations

**Trajectory acquisition.** Computing the exact current-bank trajectory
first already pays approximately490B metered FLOPs on the archived rank35/
100 census. It cannot be excluded as an offline preprocessing step at
deployment. A cheaper Gaussian forecast avoids that payment only by using
an approximate trajectory and metric; its error must be tested separately.

**Backward geometry.** A generic covariance congruence reverse
`Lambda <- W @ Lambda @ W.T` uses two dense matrix products per endpoint
probe per crossed transition, before the nonlinear pair maps and all CP,
star, trace, radial, head and mean-feedback derivatives. Four probes across
15 full-width congruences amount to120 conventional FP32 GEMM equivalents,
about257.7B arithmetic FLOPs. This is an illustrative full-width census,
NOT an actual D4 meter or universal lower bound: the terminal diagonal
response can save a product per probe, active rectangles and Winograd
change prices, and exact common structure could permit sharing. Conversely,
the omitted state derivatives and metric factorization are not free.
Few random endpoint probes are not automatically a sufficient approximation
to the1024-output quadratic risk.

**Projection is not a cheaper read by itself.** Writing an old factor bank
as `P=U Z`, with U n-by-r and Z r-by-R, can reduce transport dimensions.
But reconstructing `Q=V Z` and then evaluating
`(Q*Q) @ (Q*lambda).T` still pays the dominant conventional `2*n*n*R`
plane product, as well as reconstruction. One cannot credit a transport
saving while omitting that consumer.

**A tensor core requires its own bill.** A full symmetric r-by-r-by-r core
can instead form a plane using an n-by-r-squared quadratic lift, a
contraction costing approximately `2*n*r^3`, and a final product costing
`2*n*n*r`. Three-leg core transport, changing bases, forming/projecting
new births, scalar reads, star restoration, covariance, nonlinear maps,
heads, factorization, copies and scratch must all be included. Merely
quoting the `r^3` storage or `n^2*r` basis push understates the method.
Existing unweighted Tucker/plane-projection failures remain relevant but
do not prove failure in a new fully specified score-weighted metric.

## Decision

Preserve the target-conditioned C1 receipt as an oracle diagnostic. Do not
reuse its selected low ranks as deployment evidence, and do not relabel
scalar lookahead, learned backward M, or local covariance eigenvectors as
the requested full score-aware geometry. The approximate-Hankel question
remains mathematically meaningful, but this audit establishes no cheap
new execution arm and authorizes no spectrum run, fit, rerun or candidate.
