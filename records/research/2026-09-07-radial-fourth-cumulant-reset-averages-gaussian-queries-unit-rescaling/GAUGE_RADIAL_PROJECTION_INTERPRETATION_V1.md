# Radial projection, random queries, and positive-unit gauges

September 6, 2026 PDT. Phase 2 source-only mathematical audit. No numerical
test, estimator import/run, target-array decoding, gauge optimization,
fitting, submission, or modification of a frozen source/output.

## Finding

The current radial reset is a **ratio of Gaussian-query ensemble
expectations**, in the supplied hidden coordinates. Its failure to remain
unchanged under arbitrary unequal positive unit rescaling is not, by
itself, an implementation bug. Rescaling those units also changes the
covariance of the next Gaussian weight query. Keeping an isotropic query
metric after that rescaling changes the projection objective.

This interpretation passes an independent symbolic Wick-contraction
check. No numerical certificate or accuracy claim is made.

## Exact interpretation of the nominal moment projection

Condition on a hidden-state law with symmetric covariance C and symmetric
fourth cumulant tensor Q. The independent query is w~N(0,sigma^2 I). Write

    q(w) = kappa4(w.T h) = Q[w,w,w,w],
    v(w) = w.T C w.

Gaussian fourth moments give

    E_w q(w) = 3 sigma^4 sum_ij Q_iijj,
    E_w v(w)^2 = sigma^4 [(tr C)^2 + 2 ||C||_F^2].

The radial ansatz is

    Q_rad,ijkl = (r/3)(Cij Ckl + Cik Cjl + Cil Cjk),
    Q_rad[w,w,w,w] = r v(w)^2.

Matching the two ensemble averages therefore gives, for positive
denominator,

    r = E_w q(w) / E_w v(w)^2
      = 3 sum_ij Q_iijj / [(tr C)^2 + 2 ||C||_F^2].

This is NOT `E_w[q(w)/v(w)^2]`. Nor is it the scalar least-squares fit to
individual query errors: minimizing `E_w[(q(w)-r v(w)^2)^2]` would give
`E_w[q(w) v(w)^2] / E_w[v(w)^4]`, where defined. None of these criteria is
the final-layer mean MSE or a guarantee of an adjusted-score improvement.

## Diagonal and general query metrics

For w~N(0,A), A=diag(a_i) with a_i>=0, the same contraction yields

    r_A = 3 sum_ij a_i a_j Q_iijj
          / [(sum_i a_i Cii)^2 + 2 sum_ij a_i a_j Cij^2].

Under h'=D h with D=diag(s_i), s_i>0, the same physical scalar query has
`w'=D^-1 w`. Thus `A'=D^-1 A D^-1`, or `a_i'=a_i/s_i^2`.
The transformed covariance is `C'=D C D` and the transformed plane is
`Q'_iijj=s_i^2 s_j^2 Q_iijj`. Substitution cancels every s_i: `r_A'=r_A`.
Keeping A=I instead generally produces a different scalar.

For a general symmetric positive-semidefinite A, the formulas become

    E_w q(w) = 3 sum_ijkl Q_ijkl Aij Akl,
    E_w v(w)^2 = (tr(AC))^2 + 2 tr(ACAC).

The denominator is available from C and A; the numerator generally needs
off-plane fourth information, not just Q_iijj. A proposed empirical or
downstream query metric must own that contraction and its acquisition
cost. Diagonal weighting is cheap once Q22 exists, but no particular
weighting, estimator variant, or run is authorized by this identity.

## Exact current-source ownership and qualifications

Source: `candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py`, SHA256
`3b7f55f9a814789743467bb4bcda0837c3c1aad15d120e4cdab632594bf54757`.

- Actual resets are lines697 and1164, after the scalar variance and fourth
  cumulant replace the diagonals. They use `post22` and `cov` of that
  nonlinear map. These are approximate moments, not supplied true moments.
- The mean head subsequently changes carried C at lines1466-1469. The
  radial reset is NOT recomputed there. The exact ensemble interpretation
  above applies to the map's own pre-head C/Q pair, not automatically to
  a new literal fourth tensor accompanying the corrected carried C.
  This is ownership disclosure, not authority to change the co-adapted head.
- The Wick/Frobenius equality assumes symmetric C. In finite precision,
  the code does not explicitly symmetrize every covariance push. If
  C=S+K with S symmetric and K skew, a real quadratic query sees S; the
  code's denominator exceeds `(tr S)^2+2||S||_F^2` by `2||K||_F^2`.
  This is a quadratic-in-antisymmetry numerical qualification; no size or
  raw-score consequence was measured here. No PSD qualification is inferred.

## Prior unit-audit coverage and remaining physical algebra

No whole-current-bank positive-diagonal gauge execution was located in the
bounded current-package and tracker source search. The saved
`certify_two_matrix_star_transport_v1.py` checks one standardized V/unit
identity and dense tensor action; the learned-memory synthetic certificate
checks a congruence unit conversion. Neither certifies the nonlinear
CP/radial trajectory or learned heads under a network-wide rescaling.

With invariant radial scalar, inactive absolute floors, and real arithmetic,
the current physical formulas have the expected unit laws: m_i scales s_i;
Cij scales s_i s_j; K21ij scales s_i^2 s_j; trace b_i scales s_i; stored
star Aij scales s_i s_j^2. A CP column born at unit j carries an additional
birth-unit factor: p_ic scales s_i s_birth,j and lambda_c scales
s_birth,j^-3. Those factors cancel in shared newborn transport and every
cubic read. Birth beta/birth makes u,t,s,delta scale as squared birth units;
the bounds involving v and v^2 are homogeneous. No further dimensional
mistake was established in covariance/cold diagonals, star transport,
signed births/repair, scalar/Mehler maps, or moment recentering.

Selection uses g and min(g,1-g), hence is dimensionless absent floor and
rounding changes. Absolute variance/birth floors and FP32 arithmetic add
numerical non-invariance. The learned heads are not constructed to be
gauge covariant: they use relative log-sigma, raw W-power features, and
message states without a matching per-unit transformation law.

Conclusion: standardized-radial replacement is NOT justified as an
automatic bug fix. The random-query interpretation explains the existing
natural-coordinate projection. Other task-aware fourth projections remain
separate research questions with their own information, cost, and execution
gates; this note grants no such run and records no raw improvement.
