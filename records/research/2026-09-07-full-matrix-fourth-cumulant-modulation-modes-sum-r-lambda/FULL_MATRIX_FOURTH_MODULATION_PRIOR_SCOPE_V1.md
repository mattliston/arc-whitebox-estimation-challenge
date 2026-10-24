# Full-matrix fourth modulation: exact prior-work scope

Phase2 only, September6,2026 PDT. Source/algebra audit, no numerical
execution, network/target arrays, sampling, training, candidate or submission.
Frozen predecessors are unchanged and their specified arms remain closed.

## Conclusion

The proposed tensor form is **not absent from the archive**. Radial,
SSC, transported Sym(C,D), free-semicircular C,Q, and full-trace work
contain exact subclasses of it. However, the inspected Astra/G22/MATH
sources do **not** show a numerically tested carrier of arbitrary,
independently born full symmetric modulation matrices V_r with a specified
nonlinear reset and complete fourth-to-third source. The old mode failure
is not a general impossibility result for that representation.

The missing mechanism is not the two-product affine push or the cheap
repeated-plane read. It is an accurate, economical source of the modes
through ReLU, including what their fourth tensor emits into the retained
third tensor. No new run or rank/source sweep is proposed by this audit.

## Exact normalization and overlap

Define

    S(V)_ijkl = V_ij V_kl + V_ik V_jl + V_il V_jk,
    T4 = sum_r lambda_r S(V_r).

The archive convention Sym(A,C) is the average of the SIX cross products,
so exactly

    Sym(A,C) = [S(C+A)-S(C-A)]/12.

Thus the old full Sym(A,C) carrier is a signed two-square instance, albeit
with its two modes constrained to share the designated metric C. A radial
fourth tensor is alpha*S(C)/3. A general symmetric SSC coefficient matrix
on basis B_a can likewise be diagonalized into signed full-matrix squares
within span{B_a}; changing notation is not new information.

For symmetric V_r the requested marginal/pair reads are exactly

    k4_i = 3 sum_r lambda_r (V_r,ii)^2,
    K31_ij = 3 sum_r lambda_r V_r,ii V_r,ij,
    K22_ij = sum_r lambda_r [V_r,ii V_r,jj + 2(V_r,ij)^2],
    R_ij = sum_r lambda_r [trace(V_r)V_r,ij + 2(V_r^2)_ij].

Affine transport is V_r -> W.T V_r W. It is two dense products per full
mode, leading4*n^3 per transition, or64.42450944B per mode over15 width1024
transitions. Repeated-plane reads are O(r*n^2), whereas acquiring full
partial trace R includes the V_r^2 products. These are classical arithmetic
subtotals, not a full-card bill or native meter. They omit nonlinear source,
third descendants, base covariance, heads, masks and runtime costs.

## What was actually tried

Paths below are relative to the Phase2 experiment directory.

1. **SSC fixed local modes, not persistent free matrices.**
   `public_g22_operator_features_v1/ssc_k4_test.py` constructs nine matrices
   from true post C, K21, m, marginal third cumulants, and gates. It regresses
   45 products of next-weight quadratic forms onto next marginal k4, with
   an intercept and held-out-unit choice among four ridge penalties. It
   does not carry those matrices across ReLU, predict K31/K22, or evaluate
   a complete estimator. Original trackerT-134a R2 .02/.26/.43/.53 was
   affected by131k-label noise. Astra's `audit_ssc_noise.py` used independent
   32M labels; `ssc_noise_r35.json` gives nine-mode held-out R2
   .60794/.63687/.68075/.69298 at post4/8/12/14. Noise-adjusted values are
   .62307/.64006/.68185/.69380. This is a restricted basis/consumer result,
   not a test of arbitrary full V_r or a proof about their achievable rank.

2. **Transported full Sym(C,D), real but restricted recurrence.**
   Astra `radial_trace_matrix.py` pushes a full D by W.T D W, gates its
   off-diagonal entries and solves only its diagonal from K22 row sums.
   `RESEARCH_RECEIPTS_2026_09_04.md` records raw35/100 of3.109e-8/3.016e-8
   for carry1 and3.189e-8/2.934e-8 for carry0 on the older plain host.
   `radial_compact_fourth_matrix_v2.py` adds the full off-diagonal K22
   ridge heuristic from v1. Its preserved `compact_fourth_matrix_v1.json`
   records6.27127e-4/5.13048e-4 raw. The v2 wrapper fixes only a missing
   Gaussian-helper binding before the unchanged fixed screen resumes.
   Crucially, both inherit `k3_radial_numpy.nonlinear`: marginal k4 and K22
   consumption, NO full K31 interface or complete fourth-to-third source.
   Their failures cannot be relabeled a faithful full-modulation test.

3. **Current full-pair Sym(C,D) plus declared approximate cubic feed.**
   `radial_anisotropic_fourth_source_v1.py` writes D=radial*C+E, pushes E,
   supplies exact K4/K22/K31 for that tensor, and updates only its diagonal
   partial traces. Its new vector-matrix cubic source is projected into
   the old covariance-trace carrier; its existing CP birth column is
   shifted only to first order in E. These approximations are explicit.
   `ANISOTROPIC_FOURTH_SOURCE_ROUND_V1_CLOSED.md` records current unlearned
   raw35 2.4910582e-8 versus2.4641269e-8 and raw100 2.8609074e-8 versus
   2.6988241e-8: +1.09%/+6.01%, both failing the15% gain gate. No coadapted
   fit occurred. This prescribed source/trace implementation stays closed.

4. **Two exact full-trace reconstructions were separately measured.**
   `FOURTH_TRACE_ORACLE_VALUE_V1_CLOSED.md` fits the true post12 Euclidean
   partial trace using Sym(A,Cbank), then pushes BOTH factors through W13.
   `CANONICAL_FOURTH_TRACE_VALUE_V1_CLOSED.md` instead uses the actual
   orthogonal trace-generated Sym(A,I) projection. The same independent8M
   replicas and fixed rank35 fourth-only consumers give next-mean residual
   cross-energy ratios .6112197142/.6108077062 respectively; both fail .50.
   Covariance ratios are .6135335245/.6336143110. K3 is held fixed. These
   are isolated information screens, not general matrix-square fits or
   estimator raw scores; they do not span arbitrary independent modes.

5. **Persistent full-covariance GMM4 was run, but is a different closure.**
   `positive_gmm4_split_merge_prototype_v1.py` retains actual component
   means and full covariance factors, with a fixed positive split/merge
   source. Its whole raw test failed at2.61823e-6/4.25983e-6 against a
   single-Gaussian control2.89618e-6/4.63005e-6. Its fourth cumulant contains
   covariance-modulation terms AND component-mean/mixed terms. This closes
   the fixed no-fit four-component recursion, not every signed formal
   modulation tensor or a new nonlinear source mechanism.

## MATH already derived the transport, plus a scoped null result

`public_g19_math_acceleration_v1/LANE_h4_square_lift.md`, sections27–28,
derives the complete external free-semicircular law

    kappa4 = (4/r) D S_C[Q]

with all K4/K31/K22/q211 faces and exact congruence transport of C,Q.
It assumes ordered free-Wick and vector-isotropy identities and requires
separate quadratic-covariance Q residency. It is a conditional mathematical
replacement, not an executed arbitrary-mode closure. Sections31–32 prove
that this C-tangent law's first Edgeworth action on a zero-mean raw ReLU
pair is zero; its centered-covariance change is then rank at most two.
That theorem does NOT apply to arbitrary S(V) away from the C-tangent
subspace or automatically to nonzero preactivation means. The separate
common-scale resummation discussion is likewise one covariance ray, not
independently evolving anisotropic modulation modes.

## Material missing state and no universal closure claim

A full mode can contain all-distinct external entries and need not be low
rank as an n-by-n matrix. The old one-high external-direction obstruction
therefore cannot simply be transferred to mode count r. Conversely r is
only the rank of the UNSYMMETRIZED covariance-modulation Gram; S(V)'s
ordinary pair flattening need not have rank one.

For one mode, the quadratic ReLU contribution to third order includes

    (1/2) sum_cyclic d_i g_j g_k
            [V_ii V_jk + 2 V_ij V_ik].

It creates vector-matrix and coordinate/mixed-column third sources, not
only new K31/K22 reads. Astra already certified the corresponding Sym(A,C)
formula in `fourth_trace_full_edge_reference_v1.py`; its general exact
descendants are not owned for free by the old single covariance-trace
state. Also `FOURTH_TRACE_DISTINCT_NECESSITY_V2_CLOSED.md` shows that simply
dropping all-distinct incoming-third terms from one current fourth-trace
source leaves .87419 relative trace error and .68690 next scalar-response
error. That closes the prescribed omission, not every alternative source.

Finally, nonnegative lambda implies T4(w,w,w,w)>=0 in every direction.
Signed weights enlarge the formal tensor class but are not literally a
covariance-of-conditional-covariance law. Even with positive weights, a
physical conditional-Gaussian interpretation must separately own a PSD
conditional covariance distribution and any conditional-mean terms.

The defensible open distinction is therefore a NEW source/closure for
independent full modes and their required odd descendants. Congruence
transport, fixed SSC mode products, Sym(C,D) trace updates, or a new rank
of the same closed update are not by themselves that mechanism.
