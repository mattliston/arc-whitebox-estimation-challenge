# Fixed-count within-cohort CP pair retention: target-free prerequisite

September 6, 2026 PDT. Phase 2 only, current full-pair P0 + scalar final Q,
fixed development ranks 35 and 100. No target access, fits, actual random
draws, carrier edits or submission. This is distinct from independent
Bernoulli retention: selecting exactly m of N signed pairs in a birth cohort
induces negative correlations and cancels common cohort fluctuations.

For final pair effects v_j = J a_j, where J is the certified diagonal plus
rank-one final k3 Jacobian and a_j is the complete signed-pair k3 atom, let

    S_l = sum_j ||v_j - mean(v)||^2 / n,
    A_l = N_l^2 S_l / (N_l - 1).

Uniform sampling without replacement of exactly m_l pairs in cohort l,
with unbiased weight N_l/m_l, has local final-output MSE variance

    V_l(m_l) = A_l (1/m_l - 1/N_l).

Different cohorts are sampled independently. At exactly half the total
number of pairs, optimize the CONTINUOUS allocation (an optimistic lower
bound on the integer-allocation problem):

    m_l = clip(sqrt(A_l)/tau, 1, N_l),  sum_l m_l = sum_l N_l / 2.

Use fixed 100 bisection steps. Report both uniform half-per-cohort and the
optimized continuous allocation. Gate: optimized local added final MSE
<= 1e-9 on BOTH ranks, unchanged from the independent-retention screen.
All earlier-layer effects, allocation acquisition, weight feedback,
sample-index construction, cost and runtime are free in this prerequisite.
Passing permits only a new whole-loop cost/error design. Failing closes
this within-cohort uniform fixed-count sampling proposal, not arbitrary
direction-matched correlated or deterministic compression.

Before wide execution certify the formula against exact enumeration of
two tiny independent cohorts, including a large common component. Check
the constrained continuous solution by KKT residuals and objective against
the uniform allocation. The signed-pair identity and J are re-certified by
the preserved paired_cp_thinning_bound_v1 helpers on one COHERENT recapture
per rank. Do not mix the older NumPy cohort sums with MPS pair norms: these
have the same model, but not identical floating-point factor trajectories.
Require unchanged full-forward <=2e-6 and all 30 active/birth memberships.

Use one nice-10 worker. Tiny mode uses NumPy and one CPU thread, no Torch or
MPS. Wide mode uses the sole MPS slot only after root explicitly releases
it; at most three CPU threads, no threadpoolctl and no MPS fallback. Archive
source/output hashes, wall time and separately qualified process/GPU memory.

