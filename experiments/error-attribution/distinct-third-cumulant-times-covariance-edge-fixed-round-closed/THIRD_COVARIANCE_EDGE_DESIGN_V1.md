# Distinct third tensor times one covariance edge: fixed source test

Phase 2 only, September 5, 2026, 17:23 PDT. A new omitted source, NOT the
closed Gaussian triangle, two-edge hypertree birth shift, or projection
of the entire third tensor into q4 modes. No old family is reopened.

Primary factor_k3.py explicitly drops the augmented kappa3-times-edge
diagrams; kprop_harmonic.factored_keeps_term excludes them because a
three-support block and two-support block overlap in two vertices. Our
current inherited cubic scales by g_i*g_j*g_k and also omits this term.
For all distinct indices, the missing source is exactly

  D_ijk = Tdistinct_ijk * (C_ij*d_i*d_j*g_k
                         + C_ik*d_i*d_k*g_j
                         + C_jk*d_j*d_k*g_i),

where d=phi(mu/s)/s, g=Phi(mu/s). All repeated entries are zero; the
existing post repeated-plane restoration remains unchanged. This is one
specified primary diagram class, not the complete augmented algorithm.

First certify the coefficient against the primary un-factored diagram
enumeration at widths8/16. Certify the exact next-K21 query against a
small dense tensor independently, including the current CP, pushed
coordinate-star and covariance-trace contributions and repeated removal.

Then use the current coherent oracle-previous-mean/covariance source-v2
protocol, rank35 L12 followed by rank100 L12 only if35 passes. Select the
32 next-output repeated indices 0,32,...,992 BEFORE reading targets.
Compute the exact full-covariance-edge next K21 rows without materializing
the full n^3 tensor. It costs two large CP matrix products per selected
row and is explicitly an OFFLINE information diagnostic, not deployable.
Compare natural-strength source addition to the model's same current
post tensor against existing documented8M pre K21 at L13. Also report
leading-consumer-weighted error using .5*d_next_i*g_next_j. No fitted
slopes. Same-sample oracle states mean this is not an endpoint score.

Research gate: natural coefficient1 reduces the weighted K21-error MSE
by at least5%, has weighted cosine with the missing plane >=.4, and does
not worsen unweighted K21 error, on BOTH nets. If the first net fails,
stop without the second. In parallel within the SAME queries, evaluate
only the fixed top4 covariance-eigenmode edge approximation. Its source
response relative error must be <=15% (weighted and unweighted) to pursue
that specific cheap representation. No q8/q16 sweep on failure.

If the information gate passes, a complete cost/memory plan (including
source-specific compression if needed) is required BEFORE a rollout or
training. q4 times the entire CP bank is too large to assume affordable.
No new training, bank, score, panel or submission is authorized by a
static information result alone. All existing submission authority and
release gates remain unchanged.
