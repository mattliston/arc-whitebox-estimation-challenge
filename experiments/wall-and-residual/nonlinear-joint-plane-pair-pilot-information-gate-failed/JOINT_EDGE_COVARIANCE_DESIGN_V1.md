# Nonlinear pair closure with an explicit joint-third input

Phase2, September5,2026. One fixed source-information pilot, not deployment.

Previous covariance pair regression had14 marginal columns and rho powers,
but NO K21 pair input. The48-feature joint-plane mean probe linearly reduced
nodewise summaries, not a nonlinear full-pair covariance write. The
quadratic-skew proxy used one inverse-R Gram and three scalar coefficients;
it was not this pair model. All those failures stay closed.

Test a symmetric 16-hidden-unit tanh edge network. Its affine hidden input
is a14-feature node embedding for i, another for j, rho_ij, S_ij and S_ji,
where S_ij=K21_ij/(sigma_i^2*sigma_j). Node embeddings are formed BEFORE
broadcasting; no n^3 learned kernel. The output is symmetrized with fixed
phi_i*g_j response factors and scaled by sigma_i*sigma_j. Zero-initialized
output gives exactly the unmodified covariance initially. Compare to the
identical architecture/initialization with both S inputs zeroed. No deeper
network, basis/rank/epoch sweep or ensemble.

Current source-v2 coherent8M mean/covariance trajectories, layers8..14.
Train35/100/150/200/250/300; held-out400/500. Freeze S before this layer's
target feedback and require all old14 features, rho, sigma, covariance
target, next weight and response BYTE IDENTICAL to the original v2 rows.
No new Monte Carlo, hidden/gated nets, or14-net grader-panel fitting.

Fixed seed5290826,40 epochs, Adam lr .005 cosine to .0005, weight penalty
.001 mean parameter-square. Loss equally weights covariance MSE and the
fixed next-Gaussian-mean response MSE, each normalized by training baseline
energy. Feature scales use only the six training networks. Batch is one
network/layer; random order is fixed and shared between full/control fits.
Final epoch only decides; no checkpoint selection on holdouts.

Information gate: BOTH heldouts must have covariance and next-mean ratios
<=.75, and the joint-input model must beat the zero-S control by at least
10% in next-mean error on EACH. Otherwise close this fixed nonlinear pair
pilot. No200-net labels, fit expansion, rollout, candidate or bank on fail.
The raw source pilot has oracle previous state and is not a deployable
or noise-corrected score. A pass still requires coadapted closed-loop and
all independent14-net, bill, memory and persistent-runtime release gates.

Conservative architecture envelope: sixteen hidden pair maps, roughly
16 tanh charges plus <400 elementary operations per pair/layer, seven
layers: <5e9 extra float32 FLOPs (<1.1% of the present bank). This is a
shape-only pre-gate, not native billing or wall-time qualification. One
MPS job at a time; at most two nice10 CPU row workers with two BLAS threads.
