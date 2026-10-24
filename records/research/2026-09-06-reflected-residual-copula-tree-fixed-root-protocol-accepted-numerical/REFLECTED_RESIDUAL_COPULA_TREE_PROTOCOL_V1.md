# Reflected residual-copula tree: one source-fidelity protocol

Phase 2 only: width 1024, depth 16. Draft for root review, written September
6, 2026. No acquisition, target decoding, inverse evaluation on a real
state, fit, rollout, candidate, or submission is authorized by this file.
All numerical choices below are prospective and constitute ONE arm. Once
root accepts them, failure does not authorize a bracket/order/iteration,
support, coefficient, or layer sweep.

## Question and nearest closed results

Can the current post mean/covariance and repeated third plane identify an
affordable, useful all-distinct third completion without retaining CP
history? This is a formal residual tensor completion, not an assertion
that its residual is a probability law, that the network has a Gaussian
copula, or that tree plus trace is the exact third tensor of that copula.
No independent joint observation is added; a different structural
assumption is imposed on existing causal moments.

The fixed old covariance-metric completion on rank35 post12 restored its
post repeated CP plane exactly, but its two-step weighted next-plane
error was 0.337814, against a 0.03 fidelity gate. Its 30-max reference was
0.316703. That implementation used the physical covariance correlation
R in all three legs and solved for an n-by-n mixing matrix H. This proposal
uses a different latent correlation inferred from marginal skew and
pair covariance, with diagonal curvature anchors, then restores the whole
residual repeated plane by the coordinate star. It is not the same
algorithm or a solved impossibility for that larger tensor family.

In the special Gaussian-image, zero-trace limit, shifted-ReLU marginal
inversion and exact pair inversion recover the Gaussian pre-state. The
tree below then IS the existing Gaussian age-one birth. Outside that
limit the fitted latent correlation and curvature can differ, but it
does not magically recover the lost historical information. This is the
principal scientific risk. Old age retirement measured deletion of
cohorts, not this post-moment reconstruction. The old copula-tail source
changed only a pair-covariance tail; the quadratic-skew source supplied
fourth planes from K21. Neither executed this full-third completion.

An exact quadratic Gaussian transformation already illustrates the cost
obstruction. For Y_i=a_i Z_i+b_i(Z_i^2-1), its third tensor is the tree
2*sum_cyclic b_i*a_j*a_k*R_ij*R_ik PLUS
8*b_i*b_j*b_k*R_ij*R_jk*R_ki. The latter triangle's complete next repeated
plane entails an output-dependent weighted Gram for every query,
O(n^4), approximately 2^41 classical FLOPs per full plane at n=1024.
The quadratic unit-variance skew is also bounded by 2*sqrt(2). The present
reflected-ReLU tree avoids that marginal cap, but explicitly omits all
triangle/higher connected diagrams. They are not free or secretly exact.

## Exact ownership of the prescribed residual

Use actual frozen bank rank35 post12, with its existing live mask I; do
not select a new support threshold. Let S be the physical total post K21
read from the actual stored CP atoms, decoded star, and actual trace.
Let b and B be the bank's actual post trace vector and metric:

    B = actual C with B_ii = C_ii * alive_i
    trace21_ij = B_ii*b_j + 2*B_ij*b_i
    E21 = S - trace21

The live-block auxiliary covariance is (C[I,I]+C[I,I].T)/2. This explicit
symmetrization defines only the auxiliary reference for inversion. Report
its normalized/Frobenius discrepancy from actual C; do not mutate the
bank C, its mean feedback, b, B, or next Bpre. Require all live variances
strictly positive and the auxiliary live covariance strictly positive
definite without jitter. Cold bank variances may be negative: report them,
do not floor them into a purported probability law. Cold coordinates are
excluded from the copula fit solely by the existing bank mask; their full
repeated residual remains in the coordinate star.

On I set sigma_i=sqrt(C_ii) and gamma_i=E21_ii/sigma_i^3. Thus the original
trace's diagonal skew is NOT fed into the tree and then added a second
time. A nonzero gamma uses eta_i=sign(gamma_i) and the unique alpha_i with
the centered shifted-ReLU skew equal to abs(gamma_i). Set

    tau_i = sigma_i / sqrt(v_relu(alpha_i))
    f_i(z) = eta_i*tau_i*((eta_i*z+alpha_i)_+ - m_relu(alpha_i))
    a_i = E[f_i(Z)*Z] = tau_i*Phi(alpha_i)
    q_i = E[f_i(Z)*(Z^2-1)] = eta_i*tau_i*phi(alpha_i)

q is the full second Hermite response, NOT q/2. Reflection permits either
skew sign while keeping f monotone. Exactly zero gamma uses f=sigma*z,
a=sigma, q=0, not an arbitrary small-skew cutoff. The supplied mean is
retained separately and does not affect this centered construction.

For unit-diagonal latent correlation R, infer each off-diagonal by

    C_aux_ij = eta_i*eta_j*tau_i*tau_j
               * cov(ReLU(Z+alpha_i), ReLU(Z'+alpha_j))
    corr(Z,Z') = eta_i*eta_j*R_ij.

Gaussian-limit pairs have the corresponding analytic linear inverse.
Pair feasibility does not imply that the assembled R is positive
semidefinite. Require exact unit diagonal, finite entries, abs(R_ij)<1
for distinct live coordinates, and a successful FP64 Cholesky with no
jitter, retry, eigenvalue clipping, or shrinkage. A failed numerical or
domain qualification stops this fixed arm before teacher loss decoding;
it is not a theorem ruling out every exact copula.

## One numerical spelling, qualified before teacher loss

Use root's stable marginal moments and exactly 64 bisections in [-8,12].
A nonzero requested skew outside the endpoint range is a domain failure,
not permission to change the bracket or label it Gaussian. Require
relative reconstructed absolute-skew error <=1e-8, finite positive
variance, and relative reconstructed variance error <=1e-10.

The sole affordable pair inverse is the signed degree-12 Hermite/Mehler
covariance polynomial, with 24 bisections on [-1,1] per unordered live
pair. For k>=2 the unnormalized response is

    h_ik = tau_i * eta_i^(k+1) * phi(alpha_i) * He_(k-2)(-alpha_i)
    h_i1 = a_i
    c12_ij(r) = sum_(k=1..12) h_ik*h_jk*r^k/k!

Use standardized h_ik/sigma_i for the inverse. Count and report any failed
endpoint bracket, non-finite arithmetic, failed polynomial residual, or
pair-range failure. Do not silently select another root/order. Polynomial
standardized covariance residual must be <=1e-6 per pair. Its truncated
diagonal variance is not substituted for the exact supplied variance;
R_ii is explicitly one, and post variances remain the actual ones.

Independently qualify every inferred pair with the following exact
ordinary-ReLU covariance identity, evaluated at fixed 32 and 64
Gauss-Legendre nodes on [0,1], in bounded row blocks:

    k_alpha,beta(r) = r*Phi(alpha)*Phi(beta)
       + r^2/(2*pi) * integral_0^1 (1-u)
           * exp(-(alpha^2-2*r*u*alpha*beta+beta^2)
                 /(2*(1-r^2*u^2))) / sqrt(1-r^2*u^2) du.

Apply the eta_i*eta_j reflection and tau_i*tau_j scaling outside this
identity. Exact endpoint covariance uses the corresponding 1-D truncated
normal product; Gaussian-limit pairs use their analytic values. For all
off-diagonal pairs require abs(c64-c32)/(sigma_i*sigma_j)<=1e-8 and
abs(c64-C_aux_ij)/(sigma_i*sigma_j)<=1e-6. Report Frobenius residuals too.
These checks distinguish numerical inversion error from the subsequent
latent-PD and source-fidelity questions. They do not certify the omitted
third-order diagrams. Quadrature is a one-cell qualification reference,
not part of the claimed online inverse's compute bill. A deployed inverse
would still need an honest target-independent operational domain/guard;
this one-cell pass is insufficient to establish one.

## Tree, restoration, and the actual next-bank consumer

Let U=diag(a)*R on the live block, embedded in the full width with zero
cold rows and columns. Then U_ii=a_i on I. Define the symmetric tree

    Ttree_ijk = sum_r q_r*(delta_ir*U_jr*U_kr
                         +delta_jr*U_ir*U_kr
                         +delta_kr*U_ir*U_jr)
    tree21_ij = U_ij^2*q_j + 2*(q_i*a_i)*U_ji
    A21 = E21 - tree21
    Tnew = Ttree + Star(A21) + original trace

Star is the unique coordinate-supported symmetric tensor with the
specified repeated plane and zero all-distinct entries. Therefore the
current repeated plane is exactly restored in real arithmetic. Require
relative current-plane restoration <=1e-10, with max-entry normalized
error also reported. E21 includes the previous star; do not add that old
star once more. The root certificate covers this algebra for symmetric
trace metrics. For the real bank's small covariance asymmetry, retain the
actual original trace READ and push unchanged; do not claim the rounded
metric is exactly a symmetric probability tensor.

For actual W13, write X=W13.T@U and E=A21-(2/3)*diag(diag(A21)). The
complete unmasked next repeated reads are

    tree_next = (X*X)@(W13.T*q).T
                + 2*(X*W13.T)@(X*q).T
    Q = E@W13
    star_next = (W13*W13).T@Q + 2*(W13*Q).T@W13

Apply the original pre13 live mask to the off-diagonal non-trace sum and
retain its full diagonal, exactly as the bank does. Then add the actual
original pre13 trace read Bpre_ii*bpre_j+2*Bpre_ij*bpre_i. Bpre and bpre
must be captured at that precise bank point or reproduced with the
identical bank arithmetic, not replaced by an FP64 symmetrized product.
All mean, C, radial fourth, P/Q heads, gates, and next support are held
fixed. No tree output is fed into the recapture.

The sole information test uses rank35 post12 -> pre13 and query rows
0,32,...,992, all 1024 columns. Freeze tree parameters and its predictions
before decoding the existing pre13 K21 loss array. Use the actual bank
pre13 alpha/sigma to define the same Gaussian-pair response weight formula
as the old completion test:

    weight_ij = 0.5*phi(alpha_i)/sigma_i * Phi(alpha_j)
    fidelity = ||weight*(new_K21-old_actual_K21)||_F
               / ||weight*old_actual_K21||_F

The weight formula/rows match that earlier test; its old symmetrized FP64
covariance replay is not silently treated as byte-identical actual-bank
weights. Require a finite nonzero denominator and fidelity <=0.03. Report
unweighted error, diagonal-only error and finite numerical norms as
diagnostics, not alternative selection gates. This is an intentionally
strict replacement-fidelity test, NOT a mathematical prerequisite for
lower raw ReLU-mean MSE. Failure closes this exact numerical/completion
arm without more data, co-adaptation, variant search, or raw scoring. A
pass only earns a separate root review of full-card cost, domain and
co-adaptation; it is not an accuracy result or rollout authorization.

## Missing state, one coherent recapture, and causal seals

The existing rows_two_memory_teacher_action_v2 post_12.npz saves only
layer, m, variance, physical_K21, alive and radial. It lacks full C, b and
B. check_covariance_metric_completion_wide_v1.py captured these transiently
but did not persist them. The pre8..14 trace/meanmix cache belongs to a
different, oracle-corrected coherent trajectory. None may be substituted
for actual autonomous-bank post12 trace ownership.

Accordingly the one proposed acquisition is an observation-only rank35
recapture, using the identical frozen current-bank source, weights and
P/Q payloads and the v2 fixed2048-column FP64 CP diagnostic accumulation.
It adds only post12 C/b/B and exact pre13 Bpre/bpre to a NEW versioned
artifact. The actual post12 m/variance/physical_K21/alive and original
prefix/final results must match the qualified v2 receipt; preserve its
bank byte-parity gates. No P/Q, bank arithmetic or original file changes.
Separate causal input fields from the sealed pre13 action loss fields.

Before any inverse evaluation, hash the new protocol, all transitive
capture/reference sources, weight and P/Q files, original v2 artifacts,
and their receipts. Require original capture/qualification passed with
peak RSS<=6 GiB. Hash and preserve newly captured inputs and receipt.
After numerical/PD qualification, write a frozen predictor artifact and
its hash before loading K21_loss_only. Then compute the single declared
fidelity. Recheck all hashes at exit; preserve any failure receipt and
partial artifacts, never overwrite/replay to conceal a failure. No MC,
true means, private labels, fitted gains, or current-layer corrected
targets as predictor inputs. The exact current S is a causal post12
state, whereas pre13 K21 is solely the next-action teacher label.

Acquisition requires root's separate go: one nice10 NumPy-only process,
<=2 BLAS threads, no Torch/MPS, <=6 GiB RSS. The bounded inverse/action
judge should use <=2 GiB with row-blocked pair operations. No job is
launched by this document. No forum writes or submissions.

## Complete arithmetic ownership, not a measured runtime claim

The tree affine action costs three square matrix products, the restoring
star three, and ordinary covariance two per hidden transition. The
original trace reuses the candidate's active covariance product; its cold
diagonal remainder and vector push remain explicitly present. This reuse
must be demonstrated in a later candidate port, not inferred from the
NumPy reference which can compute the trace covariance a second time.

Conservatively pricing all 15 transitions at full width, plus the initial
Gaussian covariance, gives 121 square products: 259,845,521,408 classical
multiply/add FLOPs. Fifteen FP64 Cholesky guards at twice an n^3/3 cost add
10,737,418,240 metered-like FLOPs. The subtotal is 270,582,939,648, about
0.12305 of 2^41. This subtotal assumes the eventual large products use the
current candidate's FP32 policy; new tree/restoration FP32 fidelity has
NOT been established by the FP64 information diagnostic. Retaining FP64
for those products would double their metered product cost and erase
the claimed arithmetic advantage. It EXCLUDES the following explicitly
owned costs:

- 15*(24*12)*n*(n-1)/2 standardized polynomial terms for pair inversion,
  about 2.26 billion terms, plus their powers/normalization/bracketing;
  FP64 operations receive the applicable multiplier. This is O(n^2) but
  not free. Marginal inversion is 15*64*n scalar moment evaluations.
- The existing Gaussian Hermite pair map, cubic and radial fourth maps,
  trace/vector/elementwise updates, mask handling, and unchanged P/Q heads.
- Copies, allocations, and return conversion, and any genuine deployed
  inverse-domain guard. The 32/64-node diagnostic is not billed as online
  work unless it is actually retained in deployment.

Thus 0.12305 is a product/factorization subtotal, NOT a complete utilization
or runtime prediction. A complete cost listing plus a flopscope/runtime
port is required before promotion. The state itself is O(n^2): covariance,
tree matrix, residual star, b and scalar/head state; transient CP history
is used only for this teacher recapture, not the proposed replacement.
The earlier release requirement remains >=20% whole-bank compute savings
on both ranks35/100, then 14-net raw/adjusted and persistent runtime gates.
Neither a score below1e-9 nor enough whole-card savings is established here.

## Existing synthetic receipts and primary background

Root's reflected_copula_tree_action_certificate_v1.json passed ten
reflected Hermite endpoint cases (worst normalized absolute1.853e-16) and
12 tiny tree/trace/star-action fixtures (worst relative3.476e-15).
Its source SHA is
42b1cb038d10bdab94aeb15bfeadbcae5cf26ab9d1e92dd6700d5157f7b8874b;
receipt SHA is
083d38baf0fceca2f32680c45f6ec9fb2a218d150a41c13bdb51b6964e84b537.
That .262s synthetic run was not priority-qualified because setpriority
was denied; it involved no network/targets/card worker. Do not rerun it.

Root's reflected_relu_marginal_certificate_v1.json independently checked
tail-integral moments (worst relative2.987e-10) and fixed-bracket alpha
recovery (worst4.095e-12). The derivative's strict sign is proved by
Cauchy-Schwarz in REFLECTED_RELU_SKEW_INVERSION_ALGEBRA_V1.md; sampled
evaluations of that derived derivative are not independent finite
differences. Source SHA
f77b5ac71b08a172ec5cf481d4a0a8ea9387f9039cacf276aa14886e9acfa4a1;
receipt SHA
827eb01d67affe38627e5d2e816ee87216a927cac388609f10a4a46575d15fcf.
No inversion/PSD/source/score claim follows from either certificate.

Hermite/Mehler correlation inversion is primary background, not a novel
identity attributed to this project: [Qing Xiao, 2016](https://arxiv.org/abs/1608.00738).
The global-feasibility caveat is also standard; the author's
[NORTA software discussion](https://people.orie.cornell.edu/shane/NORTASoftware.html)
explicitly describes correlation targets that the transform cannot match.
No external repair method is imported into this fixed no-repair protocol.
