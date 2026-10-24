# CP-free nonlinear two-memory student: bounded proposal v1

Phase 2 only. Written September 6, 2026 (clock read 20:40 UTC), before any
new execution or fit by this agent. This is ONE proposed family and ONE
small capacity falsifier, not permission to train a full card. No energy
A/B arrays were read. Existing sources and failures remain unchanged.

## Decision and novelty

Try replacing the entire growing CP bank with a nonlinear recurrent state
containing a repeated-third plane D and ONE additional full latent matrix H.
First ask whether H can recover missing bank third actions from lawful
history, using noiseless deterministic bank targets. Do not yet train to
Monte Carlo truths or advertise a score.

This is not a claim that two matrices determine a general third tensor.
The all-distinct non-identifiability obstruction remains. The hypothesis
is instance-family compression learned through a nonlinear, weight-driven
recurrence. A failure of the small action test ends this fixed proposal.

Nearest closed work:

- Plane-only/star carry lost important all-distinct information. D alone
  is explicitly the negative control, not a rediscovered solution.
- MATRIX_ERROR_OBSERVER_V1_CLOSED used a *linear Gaussian-tangent error*
  recurrence and a fixed 14-feature bilinear source, added to the full bank.
- LEARNED_PAIR_HISTORY_INFORMATION_V1_CLOSED decoded endpoint histories;
  it did not evolve a nonlinear full pair field that replaces CP inference.
- LEARNED_JOINT_GRAPH_PROBE_V1_CLOSED used one-hop messages into a scalar
  head, not nonlinear pair-state transport or noiseless action distillation.
- Full-bank P/Q fitting changes the readout of an expensive fixed carrier.
  This proposal changes both retained information and deployment cost.

These distinctions establish novelty of the spelling, not likely success.
There is currently NO measured raw gain or compression capacity result.

## Exact transport and explicit state ownership

At a post-layer boundary keep m, PSD covariance C, standardized repeated
third plane D, and dimensionless latent H. D and H are n by n; H is NOT a
cumulant. Let s_old=sqrt(diag C). At the next affine layer:

    mu = W.T m
    Cpre = W.T C W
    s = sqrt(diag Cpre)
    V = diag(s_old) W diag(1/s)
    rho = Cpre / outer(s,s)

The normalized coordinate-star tensor encoded by D is

    E = D - (2/3) diag(diag D)
    T_ijk = delta_ij E_ik + delta_ik E_ij + delta_jk E_ji.

Its complete repeated-plane affine action, including diagonal restoration,
is exactly THREE matrix products:

    Q = E V
    Kstar = (V*V).T Q + 2 (V*Q).T V.

This preserves the three odd legs. It is not replaced by V.T D V.
Root independently certified this identity on 12 small directed,
rectangular, permutation and positive-rescaling fixtures; worst error
1.036e-15, in two_matrix_star_transport_certificate_v1.json.

The second channel has the explicitly phenomenological TWO-product update

    U = V.T H V.

It is a transported latent pair field; it has no asserted tensor-order
interpretation. Its ability to encode omitted third information is the
thing to test, not an assumption made to justify this transport.

## Fixed nonlinear decoder/source

One shared edge MLP, 16 -> 8 -> 8 -> 2, with ReLU hidden activations.
For ordered pair (i,j), its 16 dimensionless inputs are:

    alpha_i, alpha_j, gate_i, gate_j, pdf_i, pdf_j, rho_ij,
    Kstar_ij, Kstar_ji, U_ij, U_ji,
    Kstar_ii, Kstar_jj, U_ii, U_jj, layer/15.

Here alpha=mu/s and gate/pdf are the Gaussian functions. Clip ONLY these
MLP inputs to [-8,8]; do not clip the physical Kstar action. There are no
truth values, current teacher K21 values, inherited CP factors, neuron IDs,
network ranks, target-derived quantiles, or old P/Q outputs in this vector.
The MLP emits f_D,f_H, and defines

    Dpre = Kstar + 0.05*tanh(f_D)
    Hpost = tanh(U + rho*rho - diag(diag(rho*rho)) + f_H).

The fixed rho-squared seed is an already known Gaussian pair feature, not
claimed new physics. The new hypothesis is its nonlinear persistent use
and the learned decoder of H into an odd repeated-plane correction.

Dpost has a precise physical warm start: compute the Gaussian-plus-FIRST-
ORDER-cubic bivariate ReLU map on (mu,Cpre,Dpre), with fourth inputs zero,
and normalize its returned post-K21 by outer(s_post**2,s_post). Use the
current bank's Gaussian Mehler order 6 / cubic-edge order 2 convention as
the declared approximation. Remove the unused K22/radial products rather
than silently carrying an old fourth bank. Equivalently, its raw moments
use G_pq plus

    [k_iii*d_i^3 + 3*k_iij*d_i^2*d_j
     + 3*k_ijj*d_i*d_j^2 + k_jjj*d_j^3] G_pq / 6,

for Gaussian ReLU raw moments G_pq; d_i differentiates pre-mean mu_i.
Centralize the raw (2,1) moment consistently with that map's own mean.
This formula specifies the source; it is not a declaration that truncated
Edgeworth is a positive law or that Mehler order 6 is exact.

Input initialization is m=0,C=I,D=H=0. The first layer is exactly Gaussian
before ReLU. Its mean, covariance and post-K21 may be initialized directly:
q=1/sqrt(2*pi), h(r)=[sqrt(1-r*r)+(pi-acos(r))*r]/(2*pi),

    m_i = sigma_i*q
    C_ij = sigma_i*sigma_j*(h(rho_ij)-q*q)
    K21_ij = sigma_i**2*sigma_j *
             [q*(rho_ij + rho_ij**2/2 - 2*h(rho_ij)) + 2*q**3].

This exact zero-mean Gaussian initialization needs a separate tiny formula
certificate before use. H after layer 0 uses the same stated nonlinear
source; no learned layer-0 scalar correction is added.

## Covariance/mean policy and the extra cost it creates

The small capacity test below TEACHER-FORCES m,C,D at previous boundaries;
it does not pretend these are solved student outputs. A later full student
must implement the following declared PSD policy, not signed covariance
edits followed by free eigenvalue repair.

Let C_G be Gaussian ReLU covariance from a finite positive-coefficient
Mehler sum, with its nonnegative diagonal tail restored, and R_G its
correlation. This is PSD in exact arithmetic by the Schur product theorem.
Let Hpost be the latent matrix above and form the EXTRA full product

    G_H = Hpost Hpost.T + 1e-6 I
    R_H = G_H / sqrt(outer(diag G_H,diag G_H))
    Rpost = (1-beta_layer)*R_G + beta_layer*R_H,  0 <= beta_layer <= 1.

beta is a bounded learned scalar, initialized to zero. It is a model
parameter; parameter projection during offline optimization is not a
runtime covariance repair. This convex mixture is PSD with unit diagonal.
A later fixed small node head can set m_post and strictly positive
v_post, using the stated local scalars and diagonal/row-norm summaries of
Dpre,Hpost, then Cpost=diag(sqrt(v_post)) Rpost diag(sqrt(v_post)).
The node head is not needed or fitted in the first capacity experiment.
Its exact specification would require a separate review before a full fit.

Means and central moments define the modeled raw second moment as
Cpost+m_post*m_post.T. Do not reuse a frozen raw-second matrix after a
mean update, or borrow the old P/Q variance-head semantics. This is a new
approximate state, not byte parity with the full bank. Strictly nonfinite,
invalid variance, or unsafe PSD numerical behavior closes the attempt; no
unpriced eigendecomposition, adaptive fallback or moment repair is assumed.

## Complete deployment envelope, not a measured bill

For n=1024 and 16 layers, retain classical dense FP32 arithmetic; do NOT
assume any D5/D6/Smirnov speedup. The fixed seven-product skeleton is:
2 covariance + 3 star + 2 latent products on each of 15 transitions,
plus the first W.T W product. Root's arithmetic gives

    212*n**3 = 227,633,266,688 FLOPs = .103515625 of 2**41.

Budgeting the PSD Gram for all 15 later layers adds 30*n**3, giving

    242*n**3 = 259,845,521,408 FLOPs = .1181640625 of 2**41.

The edge MLP has 450 ordinary arithmetic/ReLU operations per ordered pair
before input/output scaling, about 7.55B across 16 full-width layers.
Reserve a TOTAL 20B additional charge for that MLP, Gaussian/cubic pair
maps, all scalar functions, matrix/scalar normalization, parameter/input
ingress, copies, bounded node heads, masks, and output conversion. This
20B cap must be certified by a complete operation census before any port;
it is an engineering envelope, not yet established by FNP metering.

Thus the proposed go/no-go cost ceiling is 279,845,521,408 FLOPs,
utilization <=.12725901. No old CP/P/Q estimator runs alongside it at
deployment, and no runtime teacher, fit, Jacobian or eigensolver is free.
If the complete census exceeds this fixed ceiling, return to root before
any training; do not quietly expand the budget.

State proper is three matrices (C,D,H) plus vectors, ~16 MiB with C float64
and D,H float32; weights ~64 MiB. A conservative implementation allowance
of 32 working n*n arrays plus one 16-channel edge feature block is below
300 MiB of array payload. Training/autodiff is a separate, larger resource
account. Evaluate edge features in fixed row blocks if necessary; charge
all resulting copies/calls. No runtime bound follows from FLOPs alone:
the eventual persistent exact-backend gate should require <=60s maximum
total wall, <=.25s residual wall and <=2GiB process RSS on every tested MLP.
These are prospective safety targets, not measured promises.

At u=.12725901, bank raw 2.244835226e-8 would give about2.857e-9 adjusted.
Below4e-9 requires raw<3.1432e-8; below3e-9 requires raw<2.3574e-8;
below1e-9 requires raw<7.858e-9. Distillation alone cannot promise the last
threshold. For teacher error e and student-minus-teacher discrepancy d,
raw_student=raw_teacher+2*mean(e*d)+mean(d*d); disagreement cannot be
treated as independent noise. A sufficient norm bound is
sqrt(raw_student)<=sqrt(raw_teacher)+RMS(d), but actual panel raw is the
ultimate criterion. No local action percentage substitutes for that test.

## ONE smallest decisive capacity/source falsifier

Root approval required. No MC acquisition, full-card fit, new labels,
native port or submission is included.

Use exactly the existing coherent current-bank teacher artifacts for
35/100/150/200/250/300 as train and 400/500 as held out, layers 1..14.
If these exact bank D/post-C boundary artifacts are absent or have mixed
ownership, STOP for a coherent target-free recapture proposal; do not
silently substitute older teacher-forced truth-state files.

At each boundary teacher-force ONLY the PREVIOUS bank m,C,D, with current
mu/Cpre derived from them and W. H is always recursively model-generated,
starting at zero. Compute Kstar and U by the equations above. Predict the
current bank standardized pre-K21 using Dpre. The target current pre-K21
is used ONLY in the loss and evaluation, never in features, H update or
normalization. Thus the experiment grants lower-moment/plane history for
free as an explicitly optimistic capacity test, not a deployment claim.

One 16->8->8->2 model, zero initial D decoder output; fixed seed9062047;
100 Adam epochs, learning rate1e-3, no scheduler, no checkpoint selection,
no train/held-out bin or rank search. Exact epoch/time cap and batch/RSS
must be scheduled by root before execution. Normalize objectives by
TRAIN-only zero-decoder energies, not target-dependent per-entry scales.

Report full-plane MSE and a JOINT local consumer test. Holding all other
current-bank inputs fixed, substitute Dpre versus target pre-K21 into the
same frozen bank Gaussian/cumulant pair map and actual current head.
Report both immediate post-mean difference and the resulting full
covariance difference DeltaC propagated to the next pre-variance:

    Delta_v_next = diag(W_next.T DeltaC W_next).

This next-variance calculation is charged offline (one full product plus
column reduction); it is not an extra student inference operation. It
retains mixed covariance pathways missing from scalar cubic lookahead.
No same-layer corrected state may leak into predictor features.

To warrant further review require, on EACH held-out net, final-epoch
ratios <=.10 relative to the D-only/no-latent decoder for ALL THREE:
full-plane error, immediate mean-consumer error, and next-variance error.
Also report absolute errors and every layer; a near-zero denominator is
inconclusive, not a manufactured pass. A fixed matched D-only decoder
with the same MLP but H/U and H summaries replaced by zeros is the
capacity control; it is not permitted to use current teacher K21 either.

Failure closes this fixed architecture/source/width/optimizer round;
no more epochs, channels, rank/basis/seed sweep or full fit follows.
A pass merely establishes a promising conditional capacity witness.
It still requires endogenous m,C,D rollout and its actual final raw
discrepancy, complete metering, native numerics, both14-net panels and
persistent wall/RSS before banking anything.

## Rejected leads from this audit

- Deeper Winograd with generic batching is already implemented/tested.
  Existing D6/streamed6 leaves in leaf_runtime_benchmark_v2.json are much
  slower than D5; the current bank already batches recursive subproblems.
  The raw (7/8)^2 multiplication factor omits additions, copies, padding
  and time. No repeat benchmark is proposed.
- Ordinary MC, low-degree/shallow polynomial CVs and spectral sample
  shrinkage retain their archived failures. General neural Stein methods
  exist ([Hasan et al.](https://openreview.net/forum?id=by9cKVIs0F)), but
  this audit did not construct a cheap field with the required variance
  reduction. A formal OU inverse or an unknown exact surrogate is not a
  new implementable estimator and is not offered as a second experiment.

Only the bounded two-memory capacity proposal above is recommended for
root's next review. Its chance of success is unmeasured.
