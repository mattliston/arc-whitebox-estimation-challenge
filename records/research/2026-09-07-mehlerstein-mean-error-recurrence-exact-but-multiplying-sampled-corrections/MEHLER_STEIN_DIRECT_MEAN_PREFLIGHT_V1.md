# Direct Mehler--Stein mean correction: source and cost preflight

September7,2026 PDT. Phase2 only. Root and the existing accuracy agent
independently derived and checked the identities below. No numerical
experiment, target-array access, fit, new candidate or score. This note
does not reopen the unconditional Dirichlet or earlier Stein-control arms.

## Decision before a substantial experiment

Two routes were investigated: a new direct raw-information statistic and
the historical deeper-kernel/current-runtime compatibility question.
The second is recorded separately in
PUBLIC_REFRESH_AND_DEEPER_RUNTIME_SCOPE_2026_09_07_V1.md.

For the raw route, the supporting evidence is qualified local joint-K3/K4
oracle value, not a final-score forecast. On rank35 source layers8..14,
CURRENT_PAIR_ORACLE_ATTRIBUTION_PARTIAL_V2.md records next-mean-response
ratios .3275057 for off-diagonal K3 and .0209139 for jointK3/K4. Its rank100
pooled gate did not finish. The narrower q4 bridge then failed its ACTUAL
value gate: next-mean .765031/.777423 and physical-source .886166. The old
.3154 consumer-weighted tensor ratio is not permission to build that carrier.

The new statistic below can access actual joint gate-gradient information
without fitting a conditional density. Its expected Phase2 raw gain is
UNKNOWN. Complete cost is not available because the source/control,
sampling and multilayer product-bias policy are not specified. Explicit
leading circuit costs and necessary variance thresholds are given below,
not substituted for a complete bill.

Selected cheapest work: derive the direct expectation correction, location
ownership and paid probe-variance requirement. This source/algebra preflight
is complete. Neither a predictive variance receipt nor sufficient accuracy
evidence currently justifies a network experiment. No run is forced.

## 1. Actual-law first-derivative identity

Let A=a(X), X standard Gaussian, and X_t=tX+sqrt(1-t^2)Z with independent Z.
For a finite ReLU prefix, a is globally Lipschitz. Define

    H(X)=integral_0^1 E_Z[gradient a(X) dot gradient a(X_t)] dt.

Gaussian first-order integration by parts gives

    E[(A-E A) g(A)] = E[H(X) g'(A)].                    (1)

Conditioning H on A would produce a scalar Stein kernel, but is unnecessary
for (1). The Malliavin/Ornstein--Uhlenbeck background is the auxiliary
variable in [Nourdin and Viens,2009](https://www.maths.tcd.ie/EMIS/journals/EJP-ECP/article/view/707.html).
The finite-dimensional formula and application here were checked directly
by Gaussian covariance interpolation; no density theorem is assumed.

Unlike the old unconditional Dirichlet Gram, H retains cross-input,
cross-depth gate dependence. Replacing the random products by factorized
Gaussian gate averages would discard precisely that proposed information.
This is an information distinction, not evidence of useful sample variance.

## 2. Exact ReLU correction, including reference-location mismatch

Fix m_ref, v>0, sigma=sqrt(v), alpha=m_ref/sigma and

    M_ref=m_ref*Phi(alpha)+sigma*phi(alpha).

The bounded Stein solution to

    v*g'(s)-(s-m_ref)*g(s)=ReLU(s)-M_ref

can be written, with z=(s-m_ref)/sigma, as

    s<=0: g(s)=-(M_ref/sigma)*Phi(z)/phi(z),
    s>=0: g(s)=-1+((M_ref-m_ref)/sigma)*(1-Phi(z))/phi(z).

The branches and first derivatives match at0. Stable tail evaluation would
be an additional numerical requirement; literal ratios are not a qualified
implementation. Applying (1) gives

    E ReLU(A)-M_ref
      = E[(v-H)g'(A)] -(E A-m_ref)*E[g(A)].             (2)

The negative location term is essential. Exact supplied location makes it
zero; a reference-card location does not. Constant Stein kernel permits
Gaussian translates, but that fact alone is NOT an obstruction for a
propagator which separately carries the location.

No density or second derivative of a is used. A bias-free ReLU prefix can
have an atom A=0, for example from an all-dead earlier layer. On a
positive-measure constant region gradient a=0 almost everywhere and H=0.
The correction still includes v*g'(0); no atom is silently dropped.

## 3. Exact error recurrence around any deterministic reference card

Let b_l be the card's served reference means, d_l=E h_l-b_l, and
m_l=W_l.T*b_(l-1). Use the actual preactivation A_l, any deterministic
positive v_l, and the corresponding coordinatewise Stein solutions. Put

    G_l=E[g_l(A_l)],  R_l=E[(v_l-H_l)g'_l(A_l)].

With zero reference and true input means, d_input=0. Exactly,

    d_l=M_G(m_l,v_l)-b_l+R_l
          -G_l Hadamard (W_l.T*d_(l-1)).               (3)

Thus the location mismatch can be owned recursively, rather than asserted
away. The Gaussian mean M_G need not equal the served learned b_l. The
reference card and all its acquisition are still paid work.

Replacing expectations in (3) by same-path sample means is NOT automatically
unbiased: estimated G_l multiplies an estimated earlier d, and covariance
terms recur through depth. A valid design must account for their bias and
the full covariance of all correction sources, or price an independent/
cross-fitted construction. No such design is presently qualified.

## 4. Full matrices are not the only gradient-product realization

An exact full cross-gradient state obeys

    K_l(x,y)=D_l(x)*W_l.T*K_(l-1)(x,y)*W_l*D_l(y).

At final PREactivation15, one common first Gram costs2*n^3 classical FP32
FLOPs. Per input pair, fourteen intermediate two-sided pushes and the final
diagonal-only contraction cost58*n^3=62,277,025,792 leading FLOPs at1024.
Forwards, gates, integration, conditional estimation and all other work are
additional. This is a declared circuit subtotal, not a universal bound.

For p=gradient a_j(X), q=gradient a_j(X_t), an independent Gaussian probe e
instead supplies (p dot e)(q dot e), unbiased for p dot q. Its conditional
variance is

    ||p||^2*||q||^2+(p dot q)^2.

Rademacher probes subtract2*sum_i p_i^2*q_i^2. These products can be obtained
for all neurons using two shared directional forward passes, not1024
reverse-gradient runs. Independent probe averaging divides probe variance
by the probe count, but does not remove Mehler/input randomness.

For a static weights-only gradient control beta_j, the probe control

    Hhat=(p dot e)(q dot e)-(beta_j dot e)^2+||beta_j||^2

remains unbiased for p dot q. Its Gaussian conditional variance is

    2*||sym(p*q.T-beta_j*beta_j.T)||_F^2
      =||p||^2*||q||^2+(p dot q)^2
        -4*(p dot beta_j)*(q dot beta_j)+2*||beta_j||^4. (4)

This is a probe-only control: it does not remove the actual-input or
Mehler randomness. Its construction and projections are charged. More
general controls beta(X),gamma(X_t) require a paid exact inner product
and independence from the probe used for their estimation.

An affine Gaussian normal form clarifies why this distinction matters.
When p=q=beta is the exact fixed affine gradient, (4) is identically zero
and H=v. Without control even that Gaussian case creates probe variance
2*v^2*g'(A)^2 in a correction whose expectation is zero. This is an exact
sanity witness, not a Phase2 capture measurement.

## 5. Probe variance times paid cost cannot be tuned away with more probes

Under the conventional2*n^2 FP32 matvec count, for one final-site control:

- Two actual16-layer forwards:64*n^2 per input pair.
- Per probe, first W0.T*e shared between replicas plus30 later tangent
  products:62*n^2.
- One stored dense control B.T*e:2*n^2 per probe.

Hence per probe64*n^2=67,108,864 leading FLOPs, and one-probe plus forwards
134,217,728. Add input and probe RNG, Mehler mixing or integration,
activations, scalar Stein functions/tails, control/reference acquisition,
all-layer return, reductions, buffers, ingress, casts and initialization.
Separate controls at all16 recurrence sites are NOT supplied by one final
B.T*e. A sharing/first-layer-analytic policy needs its own complete ledger.
No native bill or runtime/RSS qualification exists for this new circuit.

Let Vp be the mean across output neurons of

    E[2*g'_j(A_j)^2*||sym(p_j*q_j.T-beta_j*beta_j.T)||_F^2].

With r independent Gaussian probes and IID outer draws, the single-site
statistic's conditional probe variance is Vp/r. Even granting setup and
every other noise source free, above-floor adjusted contribution from
this noise is at least

    Vp*(64*n^2)/2^41.

Therefore strict1e-9 requires Vp<3.2768e-5 for THIS conventional circuit.
At one probe, the analogous complete-statistic variance yardstick before
extra work is1.6384e-5. Neither quantity was measured here. Faster matrix
algorithms, other controls, correlated sampling and a coupled multilayer
recurrence need their own accounting; this is not a bound on those objects.

The archived rank35 realnet_measurements.json has postactivation gradient
energy trace922.5095 versus output variance trace68.3720, with160 Jacobian
samples. These are unconditional POSTactivation summaries, not this
PREactivation/Mehler/g'^2-weighted Vp. Neither they nor the old affine
capture/Dirichlet receipts establish the needed variance or rule it out.

## Endpoint

The new result replaces an unspecified conditional-density fit by an exact
direct expectation correction and an explicit measurable residual. It also
fixes reference-mean ownership through (3). It does NOT supply an affordable
high-capture gradient control, an actual variance receipt, a full sampler,
a bias-qualified recurrence, a complete bill or a final-score prediction.
Those are the precise remaining prerequisites; no larger fit, sample,
direction or probe-count sweep follows from this identity alone.

Confirmed bank #329926 remains4.982231460183304e-9; retained independent14
response4.83424098592912e-9. The ONE response exception stays complete and
preserved. Goal strictly1e-9, banks4e-9/3e-9/2e-9 remain unmet. No numerical
workers/MPS, target acquisition, refit, upload, forum post, frozen edit,
commit, promotion or deletion occurred in this preflight.
