# Mathematical derivation ledger

This is a living consolidation of the completed lanes.  It records the
substantive mathematics rather than chat transcripts.

## 1. Exact target and composition laws

For baseline residual `e0` and candidate corrections `z_i`, define

\[
G_{ij}=\langle z_i,z_j\rangle,\qquad b_i=\langle e_0,z_i\rangle.
\]

The optimally calibrated stack has

\[
\alpha_*=G^\dagger b,\qquad M_{stack}=M_0-b^T G^\dagger b.
\]

For an existing block `U` and new block `S`, the only valid incremental gain
is the Schur-complement quantity

\[
K=D-H^TA^\dagger H,\quad d=c-H^TA^\dagger a,\quad
\delta_{S|U}=d^TK^\dagger d.
\]

Thus source percentages cannot be added.  Joint refit is required, but it
cannot add information outside the supplied span.

The perfect-state intervention ladder (units `1e-9`) is

\[
31.972\to14.656\to10.286\to9.2547,
\]

with sequential work terms `17.316` (variance), `4.370` (k3), `1.0313`
(k4), and about `2.94` protected remainder to the exact-q endpoint.  If the
interventions are nested optimal projections, a source spanning only
variance+k3+k4 needs correlation above `0.98346`; a source spanning the full
exact-q signal needs about `0.9254`.  Exact variance+k3 still misses the goal
by `2.86e-10`, so only `27.7%` of the residualized k4 energy is needed once
those first two channels are exact.

## 2. Exact value consumer

For

\[
\epsilon_{l+1}=W_{l+1}^T(D_l\odot\epsilon_l+I_l)W_{l+1},
\]

and terminal `h`, the reverse planes

\[
H_L=\operatorname{Diag}(h),\quad
B_l=W_{l+1}H_{l+1}W_{l+1}^T,\quad H_l=D_l\odot B_l
\]

give

\[
h^T\operatorname{diag}\epsilon_L
=\langle H_0,\epsilon_0\rangle+\sum_l\langle B_l,I_l\rangle.
\]

Using `G12=G21^T`, `G13=G31^T`, the pair action is

\[
\langle B\odot G21,dK21\rangle+
\tfrac14\langle B\odot G22,dK22\rangle+
\tfrac13\langle B\odot G31,dK31\rangle.
\]

Moreover `G31` is algebraically determined by `G21,G22`:

\[
G31_{ij}=-(m_i/v_i)G21_{ij}-(C_{ij}/v_i)G22_{ij}.
\]

This exact `n^2` reverse consumer is mandatory infrastructure.  It does not
manufacture the missing cumulant source.

## 3. Generic source-identifiability obstruction

For `T in Sym^3(R^n)`, one repeated-plane observation is

\[
(P_RT)_{ab}=T(r_a,r_a,r_b).
\]

In the native frame its kernel is exactly the all-distinct sector, dimension
`C(n,3)`.  At `n=1024`,

\[
\dim Sym^3=179,481,600,\quad \operatorname{rank}P=n^2=1,048,576,
\]

leaving `178,433,024` hidden dimensions.  Universal cubic recovery requires
at least 172 independent truth-valued rotated contexts; under isotropic
geometry even `0.9253` correlation needs about 147 plane-equivalent contexts.
Known suffix masks without measured truth responses add no information.

Complete pair marginals are also insufficient.  A bounded density
perturbation proportional to `g(x_i)g(x_j)g(x_k)` preserves every one- and
two-coordinate marginal but changes an all-distinct triple.  Hence raw
pair-Ursell state cannot be a universal source carrier.

## 4. Closed generic compression routes

- **Raw/low-mode value compression.**  In the dense-He decorrelated regime,
  the normalized product spectrum is Fuss-Catalan.  Fifteen latent modes have
  an oracle-adaptive correlation upper bound about `0.647` over ten late
  transitions and `0.811` over sixteen; fifteen full-plane modes are bounded
  near `0.240`.  A frozen cross-net basis is generically about `0.121`.
  Reopen only with a cross-net invariant quotient of stable rank `<=17.34`.
- **Polynomial chaos/cubature.**  Degree-4 Gaussian cubature at dimension
  1024 needs at least `C(1026,2)=525,825` nodes; two dense transitions exhaust
  the card.  MUB/spherical designs have the same order and cannot supply the
  required broad shell with a small number of bases.
- **Small-k Rao-Blackwell modes.**  Exact conditioning is useful, but a broad
  quadratic or cubic action requires roughly 953 or 976 retained modes for
  `0.93` correlation.  Gaussianizing the transverse bath erases the mixed
  and transverse k3 information.
- **Finite Gaussian mixtures.**  Shared-covariance mixtures are constrained
  positive CP3; full mixtures still have a cubic zero-subspace invariant.
  A finite Gaussian mixture is not distribution-exact after ReLU, which
  creates face atoms.  Small K is viable only under a separately proved
  low-regime, action-aligned condition.
- **Stein/score closure.**  Edgeworth, Stein score, and pathwise tangent are
  the same first-order functional.  Deterministic integration encounters
  gate-surface Gaussian polyhedra; ordinary ReLU Hessians vanish almost
  everywhere and miss those terms.
- **Generic tensor-network contraction.**  Three replicas over three or more
  coupled layers contain a width-3 grid minor.  Exact contraction requires an
  `n^3` separator and `Theta(L n^4)` dense arithmetic.  Scalarizing the
  endpoint does not remove the internal separator.

## 5. CP3/star, fish, rings, and double counting

The `H2-H1-H1` three-site tree is exactly the CP3/star birth.  The two-cubic
two-line term inside the Gaussian pair kernel is already the `m=3` term of
`G11`; the existing tangent recurrence already composes it through depth.
Therefore an unqualified cubic-fish/Dyson proposal is duplicate physics.

The first clearly new simple even diagram is the four-distinct-site H2 cycle.
For a symmetric zero-diagonal typed bond `B`, the exact rooted C4 residual is

\[
\begin{aligned}
S^{C4}_{ij}={}&B_{ij}(B^3)_{ij}
-B_{ij}^2[(B^2)_{ii}+(B^2)_{jj}]+B_{ij}^4\\
&+\tfrac12[(B^2_{ij})^2-\sum_kB_{ik}^2B_{kj}^2].
\end{aligned}
\]

It costs about three dense products per layer and removes root bounces and
doubled stars.  It is not the same object as the unrestricted walk resolvent
or the amputated cubic fish.  It sources fourth-order channels, not the
mandatory independent cubic innovation, and generic four-leg transport is
`O(n^4)`.  The infinite ring resolvent has no regime in which it is both
goal-scale and controlled: away from a pole it is too small; near a pole
crossed/non-ring skeletons become equally important.

The raw nine-plane pair-Ursell hierarchy is also closed: pair projection is
not invariant under dense mixing, all-distinct terms remain leading quenched
order, and crossed K22 transport is generically `O(n^4)`.  Only a
baseline-subtracted, amputated pair-2PI residual remains conditional.

## 6. Quadratic/Jordan chaos and the H4 square lift

For latent `g~N(0,I_r)`,

\[
Q_M(g)=\tfrac12(g^TMg-\operatorname{tr}M)
\]

has

\[
E[Q_MQ_N]=\tfrac12\operatorname{tr}(MN),\qquad
E[Q_MQ_NQ_P]=\operatorname{tr}(MNP).
\]

With `r(r+1)/2 >= n` (`r=45` at `n=1024`), this gives an `O(n^2)`
noncommutative trace/Jordan state and exact dense affine transport in
`O(n^3)`.  Generic ReLU birth frames nevertheless generate the full matrix
algebra; exact fixed-r closure is not guaranteed, and pair observables do not
identify the completion.

Pure quadratic reset fails the dense-isotropic goal gate.  At zero bias its
retained centered energy is

\[
\pi/[2(\pi-1)]=0.73347,
\]

or correlation `0.85643`.  The first discarded pure fourth-chaos square is

\[
U_M=Q_M^2-\tfrac12\operatorname{tr}M^2-2Q_{M^2}
=\tfrac14 I_4(M\widetilde\otimes M).
\]

Quadratic plus this H4 square retains

\[
(\pi+1)/[2(\pi-1)]=0.96695
\]

of one-reset centered ReLU energy, correlation `0.98334`.  This is the
strongest new mathematical lead.  Its unresolved issue is rank growth:
after dense mixing a unit receives a sum of squares, not one square.  The
active question is whether one target-conditioned latent `r^4=O(n^2)`
reverse adjoint can collect these births without storing `n` forward H4
tensors.

## 7. Fail-fast theorem

For evaluated set `S` and genuine unseen bounds `l_i <= e_i <= u_i`, the
sharp fixed-corpus interval is

\[
[E_S+\sum_{i\notin S}l_i,\ E_S+\sum_{i\notin S}u_i]/N.
\]

Stop for certified failure when the lower endpoint is at least `1e-8`, and
for certified success when the upper endpoint is below `1e-8`.  The exact-q
score is not a rowwise lower bound merely because it has been achieved.
Two-network tests can detect sign reversal, implementation failure, and
frozen-transfer collapse, but cannot certify the 688-network campaign goal.

## 8. Current mechanism sequence

The minimum plausible stack is:

1. a broad late variance/self-energy state;
2. an independent signed cubic vertex;
3. a quartic/H4 square direction (or another higher-order remainder source);
4. exact reverse value contraction;
5. Gram-residualized joint refit.

Even-only physics cannot win by itself, variance-only cannot win, and a
consumer/refit without a new source cannot win.  The active lanes are testing
whether an H2/H3/H4 noncommutative chaos hierarchy plus reverse H4 action can
provide all three source classes without an `n^3` forward state.

## 9. Orchestration checkpoint 1 — 2026-08-30 16:52 PDT

All fifteen lanes were audited.  Thirteen were advancing; the two converged
lanes (`PORTFOLIO_GRAM` and `LIE_JORDAN`) were immediately assigned deeper
dependent questions.  No lane remained idle or stalled.

### Coupled deploy-frame tangent

The independent cubic-provenance and Lie--Jordan lanes now agree exactly.
Writing the Gaussian state as `theta=(m,v,C)`, the higher-plane producer as
`k=P(theta)`, and its covariance consumer as `C(theta,k)`, the full derivative
is the chain rule

\[
D[C(\theta,P(\theta))]
=C_\theta+C_kP_\theta.
\]

The compact mask terms

\[
\delta G_{11}\odot\epsilon_1
+\delta G_{21/22/31}\odot k_1
\]

are `C_theta`, the consumer half.  The three `delta A` insertions into the
cubic transport and `D S_2[delta theta]` generate `k_2=P_theta delta theta`,
which returns through the ordinary base masks as `C_k P_theta`.  A deployed
self-generated hierarchy needs both halves.  The smallest genuinely new
observable is the delayed feedback loop

\[
J_{\theta T,l+1}J_{T\theta,l},
\]

or, after eliminating the cubic state,

\[
J_{eff}(z)=J_{\theta\theta}
+J_{\theta T}(zI-J_{TT})^{-1}J_{T\theta}.
\]

This is the exact audit boundary: if the current engine is block-upper
triangular (`J_Ttheta=0`), deploy-frame co-adaptation is new; if it already
contains the loop with the exact coefficients, the proposal is a duplicate.

### Master three-source steering inequality

For the nonlinear frame source, signed cubic source, and complete physics
quartic master, define conditional recovered fractions `a_v,a_3,a_4`.  The
exact-state values give the nominal deficit simplex

\[
1.7316(1-a_v)+0.4370(1-a_3)+0.10313(1-a_4)\le0.07453
\]

in units of `1e-8`.  If the frame source begins with local correlation `q`,
the orders through two are absorbed by co-adaptation, its normalized third
order remainder is bounded by

\[
r_v=C_3(1-q^2)^{3/2}+\eta,
\]

and the governing inequality becomes

\[
\boxed{
1.7316r_v^2+0.4370(1-a_3)+0.10313(1-a_4)\le0.07453.
}
\]

The symbol `q` in this Taylor statement is specifically a **pre-Taylor source
alignment**, hereafter `q_s`; it is not automatically the measured endpoint
variance-channel correlation `q_v`.  With coefficient-value fidelity
`phi_beta`, the actual recovered variance fraction is

\[
a_v=\phi_\beta q_v^2,
\]

and the lower bound

\[
q_v^2\ge 1-[C_3(1-q_s^2)^{3/2}+\eta]^2
\]

is licensed only when the full co-adaptation theorem connects the source
alignment to the endpoint channel.

At `q_s=0.78`, `(1-q_s^2)^(3/2)=0.24506`.  Even with exact cubic and
quartic sources, unit coefficient fidelity, and the co-adaptation theorem,
success requires

\[
0.24506C_3+\eta\le0.20746.
\]

Thus `eta=0` requires `C3<=0.8466`.  A unit-constant third-order remainder
has a best possible raw estimate about `1.02946e-8`, so it still fails.  With
`C3=1, eta=0`, the **pre-Taylor source alignment** must rise from `0.78` to
about `0.806`.  This `.806` statement is invalid if the recorded `.78` is
already `q_v`.  In that endpoint-correlation interpretation, `a_v=.6084`,
whereas the deficit simplex requires `a_v>=.95696` when the other two channels
are exact.  The resummation must then recover

\[
\frac{.95696-.6084}{1-.6084}=.890
\]

of the remaining variance-channel energy (residual correlation at least
`.943`), not merely move `.78` to `.806`.  Resolving which receipt convention
applies is now the highest-value attribution question.

### NCQC observability and the genuinely new quartic component

A static `r>=55` NCQC fit remains non-identifiable.  Full core uniqueness is
unnecessary, however: a trajectory is sufficient exactly when

\[
\ker O_L\subseteq\ker D\Xi
\]

modulo conjugacy, where `Xi` is the signed H3, q4(211), K31/K22, and derived
2PI action.  Dense rotated repeated-plane probes can kill the cubic-visible
fiber; reset curvature then probes cubic-dark directions through q4(211).
Commuting seeds are an exact no-go because spectral resets cannot enlarge the
generated algebra.  A lawful seed must have scalar commutant/Burnside-full
algebra.

The notation `q4(211)` is also typed explicitly from this checkpoint onward:
`q4_z(211)` denotes multiplicities of latent Gaussian/Hermite coordinates,
`q4_ext(211)` denotes equality of external neuron labels in
`kappa4(Q_i,Q_i,Q_j,Q_k)`, and `q4_graph` denotes internal traffic-vertex
collisions.  These projections commute but are not interchangeable.  For
quadratic cores the decisive **external-label** mixed quartic identity is

\[
\kappa_4(Q_i,Q_i,Q_j,Q_k)
=48\operatorname{tr}(M_i^2M_jM_k)
+8\operatorname{tr}([M_i,M_j][M_i,M_k]).
\]

The first term is the commuting square.  The second is the new commutator
cross-Gram: q22 observes its norms (`j=k`) but not its cross orientations.
Once a full core is trajectory-identified this is a held-out consequence,
not a separately credited source.  A trajectory-contracted q4(211) action can
be evaluated in `O(n^3)` from the Jordan/commutator superoperator; materializing
the entire three-index plane remains `O(n^4)`.

### Analytic rest and tail gates

The rest lane produced target-free quartic bounds.  For a scalar `Z`, quartic
majorants/minorants imply

\[
E[Z_+]\in
\left[
\tfrac12m_1+\frac{m_2^{3/2}}{2\sqrt{m_4}},
\tfrac12m_1+\tfrac12\sqrt{m_2}
\right].
\]

For a pair, the transforms `S_t=tX+t^{-1}Y` and `D_t=tX-t^{-1}Y` give
explicit fourth-moment bounds on `E[X_+Y_+]`.  Propagating their interval
width `omega_4` with the endpoint adjoint yields a no-target hidden-information
bound `beta <= E[Omega_4^2]/4`, directly usable in

\[
M\le6.3151e-9+\beta+\eta_F^2+22.7173e-9(1-q^2).
\]

The H6 lane also sharpened to a cheap kill rule.  At an independent centered
boundary, the covariance-generated sixth-order term is

\[
-\frac1{4\pi}B^{\odot2}\odot(d\mathbf1^T+\mathbf1d^T).
\]

It is exactly zero for a correlation-only correction (`diag B=0`).  Any live
H6 direction must additionally be residualized against the cubic-fish span;
the two are both homogeneous sixth chaos and can be collinear.

### Current steering order

1. Prove or refute the exact delayed deploy-frame feedback block.
2. First type the `.78` receipt.  If it is `q_s`, bound/resum the third-order
   constant below `0.8466` or raise `q_s` above `0.806`; if it is already
   endpoint `q_v`, prove recovery of at least `89.0%` of its residual energy.
3. Establish trajectory-functional observability of the NCQC signed H3 and
   q4(211) consequences without demanding full latent uniqueness.
4. Contract only the endpoint-weighted q4(211) commutator Gram; never
   materialize its full tensor and never credit it separately from its core.
5. Use the analytic quartic-diameter bound to separate functional error from
   genuinely missing high-order information.
6. Close centered H6 immediately when the supplied plane has zero diagonal.

## 10. Orchestration checkpoint 2 — 2026-08-30

All fifteen checkpoint-1 questions produced a new theorem or a sharp gate.
Classification at the wave boundary was therefore `15 converged, 0 stalled`;
each lane was immediately assigned a dependent checkpoint-3 question, so all
fifteen agent slots remain active.  Convergence here means the bounded question
was answered, not that the raw-MSE program is solved.

### Fidelity semantics are now binding

There are three different quantities:

\[
q_s=\text{pre-Taylor source alignment},\qquad
q_v=\text{endpoint variance-channel correlation},\qquad
\phi_\beta=\text{coefficient-value fidelity}.
\]

The actual recovered variance fraction is

\[
a_v=\phi_\beta q_v^2,
\]

and the exact master remains

\[
1.7316(1-a_v)+.4370(1-a_3)+.10313(1-a_4)\le.07453.
\]

The `q_s=.78 -> .806` result is valid only under the proved third-order
co-adaptation bound, `C3=1`, `eta=0`, exact coefficient fidelity, and exact
other channels.  If `.78` is already endpoint `q_v`, the required endpoint is
`.97824`; the new mechanism must explain `89.01%` of the remaining variance
value, or correlate at least `.9434` with that residual.

### Exact NCQC quartic split and source ledger

For `Y_ij=A_i circ A_j`, `X_i=svec(A_i)`,
`P=X^T(XX^T)^+X`, and `Z_ij=(I-P)Y_ij`, external-label quartic `211` splits
exactly into

\[
q_{\parallel}(i;j,k)=\frac14
[t_{ii}^TK^+t_{jk}+2t_{ij}^TK^+t_{ik}],
\]

\[
q_{\perp}(i;j,k)=16
[\langle Z_{ii},Z_{jk}\rangle+2\langle Z_{ij},Z_{ik}\rangle].
\]

`q_parallel` is the H3/fish-predictable column Gram and gets no new source
credit.  `q_perp` is nonzero exactly when `span{A_i}` is not Jordan-closed.
The signed H3, counterterm fish, and `q_parallel` are three consumers of one
NCQC source; `q_perp` is the only possible independent NCQC quartic source.
Its exact incremental task gain after the first three is the Schur term

\[
\Delta_{\perp|Z}=d_r^2/s_r.
\]

With exact variance and H3 it must be at least `2.86e-10`, or `27.732%` of
the oracle quartic marginal.

The three partition labels are now explicit everywhere:

- `lambda_z`: latent Hermite-coordinate multiplicity;
- `lambda_ext`: external neuron-label equality;
- `lambda_graph`: traffic-vertex collision.

Normal order, latent projection, graph Mobius inversion, external collision
subtraction, and old-span residualization are separate operations.  The
physics `2:1` ratio applies only to the mixed-versus-square origins in
`P4[(T+A)^2-T^2]`, not to arbitrary `211/22` partitions.

### Deployment theorem: one stream, with a precise obstruction

The product identity

\[
I_3(\operatorname{sym}(v\otimes M))I_1(w)
=I_4(\operatorname{sym}(M\otimes v\otimes w))
+I_2(2v\odot Mw+(v\cdot w)M)
\]

proves that signed H3, its H2 return, and sequential external-`211` branches
can share one propagated matrix word plus cheap singleton vectors.  One word
over fourteen transitions costs `28` GEMM-equivalents.  A value-visible
primitive/reset `q_perp` forces a second word when either its matrix leg is
independent or its singleton-pair rank is dense.  Under the remaining budget,
that second word may live for at most ten late transitions (`28+20=48`).

The exact one-stream value-tail gate is

\[
\tau_1\le .0328064
\]

for the `.98346` three-channel route (or `.1435423` for the looser full-oracle
`.92545` route).  A persistent ten-layer forward H4 bank costs `54>49` and is
closed; only a direct target-conditioned `q_perp` contraction costing at most
four GEMMs per late layer can survive.

### Signed feedback, not a raw fish, is required

For the deploy-frame/cubic block

\[
J=\begin{bmatrix}A&B\\C&D\end{bmatrix},\qquad
J_{eff}(z)=A+B(zI-D)^{-1}C,
\]

a raw PSD fish is positive feedback and cannot reduce a unit cubic remainder
below one.  Suppression requires the counterterm-correct difference

\[
\Delta\Sigma=\tfrac12[V(G\otimes G)V^T-T_D(G\otimes G)T_D^T],
\]

including its signed cross terms.  The robust multi-pole pass gate is

\[
\operatorname{dist}(1,W(F_L))>\nu_L/.8466,
\]

plus full block stability, covariance PSD, and polewise D11 cancellation.
In one scalar hidden mode, the exact stable suppression interval is

\[
(1-\delta)(a+.1812)<k<1-a\delta.
\]

This `C3<.8466` lane applies only if `.78` is `q_s`.  If `.78` is endpoint
`q_v`, the invariant requirement is instead the direct `.8901` residual
energy recovery.

### Rest and high-tail conclusions

The `q_perp` uncertainty is a residual Gram completion.  With
`G_Z[a,b]=<Z_a,Z_b>`, the no-label feasible set is

\[
\mathcal C=\{G\succeq0:\operatorname{diag}G=d,\ \mathcal A(G)=g_{coll}\},
\]

whose support function is a finite SDP.  Doob orthogonality gives a
root-sum-square multi-layer upper bound on hidden information.  Predictable
`q_parallel` and `E[G_Z|state]` errors belong to functional error `eta_F`;
only the centered residual Gram contributes to hidden `beta`.  No state-built
frame feature can reduce that hidden term in population.

Centered H6 vanishes for a correlation-only correction.  The exact total
same-transport even tail after H4 has amplitude ratio at most `pi-3` and energy
at most `(pi-3)^2=0.02005` of H4.  Its optimistic absolute capacity is only
about `2.07e-11`; it is closed as a standalone raw-reduction lane and retained
only as zero-GEMM completeness.  That agent has been reassigned to the more
important latent-H4 mismatch bound.

### Checkpoint-3 proof targets

The smallest surviving stack is now:

1. reachable, signed, counterterm-correct deploy-frame feedback;
2. one shared NCQC matrix stream for signed H3 and its tied H2/H4 readouts;
3. at most one late `q_perp` word, Schur-residualized from all shared Gram
   consumers;
4. a joint-support/Doob certificate for hidden quartic/rest information.

It wins only if the signed feedback passes the appropriate `.8466` or `.8901`
gate, the one-stream tail fits `.0328064`, `q_perp` supplies the conditional
quartic deficit without double credit, and the total stays within `49` GEMM
equivalents and the `3.80e-6` adversarial endpoint error allowance.

## Checkpoint 3 convergence and Checkpoint 4 steering

All fifteen bounded Checkpoint-3 questions produced a falsifiable theorem,
obstruction, or promotion gate; none stalled.  Each completed lane was
immediately restarted on a dependent Checkpoint-4 question.  No numerical
science run was performed.

### New hard closures

- Support width and Doob variance bounds are one-sided.  Without an analytic
  Bellman/value relation, the sharp robust guaranteed task gain of any
  label-free `q_perp` feature is exactly zero.  A positive certificate needs
  `sqrt(c_B^T G^+ c_B) >= eta_B + 1.691e-5`, which is the threshold for the
  missing `2.86e-10` quartic gain.
- The exact graded exterior seed cannot coexist with the earlier `r=55`,
  2,099-dimensional fiber argument: 1,024 odd matrix units require `r>=64`.
  At `r=64` the q4-changing orbit has dimension 522,784, so one to three
  lawful contexts have zero uniform smallest singular value for small endpoint
  rank.  The generic `r=55` core remains conditional on a separate witness.
- A scalar endpoint does not imply a one- or two-word carrier.  The exact
  finite-horizon word count is the rank of the backward block-Krylov orbit;
  generic reset adjoints expand its invariant hull by Burnside.  The tiny-word
  route is a structural exception that must prove common wedge/rank and
  leakage identities.
- Generic dense-frame uniqueness is not conditioning.  Even an orthogonal,
  dense frame can approach an algebraically blind frame with arbitrarily small
  restricted singular value.  Promotion requires a candidate-specific
  `sigma_proj` bound and the corresponding observation-to-task error bound.
- Correlated ReLU reset can leak discarded H4 kernel state back into the
  Jordan quotient with order-one algebraic matrix elements.  Under coherent
  accumulation, the adversarial endpoint budget requires roughly one-percent
  adjoint-weighted leakage per layer unless a fresh-cavity or exact
  intertwining theorem is proved.
- The second q4 word cannot be made cheaper by rolling or early cash-out.
  Under injective second-quantized transport a genuinely perpendicular H4 word
  remains perpendicular and persistent.  The optimal use of the remaining 21
  GEMMs is one ten-transition endpoint suffix; its best word is the leading
  residualized-value singular mode.

### New exact positive structure

The signed counterterm and positive `q4_perp` do not have an intrinsic sign
conflict.  The former uses the Jordan component parallel to the carried NCQC
span; the latter is the trace-orthogonal remainder.  A bipartite direct-sum
core proves the joint PSD cone is nonempty without changing the cubic vertex.
This is an existence theorem, not yet a deployability theorem: label-free
observability, H2 slack, rank, and stream cost remain binding.

Canonical and commutator q4 readouts also reduce to one provider.  With
`Z_ab=(I-P)(A_a o A_b)`, the single PSD Gram
`Gperp_ab,cd=<Z_ab,Z_cd>` produces both the symmetric H4 sum and the
commutator contrast.  These are two consumer columns of one source and must be
jointly Schur-residualized rather than credited twice.

The complete external K31 witness has a clean latent mismatch ledger.  After
old-span projection, with latent witness `z`, external witness `f`, and
`a=||z||^2,b=||f||^2,c=<z,f>`, the label-free mismatch is
`B_z=a-c^2/b`.  It suffices only with a separate structural bridge bound
`B_bridge` satisfying
`(sqrt(B_z)+sqrt(B_bridge))^2 < 0.7453e-9`.  Without that bridge, a positive
task-alignment guarantee is impossible.

### Current Checkpoint-4 program

The active program now attacks the remaining assumptions directly: lawful
`r=55` observability, a real external-q4 Bellman identity, analytic
correlated-ReLU leakage, the D14-plus-D10 backward-Krylov schedule, candidate
max-min frame conditioning, deployability of the joint signed/q4 cone, and a
complete normal-order/Mobius/q-split/Schur audit.  The graph carries one live
node per lane, and each lane has a dedicated durable markdown record beside
this ledger.

## Checkpoint 7: tensor-free K31 pivot and claim-level graph split

All fifteen lanes advanced; none stalled.  The prior generic recurrent
`q4_perp` carrier is now closed on cost: its eleven observable quotient
directions over ten active layers require about 110 marginal provider
matmul-equivalents, far beyond the remaining deployment budget.  A scalar
endpoint does not by itself lower that local provider rank.

The strongest surviving pivot contracts before materialization.  With the
actual reverse K31 mask, define `P=W Q31^T`,
`U_a=sum_i W_ia X_i`, and `V_a=sum_i P_ia X_i`.  For every centered law,

```
sum_a kappa4(U_a,U_a,U_a,V_a)
 = sum_a { E[U_a^3 V_a] - 3 E[U_a^2] E[U_a V_a] }.
```

This is also one quarter of the directional derivative of the diagonal
fourth-cumulant potential.  If the local action is lawfully available, its
endpoint contribution is an exact one-scalar Bellman sum; no H4 word is
transported.  This removes temporal word-orbit, Burnside, boundary-wedge, and
eleven-Kraus *storage* obstructions.  It does not prove that the local fourth
action is observable or cheap.

For a synchronized affine-quadratic Gaussian law
`X_i=h_i^T z + z^T A_i z - tr A_i`, the local action closes exactly as

```
48 tr(N^3 R) + 24 { h^T N R h + h^T N^2 r }.
```

The half-quadratic convention divides this to the corresponding `3/3/6`
formula.  This identity is invariant under the physical simultaneous latent
orthogonal gauge, but not under the larger factor gauge left unidentified by
H2/H3 moments.  Consequently it is deployable only with synchronized oriented
`(A_i,h_i)` coefficients or an exact scalar action interface.  Lower moments
alone do not identify `E[U^3V]`.

The raw gate remains unchanged.  With exact frame and signed H3, the scalar
K31 route must supply at least `2.86e-10`, or `27.732%` of the oracle quartic
value.  The total unexplained frame/H3/q4 energy must stay below
`0.7453e-9`; the adversarial endpoint-amplitude allowance at the ideal point
is `3.80e-6`.

### Current decisive gates

1. prove the full endpoint derivative factors through the local K31 action;
2. expose the exact post-reset fourth action, or certify an AQ proxy and its
   beyond-AQ/frame residual;
3. prove the actual reverse-mask factorization and acquisition cost fit the
   remaining 21 GEMM-equivalents;
4. subtract the assembled CP3/Q24/DTOT2 baseline at the same site and causal
   time, counting every mixed stream monomial once;
5. pass the Schur gain, shift, and endpoint-error inequalities above.

### Graph normalization

The earlier convention `one lane = one node` packed resolved identities,
open premises, and cost no-gos under one status.  Checkpoint 7 removes that
ambiguity.  Lanes remain workstream owners, while the graph now decomposes
each lane into independently falsifiable claim nodes.  In particular, scalar
Bellman transport, direct moment acquisition, the AQ decoder, the beyond-AQ
tail, endpoint factorization, local provider cost, old-span ownership, and raw
gain have separate statuses, criteria, evidence receipts, and dependency
edges.  No solved identity promotes an unresolved sibling.

## Checkpoint 8: same-site K31 closes; future completion becomes the frontier

The complete same-site scalar K31 action is not a new feature.  A literal
recurrence audit proves that D11/Q24 already owns it through the assembled
`pre_k31 -> k31_target` path, with CP4, covariance pairing, quartic star, and
H31 occurring exactly once.  The reverse identity

```
<H, I31(D31)> = <Q31, D31>
```

therefore makes the total affine-quadratic/tensor-free K31 action a duplicate
consumer implementation.  Its residualized new gain is exactly zero.  The
only same-site candidate is

```
rho31 = <Q31, K31_true - K31_assembled>.
```

Current deployed Site A/B objects expose `K31_assembled`, not the true-law
fourth-action JVP.  A contract audit confirms that the estimator receives only
weights, shape metadata, budget, and reproducibility seeds; the repository's
`ORACLE_FOURTH` artifacts are offline truth interventions.  Thus a native
true-action callback is absent.  One projected scalar per layer is
information-minimal for a frozen endpoint, but generic acquisition can restore
the original four-leg cost and coadaptation rotates the needed action ray.

This correction is represented by separate nodes for the solved duplicate,
the open true-defect interface, residual provider cost, post-AQ H3 carrier,
and strict-depth completion.  The cheap affine-quadratic cost identities remain
valid implementation theorems but carry no raw source credit.

### First surviving mechanisms

Two nonduplicate candidates remain mathematically concrete:

1. A post-AQ cubic-chaos defect `I3(Z)` has an exact first-order external K31
   action through typed H3-H1-H1-H1 and H3-H1-H2-H2 contractions.  A shared
   CP dictionary makes these contractions tensor-free, but reset closure,
   signed orientation, and endpoint-weighted tail rank are binding.
2. A future-visible `K22` residual already exists transiently inside
   `_reset_stars`.  After removing CP4 square factors and the pairing plane,
   its typed `22` complement has dimension at least 521,727 and is provably
   nonzero by a Price-theorem derivative witness.  Its current K31 action is
   zero, but after one dense reset its scalar context action is
   `3 (u^2)^T Z22 (u*v)`.  It can be contracted at birth and discarded, so the
   open question is value/cost rather than tensor storage.

Under the current base-48/cap-49 stack, any new per-layer acquisition costing
one GEMM over ten layers is closed as an additive patch.  The surviving
economic comparison must therefore allow replacement of D10 or prove a
single precontracted residual action.  Every subsequent capacity calculation
uses the residual oracle denominator itself; the old `27.732%` shorthand is
valid only when the residual retains the full `1.0313e-9` quartic capacity.

## Checkpoint 15: independent q211 source and replacement-only frontier

The off-diagonal `211` sector is now a rigorously independent reset source.
A seven-leaf Vandermonde-star curve fixes every degree-at-most-three moment,
every marginal `K4`, every `K31`, and every pairwise `K22`, while changing
`kappa4(A,A,B,C)` with derivative 32.  A literal D11 audit shows that CP4
transports inherited rank-one factors and births only diagonal atoms; Star4
and the pairing summaries are also fixed on this curve.  Hence

```
ker D O_D11,complete  is not contained in  ker D q211,offdiag.
```

This proves source novelty, not endpoint value.  Frozen q4 capacity is only
`1.0313e-9`, so q211 is a portfolio closer rather than a standalone route to
raw MSE below `1e-8`.

Two acquisition mechanisms survive:

1. If the ordered products `P_ia=M_i N_a` are literally resident at the
   synchronized reverse seam, the complete sewn H211 action is the reduction

   ```
   sum_iab Theta_iab {32 <P_ia,P_ib> + 16 <P_ia^T,P_ib>}.
   ```

   This includes both the old-parallel and commutator pieces.  Building `P`
   afresh costs `1.5867 k` equivalents over D10 and is dead even at `k=1`;
   reuse is therefore a literal buffer identity, not a shared-shape argument.
   With resident products, dense reductions fit one spare equivalent only to
   about `k<=5`; diagonal/factored costates have separate, less restrictive
   ceilings.  Generic ReLU reverse pair kernels are full rank, so every cheap
   case still needs an endpoint-weighted rank receipt.

2. The missing `AABC` tensor has an exact CP birth.  Rank two is exact only in
   a restricted three-variable pairing quotient; one global D11 pairing
   scalar cannot cancel its local K22 leakage across the full width.  Four
   signed CP rays plus Star4 give an exact coordinate-frame repair at the same
   site or one transition, but generic dense transport rotates the compensator
   out of the coordinate-Star image.  The six-ray no-leak polarization is the
   unconditional persistent CP block.  Moving columns are cheap to push but
   must be acquired from the forward reset law, retained as a group, and
   scored against the incumbent CP4 columns they displace.  Ten six-ray births
   consume about `0.9671` equivalents before source/hooks, so the additive
   path is effectively cap-tight; dense rotated modes need the six-ray form or
   a proved compensator-naturality identity.

The graded/Clifford exception is now sharply scoped.  It can make a single
auxiliary commutator word analytic and product-free, but cannot replace the
full D14 state: at `r<=55`, Clifford representation bounds leave at most about
eleven independent covariance directions, versus width 1024.  Its only live
role is a target-conditioned q4 closer with separate support, old-module,
temporal-return, and endpoint-value certificates.

The strongest broad replacement hypothesis is now full affine-quadratic AQ
at about `r=49`, coupled to the already sourced Gaussian/Mehler high-chaos
rest.  The linear-plus-quadratic dimension and one-sided transport leave much
more arithmetic headroom than pure-H2 r50.  At centered affine base, parity
proves the true-minus-Gaussian rest has zero first variation in every
quadratic direction.  Binding gaps are a lawful zero-K3 initialization or
post-first-reset section, proof that one-sided AQ replaces D14's actual value,
the non-Gaussian rest bound away from the affine base, and the accumulated
residual cross term.  Without a Bellman cross receipt, the allowable rest
energy contracts from roughly `3.24e-2` to about `1.4e-3`.

Scalar Gaussian-deformation jets exactly acquire every local one-residual
q3/q4 equality sector, including H3 through H6.  They do not make a generic
future query cheap: the old-projected singleton cubic costate has Waring rank
`Omega(n)` under dense mixing.  Promotion therefore requires a bounded source
jet dictionary, a resident aggregate operator, or a certified future-dark
remainder.

No raw-MSE certificate exists yet.  The governing requirements remain:

```
1.7316(1-alpha_v) + 0.4370(1-alpha_3)
  + 0.10313(1-alpha_4) + E_cross <= 0.07453,
```

deploy-source retention at least `0.95696`, signed-H3 correlation above about
`0.9736` in the tight stack, and a residualized end-to-end partial correlation
above `0.8124` for a standalone correction.

### Graph presentation correction

Atomic claim splitting is preserved, but the renderer now honors an explicit
causal display parent before its legacy identifier-prefix heuristic.  The
source, obstruction, acquisition, closure, cost, conditioning, value, and raw
gates therefore render as derivation depth rather than one horizontal shelf.
The V18 graph has 15 distinct scientific workstreams; graph, layout, and
reconciliation validators pass without browser or server inspection.
