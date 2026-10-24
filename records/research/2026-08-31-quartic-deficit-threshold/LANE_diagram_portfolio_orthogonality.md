# Lane: diagram portfolio orthogonality and minimum certified stack

Status: active mathematical synthesis.  This note records the checkpoint results from the
`diagram_portfolio_orthogonality` lane.  All statements below are derivations or proof gates;
no numerical experiment was run in this lane.

## Fixed campaign quantities

The exact-state staged raw risks used as the channel-value budget are

\[
3.1972\times 10^{-8}
\xrightarrow{\text{exact frame/variance}}
1.4656\times 10^{-8}
\xrightarrow{\text{exact }H_3}
1.0286\times 10^{-8}
\xrightarrow{\text{exact }q4}
0.92547\times 10^{-8}.
\]

Consequently

\[
V_v=1.7316\times10^{-8},\qquad
V_3=0.4370\times10^{-8},\qquad
V_4=0.10313\times10^{-8},
\]

\[
V_\Sigma=2.27173\times10^{-8},\qquad
D=3.1972\times10^{-8}-10^{-8}=2.1972\times10^{-8},
\]

and the entire nominal slack of an exact three-channel stack is only

\[
m_0=V_\Sigma-D=7.453\times10^{-10}.
\]

Frame plus signed cubic is not sufficient: even with both exact, raw is
\(1.0286\times10^{-8}\).  Some nonredundant q4 value is mathematically necessary.

## Checkpoint 1: Wick typing, overlap, and the exact Gram certificate

Let \(I_p(f)\) denote a multiple Wiener integral.  The product identity

\[
I_3(f)I_3(g)=I_6(f\widetilde\otimes g)
+9I_4(f\widetilde\otimes_1g)
+18I_2(f\widetilde\otimes_2g)
+6\langle f,g\rangle
\]

shows that typed \(k_3^2\) Edgeworth, q4 one-contraction diagrams, q2 fish/self-energy,
and vacuum normalization are contraction sectors of one cubic-pair object.  They must not
be fitted as raw overlapping products.

In whitened latent coordinates, Hermite multi-index equality partitions give orthogonal
reference-Gaussian sectors.  Nominal examples are tree cubic \((3)\), all-distinct cubic
\((1,1,1)\), q4 square \((2,2)\), and latent q4 distinct \((1,1,1,1)\).  Dense transport and
the random endpoint adjoint need not preserve these sectors, so final de-duplication is in
the endpoint-weighted Gram, not by source labels.

For baseline-residualized source responses \(Z\), target residual \(R\),

\[
G=\mathbb E[ZZ^T],\qquad c=\mathbb E[ZR],
\]

and the exact jointly refitted improvement is

\[
\boxed{\Delta=c^TG^+c.}
\]

For a new scalar source \(j\) after an existing source block \(S\), its exact Schur
innovation and gain are

\[
s_j=G_{jj}-G_{jS}G_S^+G_{Sj},
\]

\[
h_j=c_j-G_{jS}G_S^+c_S,
\]

\[
\boxed{\Delta_j=h_j^2/s_j.}
\]

If \(s_j=0\), the source is a duplicate.  Raw arm gains are never additive unless the
responses have first been endpoint-orthogonalized.

### Three-scalar explicit Gram inequality

For ordered sources \((Z_2,Z_3,Z_4)\), write

\[
G=\begin{pmatrix}a&d&e\\d&b&f\\e&f&c\end{pmatrix},\qquad
c_R=\begin{pmatrix}u\\v\\w\end{pmatrix}.
\]

When \(G\) is nonsingular,

\[
\det G=abc+2def-af^2-be^2-cd^2,
\]

and the crossing condition is

\[
\frac{
u^2(bc-f^2)+v^2(ac-e^2)+w^2(ab-d^2)
+2uv(ef-cd)+2uw(df-be)+2vw(de-af)
}{abc+2def-af^2-be^2-cd^2}
\ge M_0-10^{-8}.
\]

The pseudoinverse form applies at deficient rank.

## Checkpoint 1b: the complete q4 master and missing mixed sector

Let \(T\) be the already-carried tree cubic and \(A\) the missing signed cubic.  The
complete incremental latent one-contraction master is

\[
\boxed{
U_{4,\mathrm{master}}^{\mathrm{lat},[r=1]}
=18I_4(T_3^{\mathrm{lat}}\widetilde\otimes_1A_3^{\mathrm{lat}})
+9I_4(A_3^{\mathrm{lat}}\widetilde\otimes_1A_3^{\mathrm{lat}}).
}
\]

This is \(P_4[(T+A)^2-T^2]\).  The origin coefficient ratio \(TA:AA=18:9=2:1\)
applies before independently normalizing or projecting features.  It is not a universal
ratio between Hermite partitions, external neuron-label cumulants, or graph topologies.

The tree--all-distinct cross produces a latent q4 \((2,1,1)\) sector.  In normalized
Hermites,

\[
T=\sum_i t_i h_3(\xi_i),\qquad
A=\sum_{i<j<k}a_{ijk}h_1(\xi_i)h_1(\xi_j)h_1(\xi_k),
\]

\[
P_4[2TA]=2\sqrt3\sum_{i<j<k}a_{ijk}
\left[t_i h_2(\xi_i)h_1(\xi_j)h_1(\xi_k)+\mathrm{cyclic}\right],
\]

\[
\|P_4[2TA]\|^2
=12\sum_{i<j<k}a_{ijk}^2(t_i^2+t_j^2+t_k^2).
\]

It is not killed by Gaussian symmetry when the two cubic supports meet.  ReLU also has a
nonzero fourth-chaos coefficient:

\[
\mathbb E[\operatorname{ReLU}(Z)h_4(Z)]=-1/\sqrt{48\pi}.
\]

### Mandatory q4 typing

The following are different objects:

1. latent Hermite partition \(\Pi^{\mathrm{lat}}_{(2,1,1)}U_4\);
2. external label cumulant \(\kappa_4(Q_i,Q_i,Q_j,Q_k)\);
3. the one-contraction cubic-pair graph \(\mathfrak G_{r=1}^{(3,3)}\).

With latent-to-external bridge \(\mathcal B\) and endpoint map \(L_{\rm end}\), the one
source response is

\[
\boxed{Z_4=L_{\rm end}\mathcal B
U_{4,\mathrm{master}}^{\mathrm{lat},[r=1]}.}
\]

Dense transport mixes equality patterns:

\[
K_{abcd}^{\mathrm{ext}}=
\sum_{i,j,k,l}W_{ia}W_{jb}W_{kc}W_{ld}K_{ijkl}^{\mathrm{lat}}.
\]

Latent, external, and graph views are annotations/projections of this source, not three
additive arms.  An external cumulant becomes new only after projecting away the master's
endpoint image.

## Checkpoint 2: minimum three-source accuracy region

Order the exact endpoint innovations as frame, signed \(H_3\) conditional on frame, and q4
conditional on both.  Let \(a_i=\rho_i^2\) be recovered conditional value fractions.  The
exact three-channel feasible region is

\[
\boxed{V_va_v+V_3a_3+V_4a_4\ge D.}
\]

Equivalently, in \(10^{-8}\) units,

\[
\boxed{
1.7316(1-a_v)+0.4370(1-a_3)+0.10313(1-a_4)\le0.07453.
}
\]

Necessary individual floors when the other two sources are exact are

\[
\rho_v\ge0.97824,\qquad
\rho_3\ge0.91074,\qquad
\rho_4\ge0.52661.
\]

Thus q4 must recover at least \(27.73\%\) of its exact conditional value when frame and
signed cubic are exact.  Equal fidelity requires \(a\ge0.96720\), or
\(\rho\ge0.98347\).

For normalized conditional source error \(\varepsilon_i\), optimal scalar calibration gives
\(a_i\ge1-\varepsilon_i^2\), so a sufficient source-error ellipsoid is

\[
V_v\varepsilon_v^2+V_3\varepsilon_3^2+V_4\varepsilon_4^2\le m_0.
\]

## Checkpoint 2b: robust coefficients and source conditioning

For train \(G,c\), frozen \(\beta=G^{-1}c\), and held shifts
\(G'=G+E,c'=c+e\), the held gain is exactly

\[
\Delta'_{\rm frozen}=\Delta+2\beta^Te-\beta^TE\beta,
\qquad \Delta=c^TG^{-1}c.
\]

With

\[
\delta_G=\|G^{-1/2}EG^{-1/2}\|_{\rm op}<1,
\qquad
\epsilon_c=\|G^{-1/2}e\|,
\]

\[
\boxed{\Delta'_{\rm frozen}\ge
\Delta-2\sqrt\Delta\,\epsilon_c-\delta_G\Delta.}
\]

Including baseline-risk shift \(\epsilon_M\), a sufficient preregistered margin is

\[
\Delta-(M_0-10^{-8})>
\epsilon_M+2\sqrt\Delta\,\epsilon_c+\delta_G\Delta.
\]

The minimax-shrunk coefficient and certified gain are

\[
\beta_{\rm rob}=
\frac{(\sqrt\Delta-\epsilon_c)_+}{(1+\delta_G)\sqrt\Delta}G^{-1}c,
\]

\[
\boxed{\Delta_{\rm rob}=
\frac{(\sqrt\Delta-\epsilon_c)_+^2}{1+\delta_G}.}
\]

Define

\[
D_{\rm rob}=\left[\epsilon_c+
\sqrt{(1+\delta_G)(D+\epsilon_M)}\right]^2,
\qquad
m_{\rm rob}=V_\Sigma-D_{\rm rob}.
\]

The robust accuracy region is the same deficit simplex with \(m_0\) replaced by
\(m_{\rm rob}\).  If \(m_{\rm rob}<0\), the three-source architecture cannot certify the
target even with exact sources.

For a source synthesis operator \(S\), approximate operator \(\widehat S=S+F\), and
\(\eta_S=\|FG^{-1/2}\|_{\rm op}\),

\[
\|G^{-1/2}(\widehat G-G)G^{-1/2}\|_{\rm op}
\le2\eta_S+\eta_S^2,
\]

\[
(1-\eta_S)^2I\preceq G^{-1/2}\widehat GG^{-1/2}
\preceq(1+\eta_S)^2I.
\]

These are target-blind source-conditioning gates.

## Checkpoint 2c: three distinct q conventions

The following quantities must never be conflated:

1. pre-Taylor source alignment \(q_s\);
2. post-resummation endpoint variance-channel correlation \(q_v\);
3. coefficient-value fidelity \(\phi_\beta\).

For exact variance correction \(V\) and deployed nonlinear feature span \(\mathcal F\),

\[
q_v^2=\|\Pi_{\mathcal F}V\|^2/\|V\|^2.
\]

If \(\widehat V\) uses frozen coefficients,

\[
\phi_\beta=1-
\frac{\|\widehat V-\Pi_{\mathcal F}V\|^2}
{\|\Pi_{\mathcal F}V\|^2}.
\]

The actual recovered frame value is

\[
\boxed{a_v=\phi_\beta q_v^2.}
\]

For scalar coefficient ratio \(\gamma=\widehat\beta/\beta_*\),

\[
\phi_\beta=2\gamma-\gamma^2=1-(\gamma-1)^2.
\]

The master channel requirement is

\[
\boxed{V_v(1-\phi_\beta q_v^2)+V_3(1-a_3)+V_4(1-a_4)\le m.}
\]

If a co-adaptation theorem applies to pre-Taylor \(q_s\), with projection distance
\(d_s=\sqrt{1-q_s^2}\), third-order remainder constant \(C_3\), and irreducible norm
\(\eta\), then only the following implication is valid:

\[
q_v^2\ge1-[C_3(1-q_s^2)^{3/2}+\eta]^2.
\]

At \(q_s=0.78\), \(d_s^3=0.24506\).  With exact coefficients and exact other channels,

\[
0.24506C_3+\eta\le0.20746
\]

is necessary.  At \(\eta=0\), \(C_3\le0.8466\).  For \(C_3=1\), the best three-source
raw is approximately \(1.02946\times10^{-8}\), so no allocation crosses.

The often-quoted \(q\simeq0.806\) threshold is valid only when \(q\) is pre-Taylor
projection alignment, \(C_3=1\), \(\eta=0\), coefficients and other sources are exact,
and no robust shift margin is charged.  If the Taylor theorem uses equal-norm chord
distance \(\sqrt{2(1-q_s)}\), the corresponding threshold is about \(0.825\).

If \(0.78\) is already endpoint \(q_v\), it cannot be cubed.  It must reach
\(q_v\ge0.97824\), equivalently an independent mechanism must recover

\[
\frac{0.956959-0.6084}{1-0.6084}=0.8901
\]

of the remaining endpoint variance value, a conditional correlation of about \(0.9434\).

## Checkpoint 3: q4-perpendicular gate and ranked minimum stack

Let \(X=(Z_{\rm frame}^{NL},Z_{H_3})\), and let \(P_X\) be the endpoint-value projector
onto their frozen response span.  For the typed complete q4 response \(Q_4\),

\[
Q_{4,\parallel}=P_XQ_4,\qquad
Q_{4,\perp}=(I-P_X)Q_4.
\]

Define

\[
s_\perp=\langle Q_{4,\perp},Q_{4,\perp}\rangle,
\qquad
h_\perp=\langle R,Q_{4,\perp}\rangle.
\]

Then

\[
\boxed{\Delta_{4,\perp}=h_\perp^2/s_\perp.}
\]

If \(s_\perp=0\), every q4 label under consideration is a duplicate of prior sources.
The exact three-source crossing inequality can be written as

\[
\boxed{L_v+L_3+(V_4-\Delta_{4,\perp})\le m.}
\]

Equivalently,

\[
\boxed{\Delta_{4,\perp}\ge V_4-m+L_v+L_3.}
\]

The ranked surviving deployed stack is:

1. one shared D14 nonlinear frame/signed-H3 stream;
2. a stable signed counterterm inside that stream;
3. one Schur-perpendicular q4 word over a late suffix.

The D14 stream costs 28 GEMMs.  A second q4 word over at most ten transitions costs 20
additional GEMMs, for 48 total.

### One-stream tail gate

For

\[
\tau_1=\|(I-P_1)S\|_V^2/\|S\|_V^2,
\]

the supplied thresholds are

\[
\tau_1\le0.0328064\quad(q\ge0.98346),
\]

for the equal-fidelity three-channel certificate, and

\[
\tau_1\le0.1435423\quad(q\ge0.92545)
\]

for only the broad campaign correlation screen.  If \(\tau_1\) is endpoint-value tail,
the best-case frame-specific limit is \(0.043041\).  If it is pre-Taylor source tail, it
may be converted through a proved \(C_3\tau_1^{3/2}+\eta\) theorem; without that theorem it
may not be cubed.

### Signed-feedback gate

A raw PSD fish cannot reduce \(C_3\) below one and is pruned at \(q_s=0.78\).  A negative
counterterm must prove

\[
\operatorname{dist}(1,W(F_L))>\nu_L/0.8466
\]

and exhibit the scalar feasibility window

\[
\boxed{(1-\delta)(a+0.1812)<k<1-a\delta.}
\]

The window is nonempty only if

\[
a<0.8188+0.1812\delta.
\]

This is a stability/remainder proof, not by itself a lower bound on signed-H3 endpoint
gain.

## Checkpoint 4: sole surviving stack and proof-level decision tree

### Support is not value

Let a source sequence \(u_l\) propagate through tangent dynamics

\[
x_{l+1}=A_lx_l+B_lu_l,
\qquad y=C_Lx_L.
\]

The endpoint response is

\[
y=\sum_l C_L\Phi_{L,l+1}B_lu_l.
\]

Knowledge that \(u_l\) has the right support, rank, or diagram type gives no positive
lower bound on \(y\): signs can cancel and the supported cone can intersect the endpoint
kernel.  Absent a Bellman/observability coupling theorem,

\[
\boxed{\inf \text{ certified endpoint gain}=0.}
\]

The stream proof must therefore establish a positive value angle or conditional Gram gain,
not merely source support.  An acceptable form is a cone-restricted bound

\[
\inf_{u\in\mathcal C,\ \|u\|=1}\|L_{\rm Bellman}u\|_V>0
\]

or the actual frozen conditional covariance/gain used in the master inequality.

**Earliest kill:** if no Bellman coupling or positive conditional endpoint gain is proved,
the stream remains support evidence only and cannot certify raw improvement.

### Exterior r64 context no-go

The converged no-go excludes an exterior rank-64 carrier built only from 1--3-context
features as the source realization for the required shared frame/H3 stream.  Any candidate
whose source map factors entirely through that class is killed before coefficient fitting or
endpoint grading.  Such features may remain diagnostics/support, but not the certified
carrier.

**Earliest kill:** factorization through the ruled-out r64 exterior 1--3-context class.
The survivor must use the full D14 shared stream or escape the no-go premise with a genuinely
different context/state class.

### Optimal late q4 suffix

Under the converged suffix result, among all placements of one second word on at most ten
transitions, the conditional q4-perpendicular value is maximized by the late ten-transition
suffix (the L5--L14 correction window):

\[
\boxed{
\Delta_{4,\perp}(\{5,\ldots,14\})
=\max_{|J|\le10}\Delta_{4,\perp}(J).
}
\]

This is a Bellman/Schur value statement, not merely a support statement.  At two GEMMs per
transition it costs 20 GEMMs.

**Earliest kill:** if even the oracle or certified upper value of this suffix is below
\(V_4-m+L_v+L_3\), no other ten-transition placement can rescue the 48-GEMM architecture.

### Direct-sum compatibility cone

The signed counterterm and q4-perpendicular word occupy separate state/control summands.
Write their feasible cones as

\[
\mathcal C_{\rm sign}\subset\mathcal H_{\rm shared},
\qquad
\mathcal C_{q4,\perp}\subset\mathcal H_{\rm late}.
\]

The combined construction cone is

\[
\boxed{\mathcal C_{\rm stack}
=\mathcal C_{\rm sign}\oplus\mathcal C_{q4,\perp}.}
\]

If both component cones are nonempty, their direct sum is nonempty.  Negative sign needed
for the counterterm and the sign/value of q4-perpendicular are therefore not intrinsically
incompatible.  Endpoint statistical overlap is removed by the q4 Schur projection.

**Earliest kill:** only an actual shared deployment constraint, empty component cone, or
budget collision can create incompatibility.  A generic “negative feedback versus positive
q4” objection is not a kill.

### Final ordered proof gates

1. **Provenance/type gate.**  Sources use deployed observables; q4 latent/external/graph
   labels are not double-counted.
2. **Source-class gate.**  Reject the exterior r64 1--3-context shortcut.
3. **Bellman gate.**  Prove positive endpoint coupling for the shared stream; support alone
   has lower bound zero.
4. **Signed-stability gate.**  Prove numerical-range separation, a nonempty scalar window,
   and the stated \(C_3,\eta,\phi\) bound.  PSD-only feedback is dead at \(q_s=0.78\).
5. **Shared-stream value gate.**  Certify the joint frame/H3 loss \(L_{FH}\).  If
   \(L_{FH}>m\), even exact q4 cannot rescue the stack.
6. **q4 novelty gate.**  Require \(s_\perp>0\).
7. **Suffix upper-bound gate.**  Require the optimal L5--L14 suffix to have
   \(\Delta_{4,\perp}\ge V_4-m+L_{FH}\).
8. **Direct-sum/resource gate.**  Combine the nonempty sign and q4 cones at 48 GEMMs.
9. **Frozen robust-Gram gate.**  Require worst-shift raw MSE below \(10^{-8}\), with no
   held-network retuning.

### Minimum proof package

Three theorem components suffice:

1. a shared-stream Bellman/tail theorem giving joint frame/H3 endpoint value and excluding
   the r64 shortcut;
2. a signed-counterterm theorem giving stability and the nonlinear remainder bound;
3. a late-suffix q4-perpendicular Schur theorem giving the required frozen gain.

The first two may be combined into one signed-stream theorem, leaving two proof documents,
but the three physical innovations (frame, signed H3, and q4-perpendicular) remain necessary.
No two-source proof can omit q4 under the exact-state channel ceilings.

## Pruning ledger

- **Prune:** raw PSD fish at \(q_s=0.78\); it has \(C_3\ge1\) and cannot cross.
- **Prune:** any claim that cubes an already endpoint-level \(q_v=0.78\).
- **Prune:** exterior rank-64 carriers restricted to 1--3 contexts.
- **Prune:** support/rank/diagram claims without Bellman endpoint coupling; certified lower
  gain is zero.
- **Prune as duplicate:** q4-parallel response, latent/external/graph views of the same q4
  master, and tree CP3 already in the baseline.
- **Prune within the 48-GEMM architecture:** q4 candidates requiring more than ten second-word
  transitions, or whose optimal ten-transition suffix upper bound misses the Schur requirement.
- **Do not prune for sign conflict:** signed feedback and q4-perpendicular lie in a feasible
  direct-sum cone when their individual gates pass.
- **Support only:** adjoint/Schur algebra, robust Gram bounds, one-stream tail bounds before
  Bellman coupling, finite-width channel attribution, and numerical-range stability by itself.
- **Reserve:** typed q6 and higher-chaos rest; activate only if the surviving three-channel
  stack fails after its proof-level upper bounds are exhausted.

## Current next question

The next decisive theorem is not another support construction.  It is a Bellman-coupled,
frozen-network lower bound for the actual shared D14 stream:

\[
\Delta_{FH}^{\rm frozen}>0
\]

with a typed statement of whether its reported \(q\) is pre-Taylor \(q_s\) or endpoint
\(q_v\), together with a certified \(C_3,\eta,\phi\).  Conditional on that result, the next
question is whether the optimal L5--L14 q4-perpendicular suffix has Schur gain

\[
\Delta_{4,\perp}\ge V_4-m+L_{FH}.
\]

Those two inequalities decide the surviving 48-GEMM stack.  Additional source families are
dominated until one of these proof gates fails.

## Checkpoint 5: effective-Kraus cost obstruction

The earlier statement that a scalar-target q4 group can be collapsed to one optimal Gram
direction is a statement about statistical response dimension, not deployable operator
rank.  A target-optimal scalar response may still require every Kraus contraction needed to
form that response.  Endpoint Gram rank one therefore does not imply effective Kraus rank
one.

For the generic \(\operatorname{Sym}_{45}\) q4-perpendicular operator, the converged
effective-Kraus count is eleven residual directions.  Carrying all eleven over the D10
suffix costs about 110 matmul-equivalents, outside the current cap.

Let the value-ordered deployable Kraus directions have nonnegative conditional contributions

\[
\lambda_1\ge\lambda_2\ge\cdots\ge\lambda_{11}\ge0,
\]

and define

\[
G_{11}=\sum_{j=1}^{11}\lambda_j,
\qquad
G_k=\sum_{j=1}^k\lambda_j,
\qquad
C_k=G_k/G_{11}.
\]

Equivalently, when the directions are not already value-orthogonal, \(G_k\) is defined as
the best conditional Schur gain among q4 operators of deployable effective Kraus rank at
most \(k\).  The required q4 gain remains

\[
G_{\rm req}=V_4-m+L_{FH}.
\]

### Value-versus-Kraus-rank theorem

A \(k\)-direction q4 suffix is sufficient if and only if

\[
\boxed{G_k\ge G_{\rm req}.}
\]

If the full eleven-direction gain \(G_{11}\) is known, this is equivalent to

\[
\boxed{C_k\ge
\frac{V_4-m+L_{FH}}{G_{11}}.}
\]

Since \(G_{11}\le V_4\) under the staged q4 channel budget, even the optimistic necessary
concentration is

\[
\boxed{C_k\ge
\frac{V_4-m+L_{FH}}{V_4}.}
\]

With exact frame/H3 and nominal margin,

\[
\boxed{C_k\ge0.27732.}
\]

Every unit of shared-stream loss raises the optimistic concentration requirement by
\(L_{FH}/V_4\).  For example, if the shared stream directly spends the equal-fidelity tail
\(\tau_1=0.0328064\) across all frame/H3 value, then

\[
L_{FH}=V_{FH}\tau_1\simeq0.071144\times10^{-8},
\]

so a one-direction q4 implementation would need about

\[
C_1\ge
\frac{0.02860+0.071144}{0.10313}
\simeq0.9672
\]

of the entire q4 value.  A nonlinear pre-Taylor interpretation can reduce this loss, but it
must be proved by the signed-resummation theorem rather than assumed.

### What rank eleven alone guarantees

For a flat eleven-direction spectrum,

\[
C_1=1/11\simeq0.0909,
\qquad
C_2=2/11\simeq0.1818,
\qquad
C_3=3/11\simeq0.2727.
\]

None reaches the nominal \(0.27732\) threshold.  Four directions are the first count whose
worst-case ordered concentration, \(4/11\), exceeds that threshold.  Hence generic rank
information alone cannot certify \(k=1\) or \(k=2\); a specifically top-heavy value spectrum
must be proved.

With the supplied cost estimate, a D10 Kraus direction costs about ten matmul-equivalents.
If the base D14 stream cost is 28 and no further reserved overhead is charged, a 49-GEMM cap
permits at most

\[
k_{\max}=\left\lfloor(49-28)/10\right\rfloor=2.
\]

If the deployment ledger reserves the remaining budget so only one direction is available,
the theorem below applies with \(k=1\).  In either convention, four generic directions do
not fit.

### Revised verdict for the one-word stack

The \(k=1\) stack fits the cost cap, but it is not certified by the generic
\(\operatorname{Sym}_{45}\) result.  It survives only under the explicit concentration gate

\[
\boxed{
\lambda_1\ge V_4-m+L_{FH}
}
\]

or, equivalently,

\[
\boxed{
C_1\ge(V_4-m+L_{FH})/G_{11}.
}
\]

At the most optimistic nominal point, the leading deployable Kraus direction must carry at
least \(27.73\%\) of all q4 conditional value, more than three times the flat-spectrum share.
If no top-one value-concentration theorem is available, the current one-word sole stack is
killed at proof level despite fitting the arithmetic cost cap.

### Updated q4 pruning order

1. Require q4 Schur novelty \(s_\perp>0\).
2. Determine the effective Kraus/Choi rank; do not infer it from scalar endpoint Gram rank.
3. Compute or bound the value-ordered concentration \(C_k\).
4. Set \(k\) from the actual remaining matmul budget.
5. Kill the architecture if
   \[
   G_k<V_4-m+L_{FH}.
   \]
6. Only after that gate use the L5--L14 suffix optimality and frozen network-shift proof.

### Revised current next question

The q4 question is now sharper than merely asking whether the full suffix has enough value:

\[
\text{Does the leading affordable effective-Kraus subspace satisfy }
G_k\ge V_4-m+L_{FH}?
\]

For the one-word plan this is the top-one concentration question.  A full eleven-direction
q4 result is support-only for deployment unless it also proves this low-Kraus value
concentration.

## Checkpoint 6: full 49-GEMM allocation frontier

Fixing D14 plus a single D10 q4 direction is not Pareto optimal under the new effective-Kraus
cost model.  Primary depth and q4 direction lifetimes must be optimized jointly.

Let \(d\in\{0,\ldots,14\}\) be the number of primary shared-stream transitions.  At two
matmul-equivalents per transition, its cost is \(2d\).  Let q4 Kraus direction \(j\) live on
an optimal suffix of length

\[
\ell_j\in\{0,\ldots,10\}.
\]

At the supplied effective-Kraus accounting, its cost is \(\ell_j\).  With optional fixed
overhead \(c_0\), the exact resource constraint is

\[
\boxed{2d+\sum_j\ell_j+c_0\le49.}
\]

The formulas below use \(c_0=0\); any reserved overhead is incorporated by replacing 49
with \(49-c_0\).

Let \(F(d)\) be the largest frozen Bellman-coupled frame/H3 gain attainable by a valid
depth-\(d\) primary stream, including the signed-counterterm gate, and define

\[
L(d)=V_{FH}-F(d),
\qquad V_{FH}=V_v+V_3=2.1686\times10^{-8}.
\]

Let

\[
\Gamma_Q(\ell_1,\ldots,\ell_k;d)
\]

be the conditional q4-perpendicular Schur gain after projecting against that depth-\(d\)
primary response.  This definition includes cross-direction Gram effects; when the Kraus
directions are value-orthogonal it is the sum of their individual suffix-value curves.

### Exact resource-allocation theorem

An allocation \((d,\ell_1,\ldots,\ell_k)\) certifies raw below \(10^{-8}\) if and only if it
passes all provenance/stability/shift gates and

\[
\boxed{
2d+\sum_j\ell_j\le49
}
\]

together with

\[
\boxed{
L(d)+V_4-\Gamma_Q(\ell_1,\ldots,\ell_k;d)\le m.
}
\]

Equivalently,

\[
\boxed{
F(d)+\Gamma_Q(\ell_1,\ldots,\ell_k;d)\ge D.
}
\]

Use \(D_{\rm rob}\) and \(m_{\rm rob}\) in the robust version.

For a q4 budget \(b\), define its optimal value frontier

\[
\Gamma_Q^*(b;d)=
\max_{\sum_j\ell_j\le b}\Gamma_Q(\ell_1,\ldots,\ell_k;d).
\]

The one-dimensional depth frontier is therefore

\[
\boxed{
\mathcal M(d)=m-L(d)-V_4+Gamma_Q^*(49-2d;d).
}
\]

Depth \(d\) is feasible exactly when \(\mathcal M(d)\ge0\).

### Dominance and marginal exchange rule

Allocation \(A\) dominates allocation \(B\) if

\[
\operatorname{cost}(A)\le\operatorname{cost}(B),
\qquad L_A\le L_B,
\qquad \Gamma_A\ge\Gamma_B,
\]

with at least one strict inequality.  A dominated allocation cannot be Pareto optimal.

Shortening the primary stream from \(d\) to \(d-r\) frees \(2r\) matmul-equivalents.  It is
beneficial if and only if

\[
\boxed{
\Gamma_Q^*(b+2r;d-r)-\Gamma_Q^*(b;d)
>
L(d-r)-L(d),
}
\]

where \(b=49-2d\).  In words: added q4 Schur value must exceed lost primary Bellman value.
This is the exact discrete Pareto-slope test.  Source support or nominal rank cannot replace
either side of this inequality.

### Concentration requirement at variable depth

Let \(G_{11}(d)\) be the full eleven-direction q4-perpendicular gain conditional on the
depth-\(d\) primary.  If the affordable allocation captures fraction \(C(d,\ell)\) of that
gain, it must satisfy

\[
\boxed{
C(d,\ell)\ge
\frac{V_4-m+L(d)}{G_{11}(d)}.
}
\]

Even optimistically using \(G_{11}(d)\le V_4\),

\[
\boxed{
C(d,\ell)\ge0.27732+\frac{L(d)}{V_4}
}
\]

at nominal margin.  Thus shortening the primary both buys Kraus budget and raises the q4
concentration target.

### Full-lifetime break-even table

Ignoring partial extra lifetimes for the moment, full D10 Kraus directions have cost ten.
With \(2d+10k\le49\):

| full q4 directions \(k\) | maximum primary depth | flat-spectrum q4 share |
|---:|---:|---:|
| 1 | 14 | \(1/11=0.0909\) |
| 2 | 14 | \(2/11=0.1818\) |
| 3 | 9 | \(3/11=0.2727\) |
| 4 | 4 | \(4/11=0.3636\) |

The cap leaves eleven matmul-equivalents at the D14+k1 row, so that row is immediately
dominated by adding a second full direction and one transition of a third.  The saturated
k2, k3, and k4 rows each leave one additional matmul-equivalent, which may be assigned to
the next direction.

Consequences:

1. **Rank one is dominated at D14.**  D14 costs 28 and leaves 21.  Two full D10 q4
   directions plus one transition of a third fit.  Because an unused q4 direction may always
   receive coefficient zero, D14 plus one direction is weakly dominated by this larger q4
   allocation unless an unmodeled fixed-state overhead forbids it.
2. **Flat rank two is insufficient.**  Its \(18.18\%\) value share is below the nominal
   \(27.73\%\) requirement, although a top-heavy two-direction spectrum can pass.
3. **D9 plus flat rank three still misses.**  Even with zero primary loss,
   \(3/11=27.27\%<27.73\%\).  It requires
   \[
   C_3\ge0.27732+L(9)/V_4.
   \]
   Thus shortening to D9 does not generically remove the Kraus obstruction; it needs both a
   slightly top-heavy q4 spectrum and an exceptionally small primary loss.
4. **D4 plus flat rank four has only a tiny primary-loss allowance.**  Its q4 surplus above
   the nominal requirement is
   \[
   (4/11-0.27732)V_4\simeq0.00890\times10^{-8}.
   \]
   Therefore it can pass only if
   \[
   \boxed{L(4)\lesssim8.90\times10^{-11},}
   \]
   about \(0.41\%\) of total frame/H3 value.  Removing ten primary transitions must be
   essentially value-free, otherwise D4+k4 is dominated or infeasible.

With D10 primary, 29 q4 matmul-equivalents remain, permitting two full directions plus nine
transitions of a third.  Under a flat spectrum and a linear-in-lifetime value approximation
this is only \(2.9/11\simeq0.2636\), still below \(0.27732\) before charging \(L(10)\).
The exact decision must use the proved suffix-value curves, not linear interpolation.

### Revised allocation verdict

The 49-GEMM cap removes the claim that the architecture is forced to rank-one q4: D14 already
permits two full effective-Kraus directions under the supplied ten-per-direction cost model.
However, shortening the primary stream does not generically solve the q4 obstruction:

- k2 at D14 still needs top-two concentration at least \(27.73\%+L(14)/V_4\);
- k3 requires D9 and is still slightly short under a flat spectrum even before primary loss;
- k4 requires D4 and tolerates only about \(0.41\%\) primary value loss.

The surviving Pareto candidates are therefore those for which the actual Bellman depth-loss
curve and q4 suffix/Kraus value curves satisfy the marginal exchange inequality.  No fixed
depth/rank choice is certified from cost and algebra alone.

### Current next question after allocation optimization

The decisive missing objects are now two target-value curves:

\[
L(d),\qquad d=0,\ldots,14,
\]

and

\[
\Gamma_Q^*(b;d),\qquad b=0,\ldots,49-2d.
\]

A proof-level bound at just the Pareto candidates D14+k2(+one step), D10+(2 full+9), D9+k3,
and D4+k4 is sufficient.  Any shortening is pruned immediately when its added q4 value is
no larger than its lost primary value.

## Checkpoint 7: universal tensor-free K31 Bellman action

The effective-Kraus obstruction applies only if the q4/K31 operator is materialized or
approximated as a forward carrier.  The newly derived scalar law contracts the fourth
cumulant against the reverse Bellman test before materializing any fourth-order tensor.  It
therefore bypasses rather than compresses the eleven-direction Kraus operator.

### Exact affine--quadratic scalar law

Let \(z\sim N(0,I)\) and

\[
X_i=h_i^Tz+(z^TA_i z-\operatorname{tr}A_i),
\]

with symmetric \(A_i\).  Define

\[
U_a=\sum_iW_{ia}X_i,
\qquad
V_a=\sum_iP_{ia}X_i,
\]

and the carried affine--quadratic factors

\[
N_a=\sum_iW_{ia}A_i,
\qquad
R_a=\sum_iP_{ia}A_i,
\]

\[
h_a=\sum_iW_{ia}h_i,
\qquad
r_a=\sum_iP_{ia}h_i.
\]

Then the exact centered scalar innovation is

\[
\boxed{
\kappa_4(U_a,U_a,U_a,V_a)
=48\operatorname{tr}(N_a^3R_a)
+24\left[h_a^TN_aR_ah_a+h_a^TN_a^2r_a\right].
}
\]

For arbitrary centered \(U,V\), the universal identity is

\[
\boxed{
\kappa_4(U,U,U,V)
=\mathbb E[U^3V]-3\mathbb E[U^2]\mathbb E[UV].
}
\]

The affine--quadratic law supplies these moments analytically from carried matrix/vector
factors.  Outside that model, acquisition of the general-law moments remains an unproved
cost/provenance premise.

### Revised minimum stack

The new minimum deployed state is

1. one shared D14 frame/signed-H3 tangent word with the signed counterterm;
2. a scalar K31 Bellman accumulator built from the per-layer
   \(\kappa_4(U,U,U,V)\) actions.

There is no recurrent q4 plane and no effective-Kraus rank requirement if the reverse test
\(V\) is the correct Bellman adjoint and the scalar moments are acquired exactly.

Let \(Z_{31}\) be the accumulated scalar response and let \(P_{FH}\) project onto the frozen
shared-stream response.  Only

\[
Z_{31,\perp}=(I-P_{FH})Z_{31}
\]

is new.  Define

\[
s_{31}=\|Z_{31,\perp}\|^2,
\qquad
h_{31}=\langle R,Z_{31,\perp}\rangle,
\]

\[
\boxed{\Delta_{31}=h_{31}^2/s_{31}.}
\]

The exact crossing theorem becomes

\[
\boxed{L_{FH}+V_4-\Delta_{31}\le m,}
\]

or

\[
\boxed{\Delta_{31}\ge V_4-m+L_{FH}.}
\]

At exact frame/H3 and nominal margin this is again

\[
\Delta_{31}\ge2.86\times10^{-10},
\]

which is \(27.73\%\) of the exact q4 value, corresponding to conditional scalar
correlation at least \(0.5266\) under exact coefficient calibration.

### Revised resource allocation

Let \(\mathcal J\) be the layers at which the scalar K31 action is acquired, and let
\(C_{31}(\mathcal J)\) be its incremental matmul-equivalent cost after reusing the carried
\(A,h\) factors and reverse \(H/Q31\) mask.  The exact resource condition is

\[
\boxed{2d+C_{31}(\mathcal J)\le49.}
\]

Let \(\Gamma_{31}(\mathcal J;d)\) be its conditional Schur gain.  The allocation theorem is

\[
\boxed{
L(d)+V_4-\Gamma_{31}(\mathcal J;d)\le m.
}
\]

If all required scalar actions for the relevant correction window can be acquired in at
most 21 matmul-equivalents, then full D14 plus all scalar actions fits:

\[
28+C_{31}(\mathcal J_{\rm full})\le49.
\]

Under monotone primary value and exact full-layer scalar acquisition, this allocation
dominates every shortened primary: shortening loses frame/H3 value without buying additional
K31 value.  If the scalar acquisition costs more than 21, define

\[
\Gamma_{31}^*(b;d)=
\max_{C_{31}(\mathcal J)\le b}\Gamma_{31}(\mathcal J;d).
\]

The Pareto frontier and exchange rule are

\[
\mathcal M_{31}(d)=m-L(d)-V_4+\Gamma_{31}^*(49-2d;d),
\]

\[
\boxed{
\Gamma_{31}^*(b+2r;d-r)-\Gamma_{31}^*(b;d)
>
L(d-r)-L(d)
}
\]

for a beneficial shortening by \(r\) transitions.

The closed form is algebraically \(O(n^3)\), or lower when factors/masks are reused, but its
exact billed matmul-equivalent cost is a terminal premise.  An asymptotic cost claim alone
does not certify the 49-GEMM cap.

### Exact residual terms consuming the margin

Let \(Q_4^*\) denote the exact conditional q4 correction and let \(\widehat Q_{31}\) be the
deployed scalar-action correction.  Define the total q4 Bellman residual

\[
e_{31}=Q_4^*-\widehat Q_{31}.
\]

Its components are:

1. **affine--quadratic discrepancy**
   \[
   e_{\rm AQ}=\sum_l\left[
   \kappa_4^{\rm true}(U_l,U_l,U_l,V_l)
   -\kappa_4^{\rm AQ}(U_l,U_l,U_l,V_l)
   \right];
   \]
2. **general-moment acquisition error** \(e_{\rm mom}\), if the universal moment identity
   rather than exact carried AQ factors is used;
3. **Bellman-mask error** \(e_{\rm B}\), from an approximate reverse \(P/R/r\) test rather
   than the exact endpoint adjoint;
4. **layer/suffix omission** \(e_{\rm layer}\), from unevaluated scalar innovations;
5. **coefficient/frozen-shift error** \(e_{\rm coef}\);
6. **parallel response**, removed exactly by \((I-P_{FH})\), which is not new q4 value.

Thus

\[
e_{31}=e_{\rm AQ}+e_{\rm mom}+e_{\rm B}+e_{\rm layer}+e_{\rm coef}.
\]

Without proved orthogonality, the exact q4 loss is

\[
\boxed{
E_{31}=\|e_{31}\|_V^2
=\sum_i\|e_i\|_V^2+2\sum_{i<j}\langle e_i,e_j\rangle_V.
}
\]

A safe componentwise bound is

\[
E_{31}\le\left(\sum_i\|e_i\|_V\right)^2.
\]

If the residuals are defined by nested value-orthogonal projections, their squared norms
add.  They must not otherwise be assigned independent pieces of the margin.

In loss form, the complete nominal certificate is

\[
\boxed{
L_{FH}+E_{31}\le7.453\times10^{-10}.
}

Equivalently, if the scalar observable has conditional correlation \(q_{31}\) and coefficient
fidelity \(\phi_{31}\),

\[
\boxed{
L_{FH}+V_4(1-\phi_{31}q_{31}^2)\le m.
}

At exact frame/H3, this requires only

\[
\phi_{31}q_{31}^2\ge0.27732.
\]

Every frame/H3 loss raises the required scalar-action fraction by \(L_{FH}/V_4\).  Network
shift is charged by replacing \(m\) with \(m_{\rm rob}\), not by counting it again as an
independent source loss.

### Revised pruning tree

1. **AQ identity gate:** prove the local variables satisfy the stated affine--quadratic
   Gaussian representation, or bound \(e_{\rm AQ}\).
2. **Observable/provenance gate:** prove \(N,R,h,r\), or the universal moments, are available
   from deployed target-blind state.
3. **Bellman-action gate:** prove \(V\) is the correct reverse endpoint test.  A local scalar
   cumulant without that coupling still has certified gain lower bound zero.
4. **Schur novelty gate:** require \(s_{31}>0\).
5. **Value gate:** require \(\Delta_{31}\ge V_4-m+L_{FH}\).
6. **Cost gate:** require full D14 plus the needed scalar actions to fit 49, or use the exact
   depth/action exchange inequality.
7. **Residual/shift gate:** require \(L_{FH}+E_{31}\le m_{\rm rob}\) with frozen coefficients.

### Revised current next question

The decisive proof is now whether the affine--quadratic scalar law is an observable exact
Bellman action on the deployed lineage at acceptable cost.  Specifically:

\[
C_{31}(\mathcal J_{\rm full})\le21
\]

and

\[
\Delta_{31}\ge V_4-m+L_{FH}
\]

must both hold.  If they do, the eleven-direction Kraus obstruction and the explicit q4
second word disappear.  If either the general-law moments or the reverse test require an
unavailable dense state, the claimed tensor-free route collapses back to the prior carrier
problem.

## Checkpoint 8: corrected scalar-K31 certificate, exact ownership, and final cost gate

This checkpoint supersedes the provisional Checkpoint-7 claim that a D14 primary word leaves
twenty-one GEMM-equivalents for the scalar action.  The comparison must include the already
required D10 reverse Bellman word:

\[
C_{\rm base}=2(14)+2(10)=48,
\]

so only one marginal GEMM-equivalent remains under the 49-GEMM cap.  More importantly, an
exact ownership identity removes the apparent same-site K31 source from the novelty ledger.

### Canonical normalization

Use the half-quadratic convention

\[
X=u^\top z+\frac12(z^\top A z-\operatorname{tr}A),\qquad
Y=v^\top z+\frac12(z^\top B z-\operatorname{tr}B),
\quad z\sim N(0,I).
\]

Then

\[
\boxed{
\kappa_{31}(X,Y)
=3\operatorname{tr}(A^3B)
+3u^\top(AB+BA)u
+6u^\top A^2v .
}
\]

For $Y=X$,

\[
\kappa_4(X)=3\lVert A^2\rVert_F^2+12\lVert Au\rVert^2.
\]

Equivalently, for

\[
\Phi(A,u)=3\operatorname{tr}(A^4)+12u^\top A^2u,
\]

one has

\[
\kappa_{31}(A,u;B,v)=\frac14D\Phi(A,u)[B,v],
\]

with

\[
\nabla_A\Phi=12A^3+24\operatorname{Sym}(Auu^\top),\qquad
\nabla_u\Phi=24A^2u.
\]

The earlier $48/24$ expression used $z^\top A z-\operatorname{tr}A$ without the factor
$1/2$; it is the same identity after rescaling.  The deployed reverse mask

\[
Q_{31,l}=\frac{H_l\odot G_{31,l}+(H_l\odot G_{13,l})^\top}{6}
\]

already contains the card's K31/K13 injection normalization, so no additional factorial may
be inserted when pairing the scalar action with $Q_{31,l}$.

### Ownership theorem: total same-site K31 is not a new source

Let \(\mathcal O\) be the endpoint-weighted span of actions already owned by D11/Q24.  The
same-site contraction obeys

\[
\boxed{
\langle\Lambda,K_{31}^{\rm same\mbox{-}site\ target}\rangle
=\langle\Lambda,K_{31}^{\rm assembled}\rangle
=\langle\Lambda,K_{31}^{\rm CP4}+3\alpha vC+H_{31}\rangle\in\mathcal O .
}
\]

The self-consistent same-site target is precisely this old ownership.  Therefore the only
candidate new scalar source is

\[
\boxed{
\rho_{31}:=\left\langle\Lambda,
K_{31}^{\rm true}-K_{31}^{\rm assembled}\right\rangle ,
}
\]

or a future-visible collision-free completion proved to equal a component of this residual.
The total \(\chi_{31}=\langle\Lambda,K_{31}^{\rm true}\rangle\) must never be booked as a new
source.  Since

\[
P_{\mathcal O}^\perp\chi_{31}=P_{\mathcal O}^\perp\rho_{31},
\]

fitting the total after the old columns are present is algebraically equivalent to fitting the
residual, whereas computing the affine--quadratic assembled term alone gives zero Schur gain.
In particular, exact synchronized affine--quadratic provenance is not itself a promotion gate:
if it proves $K_{31}^{\rm true}=K_{31}^{\rm assembled}$, it proves
\(\rho_{31}=0\) and kills the new-source claim.

This also fixes the typing ambiguity.  The object above is an endpoint-weighted residual of an
external K31 action.  It is not automatically the latent Hermite partition
\(\Pi_{(211)}U_M\), the external neuron-label cumulant
\(\kappa_4(Q_i,Q_i,Q_j,Q_k)\), or a graph-partition class.  A bridge must be proved before any
is a provider for \(\rho_{31}\).  The old $2{:}1$ coefficient belongs only to the $TA$
versus $AA$ Wick-origin terms and does not identify these typed objects.

### Exact residual Schur theorem

Let $S$ contain every retained old, frame, and signed-H3 source, let $z=\rho_{31}$, and
write

\[
G=\langle S,S\rangle_V,\quad
g=\langle S,z\rangle_V,\quad
c=\langle S,e\rangle_V,\quad
c_z=\langle z,e\rangle_V .
\]

After quotienting null directions, define

\[
s_\rho=\langle z,z\rangle_V-g^\top G^+g,
\qquad
h_\rho=c_z-g^\top G^+c .
\]

The exact incremental gain is

\[
\boxed{
\Delta_\rho=
\begin{cases}
h_\rho^2/s_\rho,&s_\rho>0,\\
0,&s_\rho=0.
\end{cases}}
\]

Thus the novelty gate is $s_\rho>0$, not nonzero norm of the total K31 scalar.  If a
target-blind residual provider has endpoint projection norm $F_\rho$ and deterministic
Bellman/provider discrepancy bound \(\eta_\rho\), its held gain is certified by

\[
\boxed{\Delta_{\rho,\rm cert}\ge(F_\rho-\eta_\rho)_+^2.}
\]

At exact frame/H3, and only after proving that this residual owns the relevant q4 remainder,
crossing needs

\[
\Delta_{\rho,\rm cert}\ge V_4-m=2.86\times10^{-10},
\]

or $F_\rho\ge\eta_\rho+1.691\times10^{-5}$.  In general the gate is

\[
\boxed{
(F_\rho-\eta_\rho)_+^2\ge V_4-m_{\rm rob}+L_{FH}.
}
\]

It is conditional on an attribution proof that \(\rho_{31}\) can supply that portion of
$V_4$.  The necessary ceiling test is

\[
\lVert P_S^\perp\rho_{31}\rVert_V^2
\ge V_4-m_{\rm rob}+L_{FH};
\]

failure closes the route even when the source is nonzero.

### Acquisition-cost correction

The universal identity

\[
\kappa_4(U,U,U,V)=E[U^3V]-3E[U^2]E[UV]
\]

is exact for arbitrary laws but does not make the mixed fourth moment free.  Generic direct
acquisition of $E[U^3V]$ is a fourth-action, $n^4$-scale provider unless the engine already
exposes the moments.  The affine--quadratic formula is cheaper, but computes the assembled old
same-site action rather than the beyond-AQ/collision-free residual.

Even ignoring that ownership failure, every generic aggregate branch exceeds the one-unit
slack.  For packed $r=32,s=528$ matrix/vector state, avoid the unnecessary dense
$P=WQ^\top$ by associativity:

\[
R=Q(W^\top M)=QN,\qquad r=Qu.
\]

From pre-$W$ state the exact cost is

\[
\frac{s+r}{n}+\frac{s+r}{n}+\frac{2r^3}{n^2}
=1.15625\quad\text{per layer},
\]

and from already-carried post-$W$ $(N,u)$ it is

\[
\frac{s+r}{n}+\frac{2r^3}{n^2}
=0.609375\quad\text{per layer}.
\]

Across D10 these are $11.5625$ and $6.09375$.  At $r=45,s=1035$, the
factorable gradient/Gram implementation is about $1.23$ per layer, or $12.3$ over D10,
when $(N,u,H,Q)$ are reused.  Missing $N$ raises this to about $2.24$ per layer; missing
$H$ raises it to about $4.24$.  Explicit $P$ and independent $N/R$ mixes are dominated.

The literal diagonal $V=U$ computation can fit the one-unit slack through local $A^2$,
but the ownership theorem places exactly that same-site action in the old D11/Q24 span.  It is
a cheap solved diagnostic, not a source of new raw gain.

Thus D14+D10 keeps the residual route alive only if the deployed engine already exposes an
exact residual/collision-free action at marginal cost

\[
\boxed{C_\rho(\mathcal J)\le1.}
\]

Neither generic direct moments nor generic AQ aggregation meets this gate; the latter also
fails novelty.

### Final Pareto/master certificate

For primary depth $d$, reverse suffix $t$, and residual-action schedule \(\mathcal J\),

\[
\boxed{2d+2t+C_\rho(\mathcal J)\le49.}
\]

Let $L_{FH}(d,t)$ be frame/H3 value lost relative to the full shared stream, let
\(\Delta_\rho(\mathcal J;d,t)\) be held residual-K31 Schur gain, and let the two remaining
source errors have normalized cross-correlation \(\xi\).  A sufficient raw-risk certificate is

\[
\boxed{
L_{FH}+V_4-\Delta_\rho
+2\xi\sqrt{L_{FH}(V_4-\Delta_\rho)}
+E_{\rm shift}\le m.
}
\]

The cross sign must be known or bounded.  The worst-case safe form is

\[
\boxed{
\left(\sqrt{L_{FH}}+\sqrt{(V_4-\Delta_\rho)_+}\right)^2
+E_{\rm shift}\le m.
}
\]

Equivalently set $m_{\rm rob}=m-E_{\rm shift}$ and omit $E_{\rm shift}$ on the left;
never charge it twice.  One frozen-coefficient perturbation bound is

\[
E_{\rm shift}
\le\epsilon_M+2\sqrt{\Delta_\rho}\,\epsilon_c
+\delta_G\Delta_\rho,
\]

where \(\epsilon_c\) is normalized cross-covariance drift, \(\delta_G\) relative Gram drift,
and \(\epsilon_M\) the remaining model/source mismatch charge.  The Bellman residual radius
must be assembled from typed, non-overlapping defects:

\[
\eta_\rho\le
\eta_{\rm init}+\eta_{\rm supp}+\eta_z+\eta_{22}+\eta_{\ge5}
+\eta_{\rm frame}+\eta_{\rm state}+\eta_{\rm coll}
+\eta_{\rm num}+\eta_{\rm term}.
\]

Shortening a stream is Pareto-improving only when added certified residual-action gain exceeds
the induced frame/H3 loss and the increase in cross/shift charges.  For reallocation
$A\to B$, retain $B$ only if

\[
\Delta_\rho(B)-\Delta_\rho(A)
>L_{FH}(B)-L_{FH}(A)+\Delta E_{\rm cross}+\Delta E_{\rm shift}.
\]

### Earliest falsifiable promotion gate and kill order

The first gate is a residual-provider proof, before any value run:

1. Prove a target-blind observable equals a component of
   $K_{31}^{\rm true}-K_{31}^{\rm assembled}$, not the old assembled same-site action.
2. Prove its full marginal acquisition and transport cost is at most one under D14+D10, or
   exhibit a shorter $(d,t)$ schedule satisfying the Pareto inequality.

If either fails, close the scalar-K31 residual lane without a numerical screen.  An AQ cumulant
formula, diagonal $V=U$ identity, or reproduction of $CP4+3\alpha vC+H31$ fails the first
gate by construction.  If both pass, apply the remaining gates in order:

3. Schur novelty $s_\rho>0$.
4. Ceiling/value $(F_\rho-\eta_\rho)_+^2\ge V_4-m_{\rm rob}+L_{FH}$.
5. Typed residual, cross-term, and coefficient-shift budget fits
   $m=7.453\times10^{-10}$.

The current cheap scalar identities certify old ownership, not a new source.  The sole open
scalar route is a cheap, target-blind provider for the collision-free/beyond-assembled residual.

### Graph-ready nodes (kept separate)

**K31_OLD_ACTION_SOLVED** — status: solved / support-only.  Criterion:

\[
\langle\Lambda,K_{31}^{\rm same\mbox{-}site\ target}\rangle
=\langle\Lambda,K_{31}^{\rm assembled}\rangle
=\langle\Lambda,CP4+3\alpha vC+H31\rangle\in\mathcal O.
\]

This node receives no independent gain allocation.

**K31_RESIDUAL_OPEN** — status: open, provider-blocked.  Candidate:

\[
\rho_{31}=\langle\Lambda,K_{31}^{\rm true}-K_{31}^{\rm assembled}\rangle.
\]

Terminal criterion: target-blind provider identity plus nonzero Schur residual.

**SCALAR_K31_RAW_MASTER** — status: derived, conditional.  Terminal criterion:

\[
L_{FH}+V_4-\Delta_\rho+E_{\rm cross}+E_{\rm shift}
\le7.453\times10^{-10}.
\]

**SCALAR_K31_SCHUR_GAIN** — status: blocked on residual provider.  Terminal criterion:

\[
s_\rho>0,\qquad
(F_\rho-\eta_\rho)_+^2\ge V_4-m_{\rm rob}+L_{FH}.
\]

**SCALAR_K31_ACQUISITION_PARETO** — status: current generic branches cost-closed.  Terminal
criterion $2d+2t+C_\rho\le49$; at D14+D10, $C_\rho\le1$.  Generic AQ aggregate costs at
least $6.09375$ over D10 from post-$W$ state and computes old ownership; generic
arbitrary-law moments are fourth-action scale.

**SCALAR_K31_RESIDUAL_SHIFT** — status: blocked on typed bounds.  Terminal criterion: Bellman
defect radius, source cross term, and frozen-network drift jointly fit the remaining
$7.453\times10^{-10}$ margin without double counting.

**SCALAR_K31_KILL_ORDER** — status: derived active rule.  Terminal criterion: provider identity
and acquisition cost first; then Schur novelty; then value/ceiling; then residual and shift
robustness.  Failure of either first-stage proof is an immediate no-run close.

### Current next question

Does any state already visible on the deployed D10 reverse suffix encode the collision-free or
beyond-assembled component of

\[
K_{31}^{\rm true}-\left(K_{31}^{\rm CP4}+3\alpha vC+H31\right)
\]

with no more than one extra GEMM-equivalent and endpoint conditional value at least
$V_4-m_{\rm rob}+L_{FH}$?  This is the unique scalar-K31 promotion question.  Re-deriving the
same-site affine--quadratic cumulant cannot answer it.

## Checkpoint 9: surviving portfolio after same-site K31 closure

This checkpoint ranks three genuinely different possibilities after quotienting the complete
D11/Q24 span:

1. strict-depth, true-minus-assembled K31 completion;
2. structured all-distinct H3 evaluated by its adjoint at birth; and
3. a signed post-AQ cubic dictionary.

The ranking must separate raw capacity, arithmetic cost, and information provenance.  A large
formal span has no deployable value when its coordinates are not target-blind observables; a
cheap action has no raw value when it is already in the incumbent span.

### Common conditional-Gram formulation

Let (P_0) project onto all incumbent D11/Q24 features and put

\[
d_0=(I-P_0)d.
\]

Let (U_A,U_C(k),U_K) denote, respectively, the incumbent-residualized endpoint response
spaces of the all-distinct birth circuit, a rank-(k) post-AQ cubic dictionary, and the
strict-depth K31 residual.  For any selected family \(\mathcal I\), its exact joint gain is

\[
\boxed{
\Delta(\mathcal I)=
\left\lVert P_{\sum_{i\in\mathcal I}U_i}d_0\right\rVert_V^2 .
}
\]

Standalone gains must not be added.  For example, the lawful marginal dictionary gain after
all-distinct and strict-depth sources is

\[
\Delta_{C\mid AK}
=d_0^\top\left(P_{U_A+U_K+U_C}-P_{U_A+U_K}\right)d_0,
\]

and analogously for every other ordering.  These projector differences, or the equivalent
Schur complements, are the binding double-count audit.

After an exact variance/frame reduction, the staged residual gap is

\[
D_F=1.4656\times10^{-8}-10^{-8}=4.656\times10^{-9}.
\]

A selected portfolio crosses the raw target only if

\[
\boxed{\Delta(\mathcal I)\ge D_F+E_{\rm rob},}
\]

where (E_{\rm rob}) includes approximation, cross-source, and frozen-network shift charges
exactly once.

### Candidate A: all-distinct H3 adjoint at birth

For the structured source

\[
S_{ijk}=\beta_i c_{ij}c_{ik}+\beta_jc_{ji}c_{jk}+\beta_kc_{ki}c_{kj},
\]

with pairwise-distinct (i,j,k), its contraction against an arbitrary vector (x) and
symmetric Bellman matrix (M) is exactly

\[
\begin{aligned}
S(x,M)={}&\sum_i\beta_ix_i
\left[c_i^\top Mc_i-\sum_jc_{ij}^2M_{jj}\right]\\
&+2\sum_i\beta_i\left[
(c_i^\top x)\left(\sum_kc_{ik}M_{ik}\right)
-\sum_jc_{ij}^2x_jM_{ij}\right].
\end{aligned}
\]

Only the (CM) product is cubic-cost, so each active birth costs one square-GEMM equivalent.
The discrete Green identity proves that reverse evaluation is exactly dual to carrying this
same source through the frozen tangent.  It has no transport approximation and no rank loss for
one scalar endpoint.

Its sharp raw-capacity ceiling is therefore the oracle value of that complete frozen chain,
not a sum of forward and reverse values.  For the certified chain,

\[
\boxed{
V_A^{\rm oracle}=0.187(2.9409\times10^{-8})
=5.499483\times10^{-9}.
}
\]

Thus, conditional on the exact frame result and no robustness charge, this route alone needs

\[
\frac{\Delta_A}{V_A^{\rm oracle}}
\ge\frac{4.656}{5.499483}=0.84663,
\qquad
\rho_A\ge0.92013
\]

when coefficient fidelity is one.  With robustness, replace the numerator by
(4.656\times10^{-9}+E_{\rm rob}).

The weakest information premise is only task-weighted source equality, not equality of the
entire source tensor.  At each net and layer it is sufficient that

\[
\boxed{
\left\langle Q_l,
dK_l^{\rm cert}-S_{AD}(\beta_l^D,C_l^D)
\right\rangle=0,
}
\]

or that the summed defect is bounded inside (E_{\rm rob}).  The deployed frame must generate
(\beta_l^D,C_l^D) label-free, and the reverse (J_l,Q_l) must be the tangent used by the
certified replay.  A truth-fed source or a missing source/query secant violates this premise.

There is no proved fusion between the independent (CM) product and the existing (WB),
(WBW^\top) reverse products.  With one reverse matrix word,

\[
\boxed{C_A=2L+|I_A|,}
\]

so (C_A=38) for ten active births and (42) for fourteen.  This is a **replacement** for the
shared D14 forward realization of the same chain, never an additive D14-plus-adjoint source.
If D14's same raw response is already scored, (Delta_{A\mid D14}=0); the adjoint then has
only implementation value.  Added beside D14+D10=48, the ten- or fourteen-layer form is
cost-closed.

The only exact sub-GEMM exception is a proved bounded common-rank factorization

\[
C=ZL^\top,
\qquad
\operatorname{diag}(CMC^\top)
=\operatorname{diag}\!\left[Z(L^\top ML)Z^\top\right],
\]

whose complete factor formation, reset closure, and charged-cut rank must fit the remaining
unit.  Local cubic-jet algebra does not imply this rank condition.

An approximate one-unit exception uses a frozen Rademacher Hutchinson sketch.  For
(A=CMC^\top), weights (w_i=\beta_i x_i), and independent sign probes (z_a), define

\[
\widehat{s}_k
=\frac1k\sum_{a=1}^k z_a^\top\operatorname{diag}(w)A z_a.
\]

It is unbiased for (w^\top\operatorname{diag}(A)), with exact conditional variance

\[
\boxed{
\operatorname{Var}(\widehat{s}_k\mid A,w)
=\frac1k\sum_{i<j}(w_i+w_j)^2A_{ij}^2.
}
\]

At most roughly 34 probes fit the cited one-unit implementation.  This branch is promotable
only if the frozen-probe endpoint error, summed through all active births and networks, fits
(E_{\rm rob}).  Unbiasedness alone is not an MSE certificate.

### Candidate C: signed post-AQ cubic dictionary

Write the novel local defect as

\[
X_i=X_i^{AQ}+I_3(Z_i),\qquad Z_i\in\operatorname{Sym}^3(\mathbb R^r).
\]

At (r=32), (p=\dim\operatorname{Sym}^3(\mathbb R^{32})=5984).  Under the literal
(n^2) fixed-state ceiling, after the (M_i,a_i) AQ state is stored, a generic shared
dictionary (Z=CD) permits only

\[
k_{\rm gen}\le67,
\]

whereas a symmetric-CP dictionary

\[
Z_i=\sum_{q=1}^k c_{iq}v_q^{\otimes3}
\]

permits

\[
k_{\rm CP}\le449.
\]

Let \(\sigma_j\) be singular values of the full post-AQ defect after both incumbent
residualization and whitening by the endpoint value metric.  For a generic rank-(k)
dictionary, Eckart--Young makes

\[
\boxed{
\sum_{j>k}\sigma_j^2\le E_{\rm dict}
}
\]

the exact capacity gate.  For a CP dictionary replace the left side by the best (k)-term
symmetric-CP error; singular-value decay is necessary but not sufficient.

If this dictionary alone must supply the combined ideal H3-plus-q4 remainder, its total oracle
pool is at most

\[
V_{34}=4.3700\times10^{-9}+1.0313\times10^{-9}
=5.4013\times10^{-9},
\]

and raw crossing requires

\[
\boxed{
\Delta_C\ge4.656\times10^{-9}+E_{\rm rob},
\qquad
\frac{\Delta_C}{V_{34}}\ge0.8620
}
\]

at zero robustness charge.  Equivalently the residualized tail must be at most (13.80\%) of
the combined pool, or (0.7453\times10^{-9}).  This is an upper-capacity statement, not a
current certificate: no qualifying singular/CP tail or target-blind coefficient provider has
yet been proved.

For the CP consumer, the exact rectangular route costs (4k/n) per active layer in addition
to the shared AQ state path.  With pre-(W) AQ cost (g_{AQ}=1.15625) per layer and no other
unshared lane,

\[
\boxed{
C_C(k)=L\left(g_{AQ}+\frac{4k}{n}\right)\le49.
}
\]

At (L=14,n=1024,k=449), memory binds first and (C_C=40.7422), leaving (8.2578)
GEMM-equivalents.  If the post-(W) (N,u) state is genuinely already carried, replace
(g_{AQ}) by (0.609375), leaving (15.9141).  Any (H,Q), gauge transport, or atom reset
not actually shared must be added to this census.

Its information premise is stronger than the all-distinct circuit.  Covariance reveals only

\[
\Delta C_{ij}=3!\langle Z_i,Z_j\rangle,
\]

which is invariant under (Z_i\mapsto-Z_i), while the required H3--H1--H1--H1 action changes
sign.  Therefore the atoms and signed coefficient bank require reset-local analytic provenance
or train-fold signed supervision frozen before the sentinel, together with a common latent
gauge and reset-invariant atom dictionary.  H2/covariance data alone fails identifiability.

### Candidate K: strict-depth D11-kernel completion

For births (b_j), transport (T_j), consumers (C_k), and Bellman rows (B_k), the only
novel action is the future-only sum

\[
\boxed{
\Delta_<
=\sum_{j<k}\left\langle
B_k,C_kT_{k-1}\cdots T_{j+1}b_j
\right\rangle .
}
\]

The (j=k) diagonal is old D11/Q24 ownership.  Equivalently the birth costate is

\[
\mu_j=T_j^*\lambda_{j+1}
=\lambda_j-C_j^*B_j,
\]

not the unresidualized \(\lambda_j\).  Let (F) be the full incumbent feature map,
(\Sigma_{\perp F}) its conditional birth covariance, (g) the oracle residual row, and
(O_<) the true-minus-assembled strict-lower observation.  The sharp captured fraction is

\[
p_{<}=
\frac{
(O_<\Sigma_{\perp F}g)^*
(O_<\Sigma_{\perp F}O_<^*)^\dagger
(O_<\Sigma_{\perp F}g)
}{g^*\Sigma_{\perp F}g},
\]

and

\[
\boxed{
\Delta_K^{\max}=p_{<}\Delta R_{<}^{\rm oracle}
\le\Delta R_<^{\rm oracle}.
}
\]

With perfect H3 it needs at least (0.2860\times10^{-9}).  Only if the verified
strict-depth oracle denominator equals the entire (1.0313\times10^{-9}) q4 pool may this be
restated as $p_{<}\ge0.27732$.  Diagonal ownership makes that denominator an open premise.

Its information premise is exact and severe:

\[
\boxed{
\ker(D_{new}|_{U_<})\subseteq\ker\Phi_< .
}
\]

Incumbent repeated-index moments cannot satisfy it.  Positive parity laws exist with identical
means, covariance, all third moments, K31, K22, and marginal K4, but opposite collision-free
fourth cumulant; after a dense reset their delayed K31 differs.  Thus the candidate requires a
genuine fresh-law observation or the actual precontracted target-specific residual scalar.

With D14+D10 consuming 48, an additive implementation fits iff

\[
\boxed{
G_{\rm unshared}+\sum_{k\in I_<}\frac{N_k}{n}\le1.
}
\]

For the ordinary (N_k=n) interface, every multi-layer form is closed.  The only exceptions
are an already-exposed (O(n^2)) scalar interface, one charged-cut rank-one contraction with
(\sum_kN_k\le n), or one consumer layer whose conditional capture alone clears the value
gate.  The standard ten-layer realization survives only by replacing D10:

\[
C_K^{\rm replace}=28+10=38,
\]

and must satisfy

\[
\Delta_K-\Delta_{D10,\rm lost}\ge\Delta_{\rm required,new}.
\]

### Capacity, cost, and information ranking

The conditional ranking is:

| Candidate | Sharp raw-capacity ceiling | Standard cost | Weakest lawful information premise |
|---|---:|---:|---|
| All-distinct adjoint (A) | Certified frozen-chain oracle (5.499483\times10^{-9}) | (38) for L5--14, (42) for all 14; replaces D14 | Deployed (\beta,C,J,Q) satisfy task-weighted source/query equality |
| Post-AQ dictionary (C) | At most (V_{34}=5.4013\times10^{-9}); rank-(k) tail is binding | CP (k=449): (40.7422) with pre-(W) AQ path | Signed cubic atoms/coefficients, common gauge, reset closure; not inferable from H2 |
| Strict-depth K31 (K) | (p_<\Delta R_<^{oracle}), nominally no more than the residual q4 pool | Additive multi-layer form closed at (C\le1); D10 replacement (38) | Fresh collision-free law or actual precontracted residual action |

Thus (A) ranks first on currently proved raw capacity and weakest provenance.  (C) is the
high-capacity fallback but has an unproved signed-state/rank premise.  (K) ranks last in raw
capacity and cannot replace an H3-scale source; it is only a residual closer after an H3 route
has nearly crossed.

The cost order is not a single scalar ordering.  Exceptional one-unit (K) is cheapest but
rests on the strongest oracle/fusion premise.  Standard (K)-replacement and ten-layer (A)
both cost 38; full-layer (A) costs 42; maximal CP (C) costs 40.7422 under its favorable
sharing premise.  All must also pass adjusted-score break-even.  For a gain \(\Delta\) and
total added billing (N), the exact economic gate is

\[
\boxed{
N\le N_{\max}(\Delta)
=\frac{u_0\Delta}{\kappa(R_0-\Delta)},
\qquad
\Delta\ge\frac{N\kappa R_0}{u_0+N\kappa}.
}
\]

At (u_0=0.5709,kappa=0.00161,R_0=2.9409\times10^{-8}), costs 38 and 42 require
approximately (2.85\times10^{-9}) and (3.11\times10^{-9}), respectively.  The full
all-distinct (5.50\times10^{-9}) opportunity passes; the isolated
(0.286\times10^{-9}) q4 minimum cannot finance a standard ten-layer action.

### Exact containment and dominance rules

1. **Forward/adjoint duplication.**  If D14 already carries the same all-distinct source through
   the same tangent, Green duality gives (U_A\subseteq U_{D14}) for the scored scalar endpoint
   and \(\Delta_{A\mid D14}=0\).  Keep whichever implementation is deployable and cheaper;
   never score both.
2. **Strict-depth inside the full adjoint.**  When (K) consumes the same collision-free births
   and endpoint tangent evaluated by (A), it is one downstream consumer projection, so
   (U_K\subseteq U_A) and \(\Delta_{K\mid A}=0\).  (K) survives beside (A) only after an
   explicit source/query mismatch creates a Schur residual.
3. **Dictionary containment.**  If one reset-invariant atom family represents the structured
   all-distinct source, then (U_A\subseteq U_C(k)).  A common-eigenbasis Jordan atom
   (\operatorname{Sym}(v\otimes e^{\otimes2})) can be polarized into at most three cubic CP
   atoms, and a rank-(r) common-basis source into at most (2r+1) atoms.  Unit-varying
   eigenbases or non-invariant resets destroy this containment.
4. **No automatic K/C containment.**  A strict-depth collision-free H4 direction is not
   generally a cubic tangent direction.  It may stack only through
   \(\Delta_{K\mid AC}\), never by adding standalone gains.
5. **Cost dominance.**  Candidate (i) dominates (j) only if (U_j\subseteq U_i), its
   information premise is no stronger, and (C_i\le C_j).  Span containment alone does not
   dominate when (i)'s signed state is unavailable.

### Next minimum stack

Let (F) denote the indispensable variance/frame lane and (D_F) its held residual gap.  The
minimum non-double-counted stack is selected in this order:

1. **Two-source preferred stack ({F,A}).**  Promote only if the deploy-frame
   task-weighted equality holds, the adjoint is a replacement rather than a duplicate add-on,
   and

   \[
   \Delta_{A\mid F}\ge D_F+E_{\rm rob}.
   \]

   At the staged exact-frame point this is the (84.663\%) retained-value gate above.  If it
   passes, neither a strict-depth K31 slice of the same chain nor a dictionary representation of
   the same birth receives additional value.
2. **Two-source fallback ({F,C_k}).**  Use the smallest (k) whose incumbent-conditional
   dictionary tail fits (E_{\rm dict}\) and whose signed provider and full cost are proved.  At
   the staged point it must capture at least (86.20\%) of (V_{34}).  Do not add (A) when
   (U_A\subseteq U_C(k)); otherwise add only the Schur residual of one against the other.
3. **Three-source residual stack ({F,H,K}).**  Here (H) is the surviving nonduplicated
   part of (A) or (C_k).  (K) is admitted only when

   \[
   \Delta_{K\mid FH}\ge
   D_F-\Delta_{H\mid F}+E_{\rm rob},
   \]

   its residual oracle ceiling can meet that gap, and it either satisfies the exceptional
   one-unit gate or repays the value lost by replacing D10.  With perfect H3 the right side is
   (0.2860\times10^{-9}+E_{\rm rob}).
4. If neither (A) nor (C_k) supplies the H3-scale gain, kill the present portfolio.  The
   strict-depth K31 ceiling is too small to bridge (4.656\times10^{-9}) by itself.

### Exact dominance/kill tree

1. **Old-span gate:** project every candidate through (I-P_0).  Same-site AQ/K31 is zero and
   closes immediately.
2. **Information gate:**
   - (A): kill the certified-capacity claim on any endpoint-visible truth-dependent supply or
     missing query secant;
   - (C): kill if only H2/covariance information fixes (Z), because (Z\mapsto-Z) is
     invisible;
   - (K): kill incumbent-only formulas by the positive parity witness; require a fresh-law
     interface satisfying the kernel inclusion.
3. **Replacement/containment gate:** identify D14--(A), (A)--(C), and (A)--(K)
   containment before assigning gains.  A contained candidate is support-only unless cheaper.
4. **Capacity gate:** apply the exact conditional projector gain.  Kill (A) below the
   (84.663\%) retained threshold, dictionary (C_k) above the allowed endpoint tail, and
   (K) below the conditional residual gap.
5. **Cost gate:** kill additive multi-layer (A) or (K) beside the 48-GEMM base.  For
   replacements apply (C_A=2L+|I_A|), (C_C(k)), or the exact K charged-cut census and repay
   every removed source's value.  Reopen additive (A) only with a proved bounded common-rank
   (C) factor or a frozen Hutchinson sketch whose exact endpoint error fits the margin.
6. **Economic/robustness gate:** require both (C\le N_{\max}(\Delta)) and the joint raw master
   inequality with cross/shift charges.  Passing raw capacity without adjusted-score capacity
   is not a promotion.

### Checkpoint-9 graph-ready split nodes

**PORTFOLIO_CONDITIONAL_GRAM** — status: solved theorem.  Criterion: all candidate values are
projector differences after (P_0), never sums of standalone gains.  Terminal equation:

\[
\Delta(\mathcal I)=\lVert P_{\sum U_i}(I-P_0)d\rVert_V^2.
\]

**ALLDIST_H3_BIRTH_IDENTITY** — status: solved theorem.  Criterion: retain the exact arbitrary
((x,M)) contraction and discrete Green duality; one (CM) product per active birth.

**ALLDIST_H3_PROVENANCE** — status: open, highest priority.  Terminal criterion: prove the
task-weighted deployed-source equality for every net/layer, including the source/query secant,
or produce an endpoint-visible truth-dependent residual and close the certified-capacity claim.

**ALLDIST_H3_CAPACITY** — status: conditional.  Terminal criterion:

\[
\Delta_{A\mid F}\ge4.656\times10^{-9}+E_{\rm rob};
\]

nominal retained fraction at least (0.84663) of the certified (5.499483\times10^{-9})
oracle.

**ALLDIST_H3_REPLACEMENT_COST** — status: conditional pass under relaxed economics, closed as
an additive neighbor of D14+D10.  Terminal criterion: (C_A=38) or (42), explicit D14
replacement, no (CM) fusion credit, and adjusted-score break-even.

**ALLDIST_H3_SUBGEMM_EXCEPTION** — status: open exceptional branch.  Terminal criterion:
either a reset-stable bounded-rank factor (C=ZL^\top) whose full charged cost is at most one,
or a frozen (k\le34) Hutchinson action whose exact endpoint error/variance fits
(E_{\rm rob}).  Generic cubic-jet locality does not pass this node.

**POSTAQ_CUBIC_SIGNED_PROVIDER** — status: open / information-blocked.  Terminal criterion:
target-blind signed (Z_i) coefficients and atoms in a common reset-stable gauge.  H2-only
provenance is terminally closed by sign ambiguity.

**POSTAQ_CUBIC_RANK_CAPACITY** — status: open.  Terminal criterion: generic top-(k) tail or
best-(k) CP error in the incumbent-conditional endpoint metric is at most
(0.7453\times10^{-9}-E_{\rm rob}); (k\le67) generic or (k\le449) CP.

**POSTAQ_CUBIC_COST** — status: conditional.  Terminal criterion:

\[
L(g_{AQ}+4k/n+g_{\rm other})\le49
\]

with every supposedly shared (H,Q,N,u) interface proved on the deployed lineage.

**K31_STRICT_CAPACITY** — status: open.  Terminal criterion: verify the delayed residual oracle
denominator and require $p_{<}\Delta R_{<}^{oracle}$ to clear the conditional portfolio gap; do
not reuse the (27.732\%) threshold until the denominator equals (1.0313\times10^{-9}).

**K31_STRICT_INFORMATION** — status: incumbent-only branch closed, genuine fresh-law branch
open.  Terminal criterion:

\[
\ker(D_{new}|_{U_<})\subseteq\ker\Phi_<.
\]

**K31_STRICT_ADD_COST** — status: ordinary multi-layer form closed.  Reopen only for a proved
precontracted (O(n^2)) action, joint rank-one charged cut, or one-layer capture clearing both
raw and economic gates.

**K31_STRICT_REPLACE_D10** — status: open fallback.  Terminal criterion: total cost at most 38
and $\Delta_K-\Delta_{D10,lost}$ clears the remaining raw/robust gap.

**SURVIVING_MIN_STACK** — status: open.  Priority order: ({F,A}), then ({F,C_k}), then
only the Schur-residual ({F,H,K}).  Terminal criterion: one branch passes provenance,
conditional-Gram capacity, full census, adjusted-score economics, and robustness simultaneously.

### Current next question

The earliest and highest-value proof is whether the certified (18.7\%) chain's per-layer
supply is task-weightedly identical to the deployed all-distinct source.  If yes, the exact
38-GEMM L5--14 adjoint is the minimum high-capacity realization and already contains its
strict-depth K31 consumers.  If no, the next question is whether the endpoint-visible mismatch
has a reset-stable signed cubic dictionary with a sufficiently concentrated top (k) tail.
Strict-depth K31 is considered only after one of those H3-scale routes has reduced the remaining
gap to its verified residual-q4 capacity.

## Checkpoint 9 cost erratum: branch-specific all-distinct census

This supersedes the unconditional (38/42) and additive-cost-close wording above for the scoped
(r=45), ten-transition implementation.  The exact conclusion is conditional on reverse-word
reuse.

If the reverse (H=WBW^\top) word is already exposed by the deployed lineage, the paid
non-source path costs (22.83) GEMM-equivalents.  Ten independent
(\operatorname{diag}(CMC^\top)) birth contractions add ten, hence

\[
\boxed{C_A^{H\,reused}=22.83+10=32.83<49.}
\]

If (H) must be rebuilt, its two dense products per transition add twenty:

\[
\boxed{C_A^{H\,new}=22.83+20+10=52.83>49.}
\]

Thus the exact all-distinct adjoint is not universally cap-closed or cap-open.  The first
falsifiable cost gate is an interface receipt proving that the exact (H) used by its Bellman
query is already billed and reusable.  The generic (CM) diagonal remains one full product per
active layer; local cubic-jet fusion does not remove it.

For an exact structural factor (C=ZL^\top), let (x=k/n).  Its marginal charged cost is

\[
g_{rank}(x)=x+2x^2
\]

per active layer.  Under the worst-branch per-layer slack (0.617), it fits exactly when

\[
\boxed{x+2x^2\le0.617,}
\]

which gives (x\le0.359), approximately (k\le367) at (n=1024).  The factor must be common
in the required gauge and reset-stable; a local low-rank fit that changes atoms across layers
does not pass.

Graph status correction: **ALLDIST_H3_REPLACEMENT_COST** is `open_interface`, with a proved
32.83 pass branch under exact (H)-reuse and a proved 52.83 fail branch without it.
**ALLDIST_H3_SUBGEMM_EXCEPTION** retains the rank inequality above or the separately bounded
Hutchinson branch.

## Checkpoint 10: transient Z22 versus strict-depth K31 and signed-H3 fallback

### Typed one-step observable

At net (\nu) and post-reset layer (l), define the connected, old-provider-subtracted
off-diagonal pure-22 residual

\[
Z_{l\nu}=\operatorname{offdiag}\operatorname{sym}\left{
K22^{target}-K22^{CP4}
-\alpha\left[vv^\top+2C\odot C\right]
\right}.
\]

No additional Gaussian disconnection is subtracted because (K22^{target}) is already
connected.  For the next dense map (W), put (W_2=W\odot W).  If the full future
off-diagonal K31 costate is (\Lambda_{31}), then

\[
\boxed{
\chi^Z_{l\nu}
=3\left\langle Z_{l\nu}W_2,
W\odot(W\Lambda_{31}^\top)\right\rangle_F .
}
\]

Sum only over the explicitly credited one-step births.  The costate may contain all downstream
response, but (Z) itself is transient; this formula is not a D10 recurrence.

Let (O) be the **complete** old CP4/star/pair/depth arena over frozen nets, with endpoint
metric (M), and define

\[
P_O=O(O^\top M O)^+O^\top M,
\qquad
z=(I-P_O)\chi^Z.
\]

Residualization against only the local CP4 or fitted (\alpha) provider is insufficient.

### Full-interface dark-sector theorem and its limitation

For a hidden quartic state at layer (l), let (N_l=\ker O_l) and let (F_l) stack every
future old-interface read and terminal endpoint action.  The missing future-visible quotient is

\[
N_l/(N_l\cap\ker F_l),
\]

with dual cyclic module

\[
\mathcal C_l=\operatorname{range}(P_{N_l}F_l^*).
\]

An added observable (A_l) is exactly future-sufficient iff

\[
\boxed{
\ker\begin{bmatrix}O_l\\A_l\end{bmatrix}
\subseteq\ker F_l
\quad\Longleftrightarrow\quad
\mathcal C_l\subseteq
\operatorname{range}(P_{N_l}A_l^*).
}
\]

At (n=1024), even after a rank-2048 CP4 tangent is removed, the coordinate-collision and
CP4-dark (H_{211}\oplus H_{1111}) intersection has dimension at least

\[
46{,}078{,}231{,}296.
\]

This proves that the old interface is not universally future-sufficient.  It does **not** prove
that the lawful (Z22) source intersects the endpoint-valued part of that dark sector.  The
promotion witness remains (z\ne0), followed by the Schur/value gate below.  Likewise,
source-level H3/H4 Wiener orthogonality cannot be used as endpoint orthogonality after dense
resets and complete-arena residualization.

If (A_Z) denotes the one-step Z22 observable and hidden-state radius is (\rho_l), its sharp
uncovered future radius is

\[
\boxed{
\eta_{Z,l}=\rho_l\left\|
\left(I-P_{\operatorname{range}(P_{N_l}A_Z^*)}\right)
P_{N_l}F_l^*
\right\|_{op}.
}
\]

Exact future sufficiency is (\eta_{Z,l}=0); partial promotion is lawful only after the summed
endpoint effect of these radii is charged to the robust margin.

### Complete shared-old-span Gram

Let (P) collect every already-selected non-old source, such as the frame lane and whichever
H3 provider is being retained.  All its columns, and the strict-depth and signed-H3 candidates
(k,h), are first residualized by (I-P_O).  Put

\[
r=(I-P_P)(I-P_O)d,
\qquad
z_P=(I-P_P)z.
\]

The exact marginal Z22 gain is

\[
\boxed{
\Delta_{Z\mid P}
=\frac{\langle r,z_P\rangle_M^2}
{\langle z_P,z_P\rangle_M},
}
\]

with value zero when (z_P=0).  For a general multicolumn Z22 library, replace this scalar
quotient by (c_Z^\top G_Z^+c_Z).

For two scalar candidates (z,k) after the same base projection, define

\[
a_z=\frac{\langle r,z\rangle}{\lVert z\rVert},\quad
a_k=\frac{\langle r,k\rangle}{\lVert k\rVert},\quad
\gamma=\frac{\langle z,k\rangle}{\lVert z\rVert\lVert k\rVert}.
\]

When (|\gamma|<1), their exact joint gain and Z-after-K marginal are

\[
\boxed{
\Delta_{ZK}
=\frac{a_z^2-2\gamma a_za_k+a_k^2}{1-\gamma^2},
\qquad
\Delta_{Z\mid K}
=\frac{(a_z-\gamma a_k)^2}{1-\gamma^2}.
}
\]

The same block-Schur formula applies to signed H3:

\[
s_{Z\mid H}=g_{ZZ}-g_{ZH}G_H^+g_{HZ},
\qquad
q_{Z\mid H}=c_Z-g_{ZH}G_H^+c_H,
\]

\[
\boxed{
\Delta_{Z\mid H}=q_{Z\mid H}^2/s_{Z\mid H}.
}
\]

These cross terms are binding.  Pure-22 source typing does not establish orthogonality to
strict-depth K31, because one dense reset maps Z22 into K31/K4/K22.  Signed H3 source-chaos
orthogonality also does not establish (g_{ZH}=0) in the nonlinear endpoint metric.

### Raw-value and robust promotion gates

After exact frame and H3 channels, the remaining raw improvement required is

\[
D_4=V_4-m=2.860\times10^{-10}.
\]

Let (\eta_Z) be the aggregate amplitude bound from full-interface omission, provider error,
the uncredited K22 branch, and network shift.  If

\[
F_{Z\mid P}=rac{|\langle r,z_P\rangle_M|}{\lVert z_P\rVert_M},
\]

then the exact conservative promotion gate is

\[
\boxed{
\left(F_{Z\mid P}-\eta_Z\right)_+^2
\ge2.860\times10^{-10}+E_{\rm other}.
}
\]

With zero discrepancy this requires

\[
F_{Z\mid P}\ge\sqrt{2.860\times10^{-10}}
=1.69115\times10^{-5}.
\]

The (2.86\times10^{-10}) floor applies only after an H3-scale source has been retained.  If
Z22 is proposed to **replace** the signed-H3 fallback while the base contains only the exact
frame lane, it must instead satisfy

\[
\boxed{
\Delta_{Z\mid F}\ge4.656\times10^{-9}+E_{\rm rob}.
}
\]

Passing the small q4 floor is therefore insufficient to displace an H3 provider.

### Exact one-GEMM and physical-completeness gates

Put (R=ZW_2) and (B=W\Lambda_{31}^\top).  The marginal census is

\[
\boxed{
C_Z=C_{Z,acq}+1+C_{B,new}\le1.
}
\]

Thus the one-GEMM branch requires all three facts:

1. (Z) is transiently and target-blindly available with no unshared dense acquisition;
2. (W_2) is the already-owned D11 quantity; and
3. the exact (B) used by the endpoint costate is already exposed and reusable.

If (B) must be formed, (C_Z\ge2) and the 48/49 additive branch closes.  K4 diagonal
information can be reduced from rowwise diagonals of (W_2^\top R) without another dense
product.  The full K22 variation contains the crossed quadratic form

\[
2(w_a\odot w_b)^\top Z(w_a\odot w_b),
\]

which is not included in the one-GEMM scalar.  Its generic diagonal-completion rank at
(n=1024) is at least 980, so neither the one-unit additive slack nor the roughly eleven-unit
D10-replacement slack can carry a complete generic branch.  Promotion therefore requires

\[
\boxed{
\langle\Lambda_{22},\delta K22_{cross}(Z)\rangle=0
}

for a Bellman-dark branch, an exact shared/exceptionally low-rank reduction, or a deterministic
bound on this omitted endpoint amplitude included in (\eta_Z).

### Cost-aware displacement inequalities

Let a common base portfolio have raw risk (R_P) and utilization (u_P).  Candidate (j)
has conditional gain (\Delta_{j\mid P}) and marginal cost (C_j).  Z22 displaces (j)
in adjusted score if and only if

\[
\boxed{
(R_P-\Delta_{Z\mid P})(u_P+\kappa C_Z)
\le
(R_P-\Delta_{j\mid P})(u_P+\kappa C_j),
}
\]

equivalently

\[
\boxed{
\Delta_{Z\mid P}
\ge R_P-
\frac{(R_P-\Delta_{j\mid P})(u_P+\kappa C_j)}
{u_P+\kappa C_Z}.
}
\]

For equal one-unit costs this reduces exactly to
(\Delta_{Z\mid P}\ge\Delta_{j\mid P}).  If Z is added rather than swapped, its marginal
gain must exceed the exact billing break-even

\[
\boxed{
\Delta_{Z\mid P}
>\frac{\kappa C_ZR_P}{u_P+\kappa C_Z}
}
\]

as well as the raw-gap gate.

Accordingly:

- **Z22 versus strict-depth K31.**  Under the one-slot cap, choose Z iff its information,
  one-GEMM, K22-darkness, and robust-value gates pass and the adjusted displacement inequality
  with (j=K) holds.  An ordinary multi-layer K route is already infeasible additively, but an
  exceptional one-unit K oracle or thin-action realization must be compared by the full Gram;
  no typed containment is assumed.  If layer (l) has (N_l) atoms and residual Bellman-action
  rank (R_l), the exact thin-action census is

  \[
  \boxed{
  C_K^{thin}\le\sum_l\frac{2N_lR_l}{n^2},
  }
  \]

  or half this when one of the two projections is already exposed.  For ten full-width
  (N_l=n) layers, (C_K^{thin}\le1) requires average (R_l\le51), or (R_l\le102) with
  one projection reused.  These are genuine competitors only after proving pre-cut factor
  availability and charging the endpoint-weighted discarded-action tail.  This is Bellman
  action compression, not a rank-one H4 carrier.  After choosing Z, prune K only when its Schur marginal

  \[
  \Delta_{K\mid PZ}
  =c_{K\mid PZ}^\top G_{K\mid PZ}^+c_{K\mid PZ}
  \]

  fails either the remaining raw gap or its own billing break-even.
- **Z22 versus signed-H3 fallback.**  Z may displace H only if the larger
  (4.656\times10^{-9}) raw gate applies and the adjusted displacement inequality with (j=H)
  holds.  If H is already retained, Z is instead a q4 closer and is judged by
  (\Delta_{Z\mid PH}), including (g_{ZH}).  A source-chaos orthogonality assertion is not a
  substitute for this Schur test.
- **Potential containment.**  If a future proof establishes (U_Z\subseteq U_K), Z is a
  cheaper coordinate selection from K, not an additive source; it displaces K only if it
  preserves enough conditional value.  If (z_P=0) after K or H, it is an exact duplicate.

### Falsifiable Z22 decision tree

1. **Source/provenance:** prove (Z) is a deployed transient connected residual with frozen
   (\alpha).  If (K22^{target}) is truth-only, close before arithmetic.
2. **Complete old-span novelty:** form (z=(I-P_O)\chi^Z).  If (\lVert z\rVert_M=0), close
   as duplicate.  The ambient dark-sector dimension cannot override this result.
3. **One-GEMM interface:** require (C_{Z,acq}=0), reused (W_2), and reused (B).  Any new
   dense (B) product closes the additive branch.
4. **Full-physics residual:** prove the crossed K22 response Bellman-dark/shared/exceptionally
   low-rank, or bound it together with the uncovered future module in (\eta_Z).  Generic full
   K22 completion is cost-closed by rank at least 980.
5. **Conditional value:** after the actually retained source matrix (P), require
   ((F_{Z\mid P}-\eta_Z)_+^2\ge2.86\times10^{-10}+E_{other}).  If Z replaces H rather than
   follows it, require (4.656\times10^{-9}+E_{rob}).
6. **Swap economics:** apply the exact adjusted displacement inequality against K or H.  When
   costs match, retain the larger conditional Schur gain; when they do not, raw-gain ordering
   alone is invalid.
7. **Joint residual audit:** if a second candidate remains, add only its Schur marginal after Z
   and require its own billing break-even.  Never add Z, K, and H standalone values.

### Checkpoint-10 graph-ready split nodes

**Z22_ONE_STEP_IDENTITY** — status: solved theorem.  Terminal criterion: retain

\[
\chi^Z=3\langle ZW_2,W\odot(W\Lambda_{31}^\top)\rangle_F
\]

for explicitly credited one-step births only.

**Z22_COMPLETE_OLD_NOVELTY** — status: open.  Terminal criterion:

\[
\lVert(I-P_O)\chi^Z\rVert_M>0.
\]

Ambient dark-sector dimension is evidence of possible missing physics, not a pass.

**Z22_FULL_INTERFACE_MODULE** — status: solved characterization / bound-open.  Terminal
criterion: exact cyclic-module coverage, or an (\eta_Z) bound small enough for the robust
value gate.

**Z22_ONE_GEMM_COST** — status: conditional.  Pass iff transient (Z), reused (W_2), and
reused (B=W\Lambda_{31}^\top) make (C_Z\le1).  New (B) closes the 48/49 branch.

**Z22_CROSSED_K22** — status: generic completion closed-cost, dark/shared exception open.
Terminal criterion: zero Bellman pairing, an exact low-rank/shared reduction, or a charged
endpoint bound.  Generic rank at least 980 is not deployable.

**Z22_SCHUR_VALUE** — status: open.  Terminal criterion:

\[
(F_{Z\mid P}-\eta_Z)_+^2\ge2.860\times10^{-10}+E_{other}.
\]

**Z22_VS_STRICT_K31** — status: open swap gate.  Terminal criterion: full joint (Z/K) Gram,
one-slot feasibility, and the adjusted-score displacement inequality.  No source-typed
orthogonality or containment premise is accepted without proof.

**K31_STRICT_THIN_ACTION** — status: open exceptional competitor.  Terminal criterion:

\[
\sum_l2N_lR_l/n^2\le1
\]

(or the proved one-projection-reused half-cost), pre-cut factors available target-blind, and the
value-weighted action tail included in the Schur/robust budget.

**Z22_VS_SIGNED_H3** — status: open swap/stack gate.  Z displaces H only at the full
(4.656\times10^{-9}) residual threshold; after retained H it is tested as a q4 closer through
the (Z\mid H) Schur complement.

### Revised next question

Before measuring value, prove the two binary deployment premises: is (Z) genuinely available
from the deployed transient K22 interface, and is the exact (B=W\Lambda_{31}^\top) already
present?  If either answer is no, the one-GEMM route is closed.  If both are yes, the next
single quantity is the complete-arena, retained-portfolio residual amplitude

\[
F_{Z\mid P}=|\langle r,z_P\rangle_M|/\lVert z_P\rVert_M,
\]

with the crossed-K22/full-interface error subtracted.  It must exceed (1.69115\times10^{-5})
to enter the final q4 slot.

## Checkpoint 11: global minimum stack after ownership and cost closures

The four current candidates are:

- (A): self-consistent all-distinct correction through Taylor order two;
- (B): owner-correct post-AQ H3 replacement, exact or CountSketched;
- (C): thin strict-depth K31 action; and
- (D): transient Z22, admitted only as a one-step/incomplete full-interface candidate.

### Global retained-value master

Use the exact staged channel values, in units of (10^{-8}),

\[
V_v=1.7316,qquad V_3=0.4370,qquad V_4=0.10313,qquad m=0.07453.
\]

Let (a_v,a_3,a_4\in[0,1]) be the retained **conditional endpoint values** after the complete
old span and all previously selected sources have been removed.  With a separately bounded
cross/shift charge (E_X), the raw target is certified by

\[
\boxed{
1.7316(1-a_v)+0.4370(1-a_3)+0.10313(1-a_4)+E_X
\le0.07453.
}
\tag{G11.1}
\]

For a single typed provider (j), use

\[
\boxed{a_j=f_j\phi_jq_j^2,}
\tag{G11.2}
\]

where (f_j) is the fraction of the incumbent-residual channel oracle lying in the provider
span, (\phi_j) is frozen coefficient fidelity, and (q_j) is conditional endpoint
correlation.  Equation (G11.2) is not used to add overlapping providers; a multi-provider
channel uses its joint conditional Gram projector.

Solving (G11.1) gives the exact required quartic fraction

\[
\boxed{
a_{4,\min}
=1-\frac{0.07453-E_X-1.7316(1-a_v)-0.4370(1-a_3)}{0.10313},
}
\tag{G11.3}
\]

clipped to ([0,1]), and analogous formulas by permuting the channels.  Three consequences are
immediate under the certified channel allocation:

\[
\boxed{a_v\ge0.95696}
\]

even with exact H3 and q4;

\[
\boxed{a_3\ge0.82945}
\]

even with exact variance and q4; and

\[
\boxed{a_4\ge0.27732}
\]

even with exact variance and H3.  Therefore the minimum stack contains one variance-scale
provider, one H3-scale provider, and one quartic closer.  Among (A)--(D), the minimum form is

\[
\boxed{A+B+(C\ \text{or}\ D).}
\tag{G11.4}
\]

Candidate D may occupy the last slot only after its full-interface/error gate; until then the
certifiable default is (A+B+C).

### Exact replace-versus-stack and dominance algebra

For retained design (E), candidate columns (J), target (d), and endpoint metric (V),
define

\[
\Gamma(J\mid E)
=d^\top\left(P_{[E,J]}-P_E\right)^TV
\left(P_{[E,J]}-P_E\right)d.
\tag{G11.5}
\]

Equivalently this is (c^TG^+c) after residualizing (J) and (d) against (E).  If old
provider (Q) is retained and a new response (J) is added, the stack gain is

\[
\boxed{\Gamma_{stack}=\Gamma(J\mid E,Q).}
\tag{G11.6}
\]

If (Q) is removed, the replacement value is instead

\[
\boxed{
\Gamma_{replace}
=\Gamma(J\mid E)-\Gamma(Q\mid E).
}
\tag{G11.7}
\]

The two quantities cannot be interchanged.  Stack weakly dominates replacement in raw span but
may lose on billing.  With common base risk (R_E), utilization (u_E), and marginal cost
(C_j), candidate (i) adjusted-dominates (j) exactly when

\[
\boxed{
(R_E-\Gamma(i\mid E))(u_E+\kappa C_i)
\le
(R_E-\Gamma(j\mid E))(u_E+\kappa C_j).
}
\tag{G11.8}
\]

Every dominance statement below means (G11.8) plus the appropriate raw target and robustness
gate, never raw feature energy alone.

### A: self-consistent all-distinct order two

For homotopy source (s_l), the exact first and second tangents are

\[
e_{l+1}=J_le_l+s_l,
\]

\[
h_{l+1}=J_lh_l+D^2F_l[e_l,e_l]+2Ds_l[e_l].
\]

Adjoint elimination evaluates the scalar through order two without an H3 tensor.  The complete
assembled task provider must include producer as well as consumer motion:

\[
\boxed{
P^{(2)}=OS+O\,DS[e]+(DO[e])S,
}
\tag{G11.9}
\]

with the (\delta A)-on-(T) and (DS_2) terms included, and with synchronized K31, K4,
uncrossed-K22, and crossed-K22 reverse interfaces.

Let

\[
d_s=\sqrt{1-q_s^2},qquad
r_v=C_3d_s^3+\eta_A,qquad
q_v^2\ge1-r_v^2,qquad
a_v=\phi_Aq_v^2.
\]

Substitution into the global master gives the exact A-frontier

\[
\boxed{
1.7316\left[1-\phi_A+
\phi_A(C_3d_s^3+\eta_A)^2\right]
+R_{34}+E_X\le0.07453,
}
\tag{G11.10}
\]

where (R_{34}=0.4370(1-a_3)+0.10313(1-a_4)).  Equivalently,

\[
\boxed{
\eta_A\le
\sqrt{\frac{(0.07453-R_{34}-E_X)/1.7316-(1-\phi_A)}{\phi_A}}
-C_3d_s^3.
}
\tag{G11.11}
\]

The radicand and right side must be nonnegative.  Even at perfect (q_v), exact other channels,
and zero shift,

\[
\boxed{\phi_A\ge0.95696.}
\tag{G11.12}
\]

The task-weighted provenance residual is

\[
R_l^\tau=dK_{l,cert}^\tau
-S_l^\tau(\theta_D)-DS_l^\tau(\theta_D)[e],
\qquad
E_Q=\sum_{l,\tau}\langle Q_l^\tau,R_l^\tau\rangle.
\tag{G11.13}
\]

Normalize (E_Q) into (\eta_A).  The analytic source-curvature bound from the all-distinct
factor supplies the observable Taylor remainder; any hidden truth-only term remains separate.

For ten active layers the current dense cost branch is

\[
\boxed{
C_A=22.83+10
+20\mathbf1_{H\,new}
+20\mathbf1_{e/JVP\,new}\le45.
}
\tag{G11.14}
\]

Both reverse (H) and forward (e/JVP) reused gives (32.83) and passes; either one missing
gives (52.83) and fails.  Gate-Hessian and (DS[e]) work is elementwise/piggybacked.

**Earliest analytic kill for A:** the complete-quotient provenance receipt.  If a lower bound on
(|E_Q|) exceeds (G11.11), close before cost or signed-feedback work.  A coefficient fidelity
below (0.95696) is an independent universal kill.  No current equality receipt licenses the
goal-scale source yet.

### B: owner-correct post-AQ H3 replacement / CountSketch

The old D11 lineage owns the assembled H31 action.  The lawful new source is only

\[
\boxed{
\Delta Z_e=Z_e^{boundary}-Z_e^{old},
\qquad
f_\Delta=\mathcal A_{H31}[\Delta Z].
}
\tag{G11.15}
\]

Absolute (Z^{boundary}) or total H31 action receives zero new-source credit.  Owner subtraction
must occur in a synchronized gauge before approximate decoding whenever possible; otherwise the
error is amplified by

\[
\kappa_{owner}
=\frac{\lVert f_{boundary}\rVert_V+\lVert f_{old}\rVert_V}
{\lVert f_{boundary}-f_{old}\rVert_V}.
\]

For fixed CountSketch (S:\operatorname{Sym}^3(\mathbb R^{55})\to\mathbb R^k), the decoded
owner-correct action is

\[
\widehat f_{\Delta,S}
=\sum_e\langle S\Delta Z_e,SG_e\rangle.
\]

Let

\[
C_{\Delta,V}
=\frac{\sum_e\lVert\Delta Z_e\rVert^2\lVert G_e\rVert^2}
{\lVert(I-P_E)f_\Delta\rVert_V^2}.
\]

Under the seed-average orthogonal-noise model,

\[
q_B^2\ge\frac1{1+2C_{\Delta,V}/k}.
\tag{G11.16}
\]

For a required H3 retained fraction from (G11.1),

\[
a_{3,min}
=1-\frac{0.07453-E_X-1.7316(1-a_v)-0.10313(1-a_4)}{0.4370},
\]

the exact correlation/fidelity requirement is

\[
\boxed{
f_B\phi_Bq_B^2\ge a_{3,min}.
}
\tag{G11.17}
\]

Combining (G11.16)--(G11.17) yields the necessary sketch-size certificate

\[
\boxed{
k\ge
\frac{2C_{\Delta,V}\,a_{3,min}}
{f_B\phi_B-a_{3,min}},
\qquad f_B\phi_B>a_{3,min}.
}
\tag{G11.18}
\]

This is an expectation statement only.  For frozen seed and endpoint-weighted source/query
bases (U_X,U_Y), the exact cross-embedding error obeys

\[
\epsilon_{XY}(S)
=\lVert U_X^\top(S^\top S-I)U_Y\rVert_{op}
\ge\sigma_{k+1}(U_X^\top U_Y).
\tag{G11.19}
\]

The deterministic action error

\[
\eta_B
=\epsilon_{XY}\mathcal A_V
+\eta_{source}+\eta_{query}+\eta_{transport}
\]

must satisfy both the (3.80\times10^{-6}) action cap and the noisy-feature Schur radius for
the required conditional gain.

The actual (r=55,L=10,n=1024,p=29{,}260), base-48 additive census is branch-specific:

\[
C_B=10\left(\frac{2k}{1024}+m\frac{29{,}260}{1024^2}+g_{src}\right)
+C_{context}+C_{hook}\le1.
\tag{G11.20}
\]

- transport only gives (k\le51);
- one transient coordinate pass ((m=1)) gives (k\le36);
- source and query coordinate passes ((m=2)) give (k\le22);
- a generic one-GEMM source at every site, (g_{src}=1), is additive-cost closed.

No (SG_Z/G_Y/\Gamma) reuse receipt has been proved.  Consequently the relevant frozen-seed
rank kills are (\sigma_{37}) for the (k=36) branch and (\sigma_{23}) for (k=22).
At the earlier joint target (q_B\ge0.92845), (G11.16) requires

\[
C_{\Delta,V}\le k/12.49,
\]

namely (2.88) for (k=36) or (1.76) for (k=22).  These thresholds tighten when
(G11.17) requires higher correlation.

If B is stacked while old H31 remains, credit only

\[
\Gamma(\widehat f_{\Delta,S}\mid E,H31_{old}).
\]

If B replaces an old carrier such as D10/H31, use

\[
\boxed{
\Gamma_{B,replace}
=\Gamma(F_{new}\mid E)-\Gamma(F_{old}\mid E),
}
\tag{G11.21}
\]

and require it to clear the remaining raw gap.  Freed billing is not value credit.

**Earliest analytic kill for B:** owner correction and cross rank.  If (\Delta Z=0) in the
complete old quotient, close as duplicate.  Otherwise, if the allowed (k\le36) or (22)
has (\sigma_{k+1}) above the normalized error allowance, no frozen CountSketch can certify
the action.  The next kills are cancellation (G11.18), generic source cost, and replacement
value debit.  A structured order-(M^2) H31 source built from a new generic Jordan costate also
fails the one-slack branch: its mode-1 rank is at least (55), above the local (q\le33) gate,
unless the old H31 query is literally reused, a shared basis has rank at most 33, or replacement
billing is authorized.

### C: thin strict-depth K31

Candidate C contains only the true-minus-assembled (j<k) response.  Diagonal K31 remains old
ownership.  Let (f_C) be the fraction of the verified incumbent-residual quartic oracle lying
in the strict-depth row space.  Its retained quartic fraction is

\[
\boxed{a_{4,C}=f_C\phi_Cq_C^2.}
\tag{G11.22}
\]

It passes capacity exactly when

\[
\boxed{f_C\phi_Cq_C^2\ge a_{4,min},}
\tag{G11.23}
\]

with (a_{4,min}) from (G11.3).  The old (27.732\%) statement is valid only at exact A and B
and only if the verified strict-depth oracle denominator is the full (1.0313\times10^{-9})
quartic pool.

Suppose layer (l)'s pre-cut residual Bellman action has (N_l) atoms and rank (R_l).  The
thin factor route costs

\[
\boxed{
C_C^{thin}\le\sum_l\frac{2N_lR_l}{n^2}\le1,
}
\tag{G11.24}
\]

or half when one projection is already exposed.  Across ten full-width (N_l=n) layers this
requires average (R_l\le51), or at most (102) with one projection reused.  The discarded
action tail must be measured in the same endpoint metric and charged against (G11.23); this is
action compression, not proof of a rank-one H4 state.

The information premise remains

\[
\boxed{
\ker(D_{new}|_{U_<})\subseteq\ker\Phi_<.
}
\tag{G11.25}
\]

Repeated-index incumbent moments fail it by the positive parity witness.  Pre-cut factors must
therefore be genuine deployed information, not a post hoc factorization of a truth action.

There is now one exact target-blind pre-cut completeness construction, but it is not a one-unit
thin receipt.  In the scoped (r=32) half-AQ stack, the carried coefficient width is

\[
d=r+r(r+1)/2=560.
\]

It polarizes the complete external q4 action, including 211 and 1111 sectors, and costs
(d/n=0.546875) per layer plus (r^3/n^2=0.03125) local work.  A full K31 JVP with an
independently mixed direction/costate costs

\[
2d/n+2r^3/n^2=1.15625
\]

per layer, (16.1875) over 14 layers, or (8.53125) when the post-(W) state is already
carried.  This proves q-at-most-4 arithmetic completeness and can support a replacement branch;
it does not prove strict-depth novelty, fit the one-unit additive slot, or bound beyond-AQ error.
Synchronized reset and complete-old residualization remain mandatory.

If C replaces D10, its actual raw exchange is

\[
\boxed{
\Gamma(C\mid E)-\Gamma(D10\mid E)
\ge \text{remaining raw gap},
}
\tag{G11.26}
\]

not its standalone quartic gain.

**Earliest analytic kill for C:** missing fresh information or excessive pre-cut rank.  An
incumbent-only construction closes by (G11.25); a proved lower bound above (R=51) (or 102 in
the reused branch), or an endpoint-weighted tail that makes (G11.23) impossible, closes the thin
route before any fit.

### D: transient Z22 as an incomplete/full-interface candidate

The cheap one-step response is

\[
\chi_D
=3\langle ZW_2,W\odot(W\Lambda_{31}^\top)\rangle_F,
\]

followed by complete-old and retained-stack residualization.  For (m_l) already-exposed
contexts at layer (l), its narrow transient cost is

\[
\boxed{
C_D=\sum_l\frac{m_l}{n}+C_{context}+C_{hook}\le1.
}
\tag{G11.27}
\]

Ten scalar contexts cost only (10/1024) before context/hook charges.  A new full-width context,
later storage of (Z), or a full crossed-K22 consumer destroys this exception.

Let (O) be the complete old interface, (N=\ker O), and (F) the complete future operator.
The one-step context (A_D) is a complete provider iff

\[
\boxed{
\ker\begin{bmatrix}O\\A_D\end{bmatrix}\subseteq\ker F.
}
\tag{G11.28}
\]

Generic crossed-K22 diagonal-completion rank is at least 980, so full-interface deployment is
cost closed unless that branch is Bellman-dark, already shared, or exceptionally low rank.

#### Label-free analytic omission bound

For a dense (W), define

\[
r_W=\max_i\lVert W_{i,:}\rVert_2,
\qquad s_W=\lVert W\rVert_2,
\qquad \beta_W=r_W^2s_W^2.
\]

Let (X=W\odot W) contain the squared columns and let (Y) contain all pairwise column
Hadamard products.  Since

\[
YY^\top=(WW^\top)\odot(WW^\top),
\]

the PSD Schur bound gives

\[
\boxed{
\lVert X\rVert_2^2\le\lVert Y\rVert_2^2
\le r_W^2s_W^2=\beta_W.
}
\tag{G11.29}
\]

For omitted future K31, K4, and K22 costates
(\Lambda_{31}^{\perp},\lambda_4,\Lambda_{22}), the exact pullback masks give the sharp
observable Frobenius bound

\[
b_l^{exact}
=\lVert Z_l\rVert_F
\lVert M_{31,l}^{\perp}+M_{4,l}+M_{22,l}\rVert_F.
\]

Without forming those masks, (G11.29) yields the closed label-free bound

\[
\boxed{
b_l
\le3\beta_{W_l}\lVert Z_l\rVert_F
\left(
\lVert\Lambda_{31,l}^{\perp}\rVert_F
+\lVert\lambda_{4,l}\rVert_2
+\lVert\Lambda_{22,l}\rVert_F
\right).
}
\tag{G11.30}
\]

Here the K22 coefficient (3=1+2) covers both uncrossed and crossed quadratic forms.  If a
branch is included exactly, delete its term.  For corpus weights (w_\nu), coherent summation
over layers followed by the value norm gives

\[
\boxed{
B_D=left[
\sum_\nu w_\nu
\left(\sum_l b_{l\nu}\right)^2
\right]^{1/2}.
}
\tag{G11.31}
\]

Because (I-P_O) is an orthogonal contraction,

\[
\lVert(I-P_O)e_{omit}\rVert_V\le B_D.
\]

No old-span reduction may be claimed from norm data alone.

For exact residual feature (f), target residual (t),

\[
T=\lVert t\rVert_V,qquad s=\lVert f\rVert_V^2,qquad c=\langle t,f\rangle_V,
\]

the deployed noisy-feature gain is bounded by

\[
\Delta_D\ge
\frac{(|c|-TB_D)_+^2}{(\sqrt{s}+B_D)^2}.
\]

Thus for the final quartic floor (g_0=2.86\times10^{-10}), the exact coherent gate is

\[
\boxed{
B_D\le
\min\left\{
3.80\times10^{-6},
\frac{|c|-\sqrt{g_0s}}{T+\sqrt{g_0}}
\right\},
\qquad c^2/s>g_0.
}
\tag{G11.32}
\]

The (3.80\times10^{-6}) tolerance is already (22.5\%) of the minimum signal amplitude
(\sqrt{g_0}=1.69115\times10^{-5}).  If the ideal feature has no margin beyond (g_0), its
allowable omission radius is zero.

There is an exact analytic no-go when current observables do not bound the omitted costates or
hidden-source radius.  The full-interface dark-sector theorem supplies (h\in\ker A_D\cap\ker O)
with (Fh\ne0).  Scaling (th) leaves every observed one-step/old feature unchanged while the
omitted endpoint action grows as (|t|\).  Hence

\[
\boxed{
\sup\lVert e_{omit}\rVert_V=\infty
}
\tag{G11.33}

without a lawful radius/norm receipt.  In that case no label-free (3.80\times10^{-6}) or
(2.86\times10^{-10}) certificate exists.

**Earliest analytic kill for D:** failure of (G11.28) together with absence of the observable
norms needed in (G11.30)--(G11.32).  A nonzero unbounded full-interface kernel closes D as a
complete provider even though its narrow K31 witness is nonzero and cheap.  If the bounds exist,
D displaces C only when its robust conditional Gram gain and adjusted score satisfy (G11.8).

### Global minimum-stack decision tree

1. **Promote or kill A first.**  A is mandatory under the channel capacities.  Require the
   complete order-two provenance receipt (G11.13), (\phi_A\ge0.95696), a nonnegative
   (\eta_A) allowance in (G11.11), and one reuse branch of (G11.14).  If A fails, B--D cannot
   make up the variance-channel deficit and the current portfolio closes.
2. **Choose B implementation second.**  B is mandatory at H3 scale.  Prefer an exact
   owner-correct H31 replacement when its old query/context is reused and cost-valid; it raw-
   dominates a sketch of the same source.  Use CountSketch only when its smaller cost is needed
   and (G11.18)--(G11.20), frozen cross rank, owner cancellation, and noisy Schur value all pass.
   Keep old H31 and stack the difference only when the extra cost is preferable under (G11.8).
3. **Compute (a_{4,min}) from actual held A/B values.**  Do not reuse (0.27732) unless both
   preceding channels are exact.  If (a_{4,min}>1), kill before choosing C or D.
4. **Default quartic closer C.**  Use thin strict-depth K31 only if fresh pre-cut factors pass
   (G11.25), the rank/tail census (G11.24) fits, and (G11.23) clears the actual (a_{4,min}).
5. **D may displace C, not silently stack.**  First require transient/context cost, full-interface
   coverage or the finite bound (G11.32), and robust Schur gain.  Compare (A+B+C) against
   (A+B+D) by (G11.8).  If both fit and have nonzero joint residuals, stack only when
   (\Gamma(D\mid A,B,C)) exceeds its billing break-even and remaining raw gap.
6. **Final joint audit.**  Grade the selected columns through one complete conditional Gram and
   use (G11.1), not per-lane gain addition.  Then apply the frozen-network/source-error bounds
   and the adjusted-score inequality.

### Checkpoint-11 graph-ready split nodes

**GLOBAL_RAW_MSE_MASTER** — status: solved theorem.  Terminal criterion: equation (G11.1) with
joint conditional retained values and all cross/shift error charged once.

**GLOBAL_MINIMUM_THREE_CHANNELS** — status: solved under certified channel capacities.  A, B,
and one of C/D are all necessary; terminal minimum is (A+B+(C\text{ or }D)).

**GLOBAL_REPLACE_VS_STACK** — status: solved theorem.  Stack uses (G11.6); replacement uses
(G11.7); adjusted dominance uses (G11.8).

**A_ORDER2_PROVIDER** — status: algebra solved / provenance open.  Terminal criterion: complete
(P^{(2)}), including producer motion and all reverse interfaces, in a synchronized deployed
gauge.

**A_CORRELATION_FIDELITY** — status: open fail-fast.  Terminal criterion: (G11.10)--(G11.12);
(\phi_A<0.95696) is terminal even with exact other channels.

**A_PROVENANCE_ETA** — status: open, earliest A kill.  Terminal criterion: normalized
task-weighted (E_Q) plus hidden source is no larger than (G11.11).

**A_ORDER2_COST** — status: conditional.  Both H and e/JVP reused gives 32.83 pass; either
missing gives 52.83 fail against the 45 envelope.

**B_OWNER_CORRECT_SOURCE** — status: open.  Terminal criterion: form (\Delta Z) in one gauge
before decoding and prove nonzero complete-old residual; absolute H31 is duplicate.

**B_COUNTSKETCH_CORRELATION** — status: open.  Terminal criterion: (G11.17)--(G11.19), including
frozen-seed cross rank and owner-cancellation error.

**B_COUNTSKETCH_COST** — status: branch-conditional.  At current r55/L10, additive caps are
(k\le36) for one coordinate pass or (k\le22) for two; generic dense source is closed.

**B_REPLACE_VALUE** — status: open.  Terminal criterion: new conditional marginal minus the
removed H31/D10 marginal clears the raw gap and noisy-action tolerance.

**C_THIN_INFORMATION** — status: incumbent-only closed, fresh-factor branch open.  Terminal
criterion: (G11.25) with target-blind pre-cut factors.

**C_THIN_RANK_COST** — status: open.  Terminal criterion: (G11.24), including discarded-action
tail; average rank at most 51, or 102 with one projection reused.

**C_HALFAQ_PRECUT_PROVIDER** — status: exact arithmetic construction / novelty open.  Scoped
r32 cost is 16.1875 over 14 layers, or 8.53125 with post-(W) state reused.  It is a
replacement resource, not the one-unit thin branch; terminal value still uses (G11.23).

**C_QUARTIC_CAPACITY** — status: open.  Terminal criterion:
(f_C\phi_Cq_C^2\ge a_{4,min}) using the actual A/B held values.

**D_Z22_NARROW_WITNESS** — status: conditional cost pass / value open.  Terminal criterion:
(G11.27), complete-old novelty, and robust conditional Schur gain.

**D_Z22_FULL_INTERFACE** — status: incomplete until (G11.28) or a finite omission receipt.
Generic crossed-K22 rank 980 remains cost-closed.

**D_Z22_ANALYTIC_ETA** — status: bound derived / inputs open.  Terminal criterion:
(G11.30)--(G11.32).  If required Z/W/costate/radius observables are absent, (G11.33) closes the
complete-provider claim.

**C_VS_D_QUARTIC_SLOT** — status: open portfolio choice.  Terminal criterion: complete joint
Gram, robust raw value at least (a_{4,min}V_4), and adjusted dominance (G11.8).  Source labels
alone do not decide.

### Current next question

The first proof is not a CountSketch seed or a Z22 screen.  It is the complete order-two
task-weighted provenance receipt for A:

\[
E_Q=\sum_{l,\tau}\langle Q_l^\tau,
dK_{l,cert}^\tau-P_{l}^{(2),\tau}\rangle.
\]

If this cannot fit (G11.11), the only goal-scale source is dead and none of B--D can redeem the
portfolio.  If it passes, freeze the resulting (a_v), derive (a_{3,min}), and test the
owner-correct B cross module.  Only after held A/B values exist is it mathematically meaningful
to choose between thin K31 and incomplete Z22 for the quartic slot.

## Checkpoint 12: direct AQ Hermite-H4 score in the quartic portfolio

Call this candidate (E).  It is a contract-native latent Hermite-H4 action, not the external
same-site K31 action already owned by D11/Q24.

### Exact local score and complete-arena novelty

For

\[
g(Z)=c+b^\top Z+\frac12(Z^\top MZ-\operatorname{tr}M),
\qquad F(Z)=g(Z)_+,
\]

and symmetric fourth-order test (H), Gaussian integration by parts gives

\[
\boxed{
\mathcal A_H
=E\left[
\delta''(g)H[p,p,p,p]
+6\delta'(g)H[M,p,p]
+3\delta(g)H[M,M]
\right],
\quad p=b+MZ.
}
\tag{G12.1}
\]

This is exactly (\langle K_4(F),H\rangle).  The one-dimensional Fourier/resolvent formula in
the H4 lane evaluates (G12.1) without storing H6/H8/H10, subject to a certified nondegeneracy and
quadrature-tail receipt.

For each frozen net (\nu), combine the credited layers with endpoint weights (\beta_l):

\[
f_{E,\nu}^{raw}=\sum_{l=5}^{14}\beta_{l\nu}\,
\mathcal A_{H_{l\nu}}(c_{l\nu},b_{l\nu},M_{l\nu}).
\]

Let (O) contain the complete D11/Q24 CP4/star/pair/depth arena, and let (P) contain the
already-retained A/B portfolio sources.  With (V)-orthogonal projectors,

\[
f_E=(I-P_O)f_E^{raw},qquad
f_{E\perp P}=(I-P_P)f_E,qquad
t=(I-P_P)(I-P_O)d.
\]

The exact conditional Schur gain is

\[
\boxed{
\Delta_{E\mid P}
=\frac{\langle t,f_{E\perp P}\rangle_V^2}
{\lVert f_{E\perp P}\rVert_V^2},
}
\tag{G12.2}
\]

and is zero if the denominator is zero.  For several mixed-direction columns, replace (G12.2)
by (c_E^TG_E^+c_E).  A costate in the old CP4/star/pair range merely recomputes D11; only the
future-visible complete-arena residual in (G12.2) can receive raw credit.

In retained-fraction notation,

\[
\boxed{a_{4,E}=f_E^{span}\phi_Eq_E^2\ge a_{4,min},}
\tag{G12.3}
\]

where (a_{4,min}) is computed from the actual held A/B values by (G11.3).  Equation (G12.2),
not a raw local H4 norm, is the binding value definition.

### Combined AQ, quadrature, and cyclic-leakage radius

Let the true preactivation be (g_l+e_l).  ReLU Lipschitzness and the Hermite-H4 norm give the
local AQ-model bound

\[
\epsilon_{AQ,l}
=\sqrt{24}\,\lVert e_l\rVert_2\lVert H_l\rVert_{HS}.
\tag{G12.4}
\]

Let (\epsilon_{quad,l}) be a deterministic error receipt for the chosen finite Fourier nodes,
including regularization/tail, and let (\epsilon_{cyc,l}) bound the future-visible component
outside the retained cyclic/Jordan module.  The latter vanishes exactly only when

\[
\boxed{L_l^*\mathcal S_{l+1}\subseteq\mathcal S_l}
\tag{G12.5}
\]

at every transition and every local descendant lies in (\mathcal S_l).  A genuine OU/fresh-
cavity transition is a sufficient zero-leakage case; a deterministic dense correlated-ReLU
transition is not OU merely because its covariance matches.

The tight coherent corpus bound from these three declared errors is

\[
\boxed{
\eta_E=left[
\sum_\nu w_\nu
\left(
\sum_{l=5}^{14}|\beta_{l\nu}|
[\epsilon_{AQ,l\nu}+\epsilon_{quad,l\nu}+\epsilon_{cyc,l\nu}]
\right)^2
\right]^{1/2}.
}
\tag{G12.6}
\]

Complete-old and retained-stack projections cannot increase this norm.  Put

\[
T=\lVert t\rVert_V,qquad
s_E=\lVert f_{E\perp P}\rVert_V^2,qquad
c_E=\langle t,f_{E\perp P}\rangle_V.
\]

For any implemented feature within (\eta_E) of the exact feature, the sharp norm-only lower
bound is

\[
\boxed{
\Delta_{\widehat E\mid P}
\ge
\frac{(|c_E|-T\eta_E)_+^2}
{(\sqrt{s_E}+\eta_E)^2}.
}
\tag{G12.7}
\]

If the required quartic gain is

\[
g_E=a_{4,min}(1.0313\times10^{-9}),
\]

then the exact robust error gate is

\[
\boxed{
\eta_E\le
\min\left\{
3.80\times10^{-6},
\frac{|c_E|-\sqrt{g_Es_E}}{T+\sqrt{g_E}}
\right\},
\qquad c_E^2/s_E>g_E.
}
\tag{G12.8}
\]

At exact held A/B, (g_E=2.86\times10^{-10}).  AQ fidelity, quadrature error, and module
leakage are combined in (G12.6); they must not also be charged as independent losses in the raw
master.

### Factorization, mixed-rank, and quadrature cost

For (D) active layers, latent rank (r), mixed-direction rank (k), and (K) certified
Fourier nodes, the local factorized arithmetic is

\[
O(Dnr^3)+O(Dnkr^2)+O(DK nkr),
\]

with state (O(nr^2+kr)).  In square-GEMM equivalents the explicit screen is

\[
\boxed{
C_E
\simeq
\frac{D}{n^2}
\left[
r^3\mathbf1_{eig\,new}+kr^2+Kkr
\right]
+C_{context}+C_{hook}
\le1.
}
\tag{G12.9}
\]

Constant factors and the certified quadrature implementation must be included in the deployed
census; (G12.9) is an arithmetic lower screen, not a billing receipt.

Ignoring quadrature only to expose the rank boundary:

- at (n=1024,D=10,r=45), new eigensystems cost about (0.87), leaving (k\le6), while
  reused eigensystems permit (k\le51);
- at (r=55), new eigensystems already cost about (1.59>1), whereas reuse permits
  (k\le34).

With (K) nodes and eigensystem reuse, the actual mixed-rank gate tightens to

\[
\boxed{
k\le
\left\lfloor
\frac{n^2(1-C_{context}-C_{hook})}
{D(r^2+Kr)}
\right\rfloor.
}
\tag{G12.10}
\]

An unstructured (r^4) costate or a future cyclic module whose rank exceeds this bound is
cost-closed.  A local AQ variable rank does not bound cyclic rank: distinct correlated-ReLU
thresholds can generate linearly independent future pullbacks.

### Exact comparison with thin K31 and transient Z22

After the common retained A/B design, write (e,c,d) for AQ score, thin K31, and Z22 response
columns, all complete-old residualized.  For scalar C, the AQ marginal after C is

\[
\boxed{
\Delta_{E\mid C}
=\frac{(c_E-g_{EC}g_{CC}^+c_C)^2}
{g_{EE}-g_{EC}g_{CC}^+g_{CE}},
}
\tag{G12.11}
\]

and analogously for (E\mid D); block Gram matrices give the multicolumn form.  No chaos label
sets (g_{EC}) or (g_{ED}) to zero after nonlinear transport.

- **Versus thin K31 C.**  C costs
  (\sum_l2N_lR_l/n^2\le1), requiring average rank at most 51 across ten full-width layers, or
  102 when one projection is reused.  E with reused eigensystems has a comparable mixed-rank
  ceiling of 51 at r45 or 34 at r55 before quadrature.  C avoids AQ-model and Fourier errors but
  needs genuine pre-cut collision-free factors and a value-weighted action-tail receipt.  E
  acquires complete local Hermite-H4 directly but needs AQ fidelity and cyclic closure.  Neither
  algebraically dominates the other; choose by their robust conditional gains and costs.
- **Versus transient Z22 D.**  A reused narrow Z22 context can cost
  (\sum_lm_l/n\ll1), so D is arithmetically cheaper.  It is only a K31-context witness unless
  K4/crossed-K22/full-interface omissions are dark or bounded.  E removes the local trace-tail
  ambiguity and directly queries latent H4, but consumes the factor/rank/quadrature budget and
  may leak cyclically.  D wins only when its incomplete-interface robust gain clears the same
  (g_E); cheapness alone is not completeness.

For mutually exclusive use of the final one-unit slot, E adjusted-dominates (j\in\{C,D\})
exactly when

\[
\boxed{
(R_P-\Delta_{\widehat E\mid P})(u_P+\kappa C_E)
\le
(R_P-\Delta_{\widehat j\mid P})(u_P+\kappa C_j),
}
\tag{G12.12}
\]

with both gains replaced by their robust lower bounds.  If the other candidate remains, stack
only when its post-E Schur marginal clears both the remaining raw gap and its billing break-even.

### Earliest analytic kill and pass gates

Apply these in order:

1. **Complete-arena novelty:** if (s_E=0), the score is old CP4/star/pair/depth ownership and
   closes immediately.
2. **Cyclic closure/radius:** prove (G12.5) or a finite (\epsilon_{cyc}).  If current interfaces
   admit an unbounded future-visible kernel, no robust global certificate exists even though the
   local score is exact.
3. **AQ-law receipt:** bound every (\lVert e_l\rVert_2).  Without it, (G12.4) is unbounded and
   local AQ exactness says nothing about the physical source.
4. **Quadrature receipt:** certify nondegeneracy, node rule, and tail.  A finite (K) with no
   analytic tail bound is not an oracle.
5. **Factor/rank cost:** apply (G12.9)--(G12.10).  At r55, nonreused eigensystems are an immediate
   one-unit cost kill; reused systems still require mixed/cyclic rank at most the K-dependent
   bound.
6. **Robust Schur value:** require (G12.8), then compare C/D through (G12.11)--(G12.12).

A clean analytic pass exists under: nonzero complete-arena residual; a reused low-rank AQ
factorization; certified quadrature; exact OU/finite-module closure or bounded leakage; finite AQ
residual; and robust Schur gain at least (a_{4,min}V_4).  None is implied solely by the local
identity (G12.1).

### Checkpoint-12 graph-ready nodes

**AQ_H4_SCORE_IDENTITY** — status: solved local theorem.  Terminal formula: (G12.1) and its
contract-native Fourier reduction.

**AQ_H4_SCORE_OLD_RESIDUAL** — status: open.  Terminal criterion: (s_E>0) after complete
D11/Q24/arena and retained A/B residualization.

**AQ_H4_SCORE_SCHUR_GAIN** — status: open.  Terminal criterion: (G12.2)--(G12.3) using the actual
(a_{4,min}).

**AQ_H4_SCORE_ROBUST_ERROR** — status: formula solved / inputs open.  Terminal criterion:
(G12.6)--(G12.8), jointly charging AQ model, quadrature, and cyclic leakage.

**AQ_H4_SCORE_CYCLIC_CLOSURE** — status: generic correlated-ReLU branch open/no-go; OU or proved
finite module pass.  Terminal criterion: (G12.5) or a bounded (\epsilon_{cyc}).

**AQ_H4_SCORE_COST** — status: conditional.  Terminal criterion: (G12.9)--(G12.10) with reused
factorization, certified node count, context, and hook costs.

**AQ_H4_VS_THIN_K31** — status: open portfolio choice.  Terminal criterion: joint Gram
(G12.11), robust raw floor, and adjusted dominance (G12.12).

**AQ_H4_VS_Z22** — status: open portfolio choice.  E supplies a complete local H4 action but
has cyclic error; D is cheaper but full-interface incomplete.  Terminal criterion is the same
robust Gram/cost comparison, not source typing.

### Revised next question

The earliest useful receipt is whether the complete future quartic costate at L5--14 lies in a
reused mixed-direction/Jordan module of rank below (G12.10).  If not, the direct score is locally
exact but globally cost-closed.  If it does, the next receipts are the AQ residual norms and a
certified Fourier tail; only then should its complete-arena Schur value be compared with thin
K31 and Z22.
