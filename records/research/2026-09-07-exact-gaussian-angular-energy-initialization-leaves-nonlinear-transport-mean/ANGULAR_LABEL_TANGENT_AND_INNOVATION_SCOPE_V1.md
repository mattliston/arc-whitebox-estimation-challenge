# Fixed angular-energy label: exact Gaussian tangent, unresolved raw readout

Phase 2 only, September 7, 2026. Use filesystem mtime for completion.
Additive source/algebra record; preserve the initializer, all old closeouts,
frozen candidates and completed response-row exception. No new numerical
experiment, model/target-array access, fit, candidate, meter or submission.

## Decision and two remaining routes

Official historical bank #329926 remains adjusted 4.982231460183304e-9.
Qualified independent-14 LOCAL 0b84 remains adjusted 4.788089831892398e-9,
NCraw 2.245315959186444e-8, mean complete F 467915916017.2143. All qualified
components are already combined. Below 3e-9 is the immediate competitive
target; below 1e-9 remains the goal, with validated banks below 4/3/2e-9.

1. **CP-free two-memory replacement.** The existing explicit conditional
   source census is 278925711248 F, not a native complete-program bill.
   Under HYPOTHETICAL uniform cost scaling and unchanged per-net raw, this
   corresponds to 2.854190926539825e-9; a raw ratio below 1.0510859564804733
   would cross 3e-9 under those assumptions. This is planning arithmetic,
   not a forecast or product of separately averaged observed score columns.
   There is still no head/model payload, useful raw measurement, qualified
   domain-safe source map or explicit fourth-feedback ownership. The old
   variance-domain counterexample remains applicable to the declared map.
   The cheapest constructive-state/domain check has already been done;
   no new sufficient-state substitution or justified accuracy case was found.
2. **Joint angular-energy source.** The exact first-layer initializer is
   genuine joint information. This round derives a low-order Gaussian
   weighted-moment transport below. The specified terminal-mean-only dense
   FP32 transport plus FP64 initializer/vector SOURCE SUBTOTAL is
   71025262592 F before pair responses, safeguards, reference acquisition
   and readout. It is ADDITIVE; it replaces no current CP/head consumer.
   In an optimistic uniform-scaling illustration using just that subtotal,
   raw must improve over 13.1787% merely to break even, or over 45.6016%
   to cross 3e-9. Neither magnitude has supporting final-raw evidence.
   Complete estimator cost is unknown. The unresolved issues are the
   weighted nonlinear remainder and an actual mean-correction readout.
   The selected cheapest decisive test was algebraic closure/ownership,
   completed here; it does not admit a numerical fit or score run.

Neither route currently earns a substantial experiment. This is a scoped
admission decision, not a universal algorithmic bound or external blocker.
The exact-reuse review also found no new dominant product elimination;
its old generous transform opportunity (~3.14%) is not a milestone route.
All small QUALIFIED gains remain preserved. No old raw-only closeout was
reinterpreted as a measured complete-adjusted loss.

## 1. Fix the label and its centering

Let L be one centered scalar label, Y have mean m and centered z=Y-m.
Retain c=E[L z] and K=E[L z z.T]. For L0=epsilon_E0 from the frozen
Gaussian initializer, BOTH angular corrections are necessary:

    c0 = c_raw - eta0*m0
    K0 = K_raw - 2*eta0*C0.

The old initializer gives c_raw and explicitly gives the second angular
correction. This note supplies the corresponding first correction; do
not combine raw c with angular K. The frozen initializer is not edited.

Affine transport is exact: c'=W.T@c and K'=W.T@K@W. Recentring z by delta
leaves c unchanged and sends K to K-delta*c.T-c*delta.T, because E[L]=0.
There is no delta*delta.T term.

## 2. The quadratic-label closure is exactly a Gaussian moment tangent

For q=N(m,C), C positive definite, define the quadratic score

    ell(z) = (C^-1 c).T z
             + 0.5*(z.T C^-1 K C^-1 z - tr(C^-1 K)).

It has E_q ell=0, E_q ell*z=c and E_q ell*z*z.T=K. Gaussian integration
by parts gives

    E_q[ell*f] = c.T E_q[grad f] + 0.5*K:E_q[Hess f].

It is the directional derivative of N(m+t*c,C+t*K) at zero. Therefore
execution needs NO covariance inverse. This is a closure of the weighted
label law, not an assertion that actual later-layer Y is Gaussian.

For H_i=(Y_i)+, let M_i=E_q H_i, g_i=Phi(m_i/sigma_i),
d_i=phi(m_i/sigma_i)/sigma_i and k_i=K_ii. Then

    cplus_i = g_i*c_i + 0.5*d_i*k_i.

Define P_ij=E_q[I_i I_j], F_ij=E_q[I_i H_j], Q_ij=E_q[delta(Y_i)H_j].
The raw weighted second response and its centered version are

    U_ij = c_i*F_ij + c_j*F_ji
           + 0.5*k_i*Q_ij + 0.5*k_j*Q_ji + K_ij*P_ij
    Kplus_ij = U_ij - M_i*cplus_j - M_j*cplus_i.

There is NO cplus_i*cplus_j subtraction. On the diagonal this simplifies to

    Kplus_ii = 2*M_i*(1-g_i)*c_i + (g_i-M_i*d_i)*K_ii.

For i != j, put nu_j_i=m_j-C_ij*m_i/v_i and
omega_j_i^2=v_j-C_ij^2/v_i. The pair quantities are

    Q_ij = d_i * GaussianReLUMean(nu_j_i, omega_j_i^2)
    F_ij = m_j*P_ij
           + C_ij*d_i*Phi(nu_j_i/omega_j_i)
           + v_j*d_j*Phi(nu_i_j/omega_i_j).

P requires a bivariate Gaussian orthant probability; installed FNP has no
direct evaluator. The existing Gaussian6 polynomial is not that exact
kernel. Zero variances and perfect correlations need valid limiting cases;
an arbitrary indefinite bank covariance is not a Gaussian law. Reusing a
bank reference requires declared rounding, domain and centering ownership.

The first discarded Gaussian-Hermite label coefficients are

    Xi_ijk = E[L*z_i*z_j*z_k]
             - (c_i*C_jk+c_j*C_ik+c_k*C_ij)
    Xi_ijkl = E[L*z_i*z_j*z_k*z_l] - sum_6 K_ij*C_kl.

They vanish in the quadratic score model, not generally in the network.
For non-Gaussian Y, these expressions should not automatically be called
pure joint cumulants. Higher weighted terms also affect ReLU consumers.

## 3. Fixed-label transport is not a mean-error correction

L0=epsilon_E0 is not the current hidden-energy label epsilon_El. With
V_L=E[L0^2]>0 and TRUE current weighted moments, the scalar regression is

    a_l = (tr(K_l)+2*m_l.T*c_l)/V_L
    Cov(epsilon_El,z_l*z_l.T)
      = a_l*K_l + E[(epsilon_El-a_l*L0)*z_l*z_l.T].

Here E[L0*R2]=0. Orthogonality of a scalar regression does NOT remove the
last matrix. It vanishes at initialization, not automatically later.
Even the exact angular partial trace does not specify full fourth-tensor
consumers or their third-order feedback. Simply carrying c,K leaves the
unweighted reference mean unchanged. c15 is label/output covariance, not
an estimated error in that output mean.

## 4. Exact fresh-weight innovation identity, limited to its ensemble

The installed whestbench/generation.py draws standard normals, multiplies
by sqrt(2/n), then casts weights to float32. The following identities use
IDEAL real iid Gaussian weights and exact real arithmetic, not the rounded
generator. No numerical bound transferring them to that implementation is
claimed. Both root and the accuracy agent checked this distinction.

Fix a bias-free homogeneous prefix h and let N(x)=||h(x)||^2. After fresh
W_aj~N(0,2/n), let N'(x)=||ReLU(W.T h(x))||^2. Then

    E_W N'(x)=N(x),  Var_W N'(x)=5*N(x)^2/n
    Cov_W(N'(x),N'(y))=N(x)*N(y)*V(rho_H(x,y))/n
    V(r)=2*r^2+(2/pi)*[(1+2*r^2)*asin(r)+3*r*sqrt(1-r^2)].

V(0)=0, V(1)=5, V(-1)=-1. If either norm is zero, the covariance is zero
without evaluating its undefined correlation. The elementary squared-ReLU
pair formula is also the order-two arc-cosine kernel, Cho and Saul 2009,
[Eq7](https://proceedings.neurips.cc/paper_files/paper/2009/file/5751ec3e9a4feab575962e78e006250d-Paper.pdf).
The Hilbert-space application below is our derivation, not a score result
from that paper.

In L2(gamma_d), R2=||X||^2 and e=E_X N, homogeneity gives

    <N,R2>=(d+2)*e,  ||R2||^2=d*(d+2)
    epsilon=P_perp N=N-(e/d)*R2.

This projection also has zero Gaussian mean. It is a Hilbert-space
martingale across independent weight layers. Conditional on initial
weights F0,

    E_weights[<epsilon_l,L0> | F0] = ||L0||^2.

If initial weights are averaged too, the right side must also be averaged.
For xi=P_perp(N'-N), independent Gaussian inputs X,Y through the SAME
fixed prefix give the exact conditional increment energy

    E_W ||xi||^2 = (1/n)*[5*E_X N^2
        - ((d+2)/d)*E_XY N(X)*N(Y)*V(rho_H)].

Proof: ||xi||^2=E_X(N'-N)^2-((d+2)/d)*(E_X(N'-N))^2; apply the two
fresh-weight covariance identities. Fresh increments are orthogonal to
earlier states AFTER weight averaging, not for every realized network.
Positive increment energy therefore does NOT prove lost scalar-regression
capture: an increment can lie along L0. Its expected energy truly outside
that direction requires a further subtraction,

    E_W ||P_L0_perp xi||^2 = E_W ||xi||^2
       - E_XY[L0(X)*L0(Y)*N(X)*N(Y)*V(rho_H)]/(n*||L0||^2).

The remaining integrals are not supplied by the mean/covariance card.
No mean-field numerical approximation, architectural example, fixed-net
capture ratio or final-mean-error implication was computed from this
identity. It makes the missing information explicit; it does not establish
that the fixed label is useful or useless on the benchmark.

## 5. Cost and distinction from the executed matrix observer

At n1024, post0 initialization and transport through layers1..15:

| Source work | Charged FLOPs |
| --- | ---: |
| Full K transport, 30 ordinary FP32 products | 64393052160 |
| Terminal c15 only: 29 products, instead of the preceding row | 62246617088 |
| Fifteen FP64 c vector products | 62883840 |
| Given-G FP64 initializer, including leading pointwise work | 8715761664 + O(n) |

Terminal diagonal contraction, pair responses/CDF provider, guards, casts,
symmetrization, readout and reference acquisition are additional. FP64
transport doubles the product charge. A separate coherent Gaussian
trajectory adds its own covariance propagation. Signed K is not made
sparse by current bank masks. This is not a complete native bill, RSS
certificate or wall/residual-time guarantee.

Root read matrix_error_observer_v1.py and its design/closeout; both agents
independently checked source scope. The old circuit starts from zero,
injects learned sources only after layer0, and differentiates g_i*g_j*C_ij
off diagonal with an exact marginal diagonal tangent. It did NOT use this
initializer or the full bivariate transition. Its held-out raw ratio
0.9701022823 remains closed without complete-adjusted qualification.

If the old 14-feature source were enabled at a zero-mean Gaussian initial
layer, its normalized features would be constants; its generic initial
source lies in sigma_i*sigma_j*span{1,rho_ij,rho_ij^2}. The initializer's
nonlocal G^2/GD/DG contractions need not lie in that INITIAL span. This is
not a theorem about the old observer's whole final-output span or a new
nonlinear joint closure. Gaussian EP fixed-point and matrix-observer
closeouts remain intact.

## Preservation and next admissible step

No numerical run is admitted by these exact formulas. Do not repeat their
derivation/pricing as new score progress or continue this fixed-label arm
without a specified useful readout and weighted-remainder argument.
A genuinely distinct replacement requires sufficient-state/domain and
complete-cost ownership, followed by a prospective SAME-configuration
raw-times-scoring-multiplier gate and all numerical/resource/release checks.

Candidate SHA rechecked unchanged:
0b84ab6fa96d32804c9adec350d41766e47d09a71a936b95652893bb1e364a96.
Initializer SHA unchanged:
453f7a78efb80f47ffe9e9d463f6d7ca58bca8c6dcca2a35998c7fc2254d173f.
Installed generation.py SHA:
68bf70d7ac1d876a67044ae9b853fb718f14bce660f6e75b8dab55164d65188b.

No upload, forum post, commit, promotion, deletion, root-estimator or frozen
edit. Both existing agents performed bounded read-only algebra/source
reviews. No numerical/MPS worker or live experiment handle was created.
No new resource usage measurement is claimed. Goal remains active and
unachieved; mathematical uncertainty is not an external blocked condition.
