# Chi-radius fourth reshaping with current-frame trace ownership

September 6, 2026. Phase 2 only. Mathematical/tiny-algebra audit; no
targets, wide replay, fit, MPS, scalar sweep, or value experiment.

## Exact proposed closure

Use the constants and exact moment transformation from
INPUT_RADIUS_RECOMBINATION_AUDIT_V1.md. Input dimension is1024, regardless
of the small output dimension used for algebra certificates. Write
Gaussian-frame inputs as m,C,K, where K_ij=T^G_iij, and define

```
mu = m/c
B = C-d mu mu^T
S_ij = [K_ij-(c/n)(2 mu_i B_ij+mu_j B_ii)-beta mu_i^2 mu_j]/a3.
```

These are the angular mean, covariance and repeated third plane inferred
from current supplied moments. No accuracy or positivity is asserted for
an approximate bank state. Let v=diag(B), u=a4-c*a3, z=2*d/n, w=kappa4S.
The identifiable chi-radius fourth contribution is

```
Qrad = (2/n) Sym3(B,B) + R
R = u Sym4(mu,T_A) + z Sym6(mu,mu,B) + w mu^4.
```

The angular remainder is modeled as `(rA/3) Sym3(B,B)`. After Gaussian
recombination its coefficient is `lambda=a4*rA/3`.

## Which trace is preserved

Preserve exactly the CURRENT PRE-FRAME bank statistic
`tau = sum_ij K22_bank[i,j]`, not the previous-layer post trace or the
numeric radial parameter alone. For a symmetric C and its current bank
radial coefficient r, `tau=(r/3)[tr(C)^2+2||C||_F^2]`. At real F32 inputs
the already-produced bank K22 sum owns tau, including its arithmetic;
do not silently replace it by a differently symmetrized trace.

Let `D_B=tr(B)^2+2||B||_F^2`, `t_j=sum_i S_ij`. Then

```
tau_R = 4*u*mu.t + z*(2*||mu||^2*tr(B)+4*mu^T B mu) + w*||mu||^4
eta = (tau-tau_R)/D_B
lambda = eta-2/n
rA = 3*lambda/a4
Qnew = eta Sym3(B,B) + R.
```

IMPORTANT cancellation: the explicit `(2/n)Sym3(B,B)` radius term cancels
identically into the fitted angular scalar. It is not an extra fourth
signal after trace matching. The surviving proposal is trace-preserving
directional reshaping plus the fixed angular covariance geometry.
If D_B is zero or nonfinite, this formula is undefined; a deployable
method needs an explicit parent fallback, not a fitted denominator floor.
Negative rA is permitted algebraically, not clipped.

## Exact planes available in O(n^2)

```
Q22_ij = eta*(v_i*v_j+2*B_ij^2)
         +2*u*(mu_i*S_ji+mu_j*S_ij)
         +z*(mu_i^2*v_j+mu_j^2*v_i+4*mu_i*mu_j*B_ij)
         +w*mu_i^2*mu_j^2

Q31_ij = 3*eta*v_i*B_ij
         +u*(3*mu_i*S_ij+mu_j*S_ii)
         +3*z*(mu_i^2*B_ij+mu_i*mu_j*v_i)
         +w*mu_i^3*mu_j

q4_i = 3*eta*v_i^2 +4*u*mu_i*S_ii+6*z*mu_i^2*v_i+w*mu_i^4.
```

All diagonals must equal q4. The true supplied Q31, not the parent's
radial standardized `t4_i*rho_ij`, must enter the pair consumer.

## Fixed tiny certificate

Before any targets or wide use: build a fixed positive covariance B and
symmetric third tensor T_A in output dimension4. Set one fixed angular
scalar rA, construct the full dense Qnew by exact radius recombination,
derive Gaussian m,C,T and its trace-owned bank r, and reconstruct all
planes from m,C,K only. Require <=1e-11 absolute agreement, exact scalar
recovery to1e-11, and trace relative error<=1e-11. Also certify:

1. all plane diagonals and full dense tensor symmetry;
2. the cancellation of `(2/n)Sym3(B,B)` into lambda;
3. zero-mean case returns the original radial fourth model;
4. Gaussian input m=T=r=0 gives Qnew=0 and rA=-6/(n+2).

No scalar/radius sweep or randomly chosen network is part of this test.

## Archive scope

radial_noncentral.py was actually screened with beta=.25 and1, but its
model was a trace-normalized scalar `Sym(C+beta*m*m,C+beta*m*m)` and had
no third-tensor dependence. Its poor r35/100 receipts are preserved.
The anisotropic Sym(C,D) source/partial-trace carrier also ran and failed
(+1.09%/+6.01% unlearned raw); its supplied fourth tensor was a covariance
pairing, not Sym(mu,T). The original angular card/fit was executed but
used the old direct angular map, not this Gaussian-frame fourth reshape.
These records do not show this exact coefficient/plane intervention
already executed. They do preclude presenting the surrounding metric,
trace or angular families as unexplored.

## Full-carrier ownership is not O(n^2) for free

The new planes plus scalar are O(n^2), a few dozen vector outer products,
elementwise operations and reductions, with no dense weight push or
fourth tensor materialization. A conservative80*n^2 operations/layer
gives1.26e9 across15 full pair layers, <.3% of a490e9 bank bill before
remote/runtime costs. Scalar final use is O(n). This estimates ONLY the
plane constructor and its existing pair consumer, not a full carrier.

In Gaussian coordinates the new tensor includes
`p Sym4(m,T_G)`, with `p=(a4-c*a3)/(a3*c)`, plus covariance/mean-only
fourth shapes. The leading fourth-to-third all-distinct emission of
this term has TWO nonlocal parts. Put g=Gaussian gate, d0=phi/sigma,
h=g*m, and `M_ij=(g_i*d0_j*K_ji+g_j*d0_i*K_ij)/2`. The contribution is

```
p Sym3(h,M)
+ p*T_ijk*(d0_i*m_i*g_j*g_k
           +g_i*d0_j*m_j*g_k+g_i*g_j*d0_k*m_k).
```

This follows from the exact leading `1/2 sum_hubs g_i*g_j*d0_k*Q_ijkk`
source. The second term is the one-leg derivative of the ENTIRE carried
third tensor. Simply replacing g by g+p*d0*m adds unowned p^2/p^3 terms;
it is a resummation/approximation, not the exact displayed source.

An explicit mixed-CP representation introduces an additional pushed
factor leg and mixed plane products; after a subsequent dense weight
map that leg is no longer an output-wise scaling of the base factor.
For a generic existing rank-R CP bank, a straightforward exact step adds
one n-by-n times n-by-R factor push and two rank-R plane products, all
O(n^2 R), plus the independent vector-matrix trace contribution. Repeated
layers require ownership of those extra sources and their descendants.
No cheap complete schedule or permitted compression has been proved.

Therefore: these planes can support a clearly labeled one-step
information diagnostic, but cannot be called a complete coherent
fourth/cubic carrier at O(n^2) added cost. No value diagnostic or rollout
is proposed until root reviews this cost/ownership boundary.
