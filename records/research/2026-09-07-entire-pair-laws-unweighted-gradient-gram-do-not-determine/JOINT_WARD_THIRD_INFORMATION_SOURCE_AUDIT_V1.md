# Joint third-moment Ward identity: precise missing information

September 7, 2026 PDT. Phase 2 source/algebra work only. No network arrays,
labels, fitting, numerical experiment, meter, candidate or submission.
Root independently derived the identity and checked the reflection proof;
the accuracy agent independently supplied the same identity and an exact
bias-free ReLU witness. No old closed arm is reopened.

## Decision and relation to the score objective

This is an exact information-gap result, NOT a raw or adjusted gain.
It strengthens the specification for a new accuracy supplier: complete
pairwise statistics and even an unweighted gradient Gram do not determine
the missing joint third moment in this architecture class. It does not
claim that the current CP bank or an all-weights/history-aware estimator
has only those inputs, or that the witness is typical of challenge nets.

Confirmed official bank stays #329926 adjusted 4.982231460183304e-9.
The qualified response component stays independent-14 local adjusted
4.83424098592912e-9. At its retained raw 2.2454337457623213e-8, the .1
scoring floor still gives 2.2454337457623216e-9. Thus a strict sub-1e-9
method needs over 55.465% less raw error even at the floor. No expected
gain of that magnitude is inferred from the identity or old one-high oracle.

## 1. Exact identity, with the kink measure retained

Let X be standard Gaussian. The h_i(X) are degree-one homogeneous,
continuous piecewise-linear functions realized by a finite bias-free ReLU
network. Define m_i=E h_i, f_i=h_i-m_i, C_ij=E[f_i f_j], and
T_ijk=E[f_i f_j f_k]. Put

    Gamma_jk(x) = gradient(h_j)(x) dot gradient(h_k)(x),
    B_i;jk = integral f_j f_k * phi_n d(Delta h_i),
    G_i;jk = E[f_i * Gamma_jk].

Delta h_i is the DISTRIBUTIONAL Laplacian: a signed polyhedral facet
measure, not the zero almost-everywhere ordinary Hessian. The B pairing
is well-defined because f_j f_k is continuous and the finite network's
polyhedral pieces have Gaussian-integrable polynomial weights. No arbitrary
gate convention on a facet is needed for this pairing.

Euler homogeneity applied to the centered product yields

    E[X dot gradient(f_i f_j f_k)]
      = 3*T_ijk + m_i*C_jk + m_j*C_ik + m_k*C_ij.

Gaussian integration by parts and the distributional product rule yield

    3*T_ijk = sum_cyclic B_i;jk + 2*sum_cyclic G_i;jk
              - sum_cyclic m_i*C_jk.

Repeated indices are permitted; keep the displayed multiplicities. Dropping
the facet measure or replacing G by an unweighted Dirichlet Gram is not
this exact identity.

## 2. Literal ReLU witness: identical entire pair laws, different third

Take independent standard-normal X1,X2,X3, and set

    s = abs(X1)+abs(X2)+abs(X3),
    q_ab = (abs(Xa+Xb)-abs(Xa-Xb))/2
         = sign(Xa*Xb)*min(abs(Xa),abs(Xb)),
    h+ = (s+q12, s+q23, s+q31),
    h- = (s+q12, s+q23, s-q31).

All coordinates are nonnegative. Every entire pair distribution agrees:
pair12 is unchanged; reflecting X3 maps the plus pair13 to the minus
pair13; reflecting X1 does the same for pair23. These orthogonal
reflections preserve the Gaussian law and s. Hence all means, covariances,
repeated cumulant planes and arbitrary other pairwise statistics agree.

The unweighted Dirichlet Gram agrees as well. For each corresponding pair
there is one orthogonal reflection R mapping BOTH coordinate functions.
The two gradients transform by the same R, so their inner product is
preserved after changing variables in the Gaussian expectation. This
also covers each coordinate's diagonal gradient energy.

Writing z=s-E s, individual sign symmetry eliminates all terms except
the three-q product in the third-moment difference:

    T+_123 - T-_123 = 2*E[q12*q23*q31]
      = 2*E[min(abs(X1),abs(X2))
             *min(abs(X2),abs(X3))*min(abs(X3),abs(X1))] > 0.

The product is strictly positive almost surely. No simulation, approximate
quadrature or fitted coefficient is needed for the distinction.

The functions have a literal two-layer bias-free ReLU realization. The
first layer computes positive and negative parts of nine linear forms:
X1,X2,X3 and Xa+Xb,Xa-Xb for the three pairs. Linear combinations form the
three nonnegative outputs above, so the second ReLU acts identically.
Padding unused coordinates and continuing those outputs by identity embeds
the example into width1024/depth16. Its sparse/degenerate weights are an
architectural witness, NOT a sample from the benchmark's random-net panel.
There is no one global reflection equating the whole three-output law.

Since m and C agree, the identity in section1 implies that the symmetrized
combination of B and G must distinguish these cases. Exact pair integration
alone cannot manufacture that missing joint dependence.

## 3. Why this is not an affordable new carrier

For a linear transition a=W.T h, consistent recentering gives

    f_a = W.T f,  Delta a = W.T Delta h,
    Gamma_a = W.T Gamma W.

Consequently BOTH B and G transport with three affine legs. Their repeated
planes do not close that transport. After ReLU, the weighted gradient
object requires, among other quantities,

    E[(ReLU(a_i)-m_i+) * 1_(a_j>0)*1_(a_k>0)*Gamma_a,jk],

and the facet object needs the corresponding weighted boundary terms.
The unweighted E Gamma and the current pair distributions do not supply
these conditional quantities. A moving low-rank index basis similarly
needs a causal acquisition/update rule; its existence is not given here.

An unrestricted B,G pair has n^2*(n+1) entries after exploiting symmetry
of each tensor's last two indices: 8,598,323,200 bytes in FP64 at1024,
before weights, other state and workspace. A naive dense transport of
both tensors uses 12*n^4 = 13,194,139,533,312 conventional arithmetic
FLOPs per transition, before any FP64 metering rate, nonlinear integration
or consumers. These describe straightforward realizations, NOT lower
bounds on every structured algorithm or complete candidate bills.

Missing work also includes weighted-field acquisition, nonlinear gate/
facet integration, subtraction of already-carried sources, all consumed
actions, feedback and numerical/resource qualification. No complete cheap
configuration or supported raw-gain forecast is available. The old .3154
one-high local oracle ratio is capacity evidence, not a causal provider
or attainable final-error forecast.

## Next decision

The cheapest decisive SOURCE check was selected and completed: determine
whether exact pair laws plus an unweighted gradient Gram close the proposed
joint identity. The exact witness says they do not. No substantial
experiment is justified by this route yet. A future proposal must supply
the connected activation-gradient/facet dependence (or equivalent genuine
joint information) causally and affordably, with its own complete
raw-times-max(.1,F/2^41) advancement criterion. No larger Dirichlet fit,
pair-memory replay, oracle expansion or CP-family rescue is authorized.

The final goal remains strictly below1e-9 and milestone banks remain below
4e-9,3e-9,2e-9. This mathematical result changes the provider specification;
it is not score progress and leaves the retained small validated gain intact.
