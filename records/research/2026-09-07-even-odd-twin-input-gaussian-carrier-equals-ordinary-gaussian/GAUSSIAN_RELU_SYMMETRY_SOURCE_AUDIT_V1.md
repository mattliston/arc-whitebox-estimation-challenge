# Gaussian / bias-free ReLU symmetry audit: no new execution arm

September 6, 2026 PDT. Phase 2 only, width 1024 / depth 16. Source and
archived-prose audit; no estimator execution, numerical certificate, target
array decoding, fitting, submission, or change to an existing file.

## Verdict and strongest new negative result

No materially distinct exact CP-free mechanism with a credible whole-card
cost/raw advantage was established. This is a bounded negative audit, not a
lower bound against every possible algorithm or non-Gaussian closure.

In particular, a seemingly new **jointly Gaussian even/odd twin-input
carrier** is algebraically redundant: its single-branch mean/covariance
recursion is exactly ordinary Gaussian propagation. Keeping its extra
antithetic covariance cannot improve raw accuracy. The proof below does
not rely on a failed numerical experiment or on the old annealed-kernel
claims. No run is proposed for this Gaussian twin formulation.

## Archive exclusions actually checked

- `INPUT_RADIUS_RECOMBINATION_AUDIT_V1.md`: the uniform-sphere card with
  exact final chi-radius mean scaling already had 200 honest rows and a
  matched fixed fit. Existing rank35/100 raw final MSE ratios were
  .9920497853 / .9914425350 against their matched Gaussian parents and
  separate 1e9 references, not against the current board bank. Exact
  recombination includes mixed `Sym(mu,T3)` in fourth order. A conjugate
  change of moment coordinates supplies no new information.
- `RELU_COMPLEMENT_PROBE_V1.md` and `_CLOSED.md`: local positive/negative
  feature averaging with the frozen head was actually tested on 24 teacher
  trajectories, with held-out37/39/42/48 and layers10..15. Held-out error
  ratio 2.00647, 0/4 wins. It reused the positive compact mask and was not
  a fully reselected negative trajectory. This closes that implementation,
  not the exact complement identity or every symmetry-aware learned model.
- `NNGP_CENTERING_AUDIT_V1_CORRECTION.md` and
  `../public_quadrature_derivation_v1/{README,QUADRATURE_DERIVATION}.md`:
  the old repeatedly centered annealed kernel was wrong; raw-kernel
  composition is the corrected law. The recorded MC / affine-CV /
  antithetic / GH2 / lattice and subspace negatives remain measurements.
  Their prose claiming universal impossibility is not adopted here.
- `CP_SUFFIX_RANGE_PRODUCT_SOURCE_AUDIT_V1.md` and
  `D4_EXACT_CONTRACTION_AUDIT_2026_09_06_V1.md`: a supplied trajectory does
  permit shared terminal suffix products, but acquiring every intermediate
  plane/birth/covariance restores the full triangular work. Shared first
  transport of signed newborn atoms is already in the bank.

File mtimes checked on September6: input-radius audit 11:35:42 PDT;
complement closeout September5 08:50:56; NNGP correction September6
02:39:30; suffix audit September6 15:35:51. No handoff date was used as
evidence of execution time.

## 1. Exact input orthogonal symmetry is already quotiented by the bank

Use column-vector activations `z_l=W_l.T h_(l-1)`. For orthogonal Q,
replacing `W_0` by `Q W_0` leaves the entire network output law unchanged
under standard-normal input, because `Q.T X` has the same law as X.
Equivalently, the first preactivation law is fully determined by
`G=W_0.T W_0`, together with the unchanged later weights.

The current candidate is
`candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py`, SHA256
`3b7f55f9a814789743467bb4bcda0837c3c1aad15d120e4cdab632594bf54757`.
Its last `_predict_binary` definition starts at line1409; first covariance
is the Gram at line1441. Initial mean, cubic, trace, and radial state are
zero. `gnn_step` at lines602-604 returns zero correction and zero state
at layer0, even though feature construction is invoked. Subsequent
physical state therefore depends on W0 only through G in real arithmetic.

A Haar/input-sign ensemble consequently gives no distinct real-arithmetic
prediction of this bank. Materializing QW0 costs a dense product (plus
construction of Q); replacing W0 with a Gram factor costs a factorization
but does not reduce the Gram dimension or remove the later CP arena.
Changed FP32 Gram rounding is not extra statistical information or a
certified accuracy improvement. No payload values were used in this argument.

## 2. Absorbable hidden bases do not provide an exact dense rotation

An invertible linear A can be absorbed through a coordinatewise ReLU
without changing that activation law only if

    relu(Ax) = A relu(x)   for every x.

Putting x=e_j forces every entry of A nonnegative. If a row has positive
entries in distinct columns j,k, putting x=e_j-t e_k and increasing t
contradicts the identity. Each row has at most one nonzero; invertibility
then makes A a permutation times a strictly positive diagonal. Conversely
these positive monomial maps commute with ReLU.

This is an all-input architecture identity, not a claim that every
particular restricted reachable set lacks accidental simplifications.
One can store moments in any dense basis, but an arbitrary basis does not
turn the rotated activation into coordinatewise ReLU or lower tensor
dimension. Rotating back for each nonlinear read restores the basis
products; discarding tensor directions is a new approximation, not a
symmetry. Positive rescaling/permutation preserves dimensions and cannot
alone remove the signed-factor arena.

## 3. Twin-input Gaussian closure collapses to ordinary Gaussian closure

For a deterministic prefix H and symmetric input X, define

    e(X) = [H(X)+H(-X)]/2,   o(X) = [H(X)-H(-X)]/2.

Then e is even, o is odd, `E[o]=0`, and `Cov(e,o)=0` exactly. Let
`a=W.T e`, `b=W.T o`. The exact next parity fields are

    e' = [relu(a+b)+relu(a-b)]/2
       = [a+max(|a|,|b|)]/2,
    o' = [relu(a+b)-relu(a-b)]/2.

Suppose the retained state is only `(m_e,C_e,C_o)`, with the joint law of
(e,o) closed as Gaussian. Zero cross-covariance then makes the two Gaussian
fields independent. The plus preactivation is itself Gaussian with

    mu = W.T m_e,
    S = W.T (C_e+C_o) W.

Therefore the mean of `e'+o'=relu(a+b)` is the ordinary Gaussian ReLU mean,
and its covariance is the ordinary Gaussian pair map of S. If the twin
map is evaluated consistently, `C_e'+C_o'` is exactly that same covariance.
Induction gives identical single-branch means at every depth, including
the scored final layer. It remains true if the nonlinear Gaussian pair
map is approximated, provided the same approximation is used in the
recombined single branch; an inconsistency is not new information.

The additional cross-branch covariance is parameterized by
`T=W.T(C_e-C_o)W`. Computing both S and T conventionally takes four square
GEMMs per interior transition, rather than two for S alone, plus a second
Gaussian pair read. A simple full-width 16-layer schedule uses one initial
Gram, 4*14 interior products, and one terminal diagonal product: 58 square
GEMM equivalents, versus 30 for ordinary Gaussian propagation. At n=1024
this is 124,554,051,584 versus 64,424,509,440 conventional FP32 multiply/add
FLOPs before maps, copies, scalar operations, and rounding. Two covariance
matrices require 2*n^2 entries. These are algebraic skeletons, not actual
flopscope/Winograd bills or wall predictions. Eliminating T removes the
redundant work and recovers ordinary Gaussian propagation, not the bank's
raw accuracy at a lower cost.

What parity does NOT remove is the non-Gaussian mixed sector. In centered
notation the physical cubic tensor contains

    T_h(i,j,k) = T_eee(i,j,k)
      + E[e_i o_j o_k] + E[e_j o_i o_k] + E[e_k o_i o_j].

Odd numbers of o vanish; these even sectors generally do not. Already at
the first layer, for centered Gaussian Z with covariance G, e_i=|Z_i|/2
and o_i=Z_i/2 give

    E[(e_i-E[e_i]) o_j o_k]
      = sqrt(2/pi) * G_ij*G_ik / (8*sqrt(G_ii)),  G_ii>0.

The zero-variance case is zero. Independent Gaussian e/o deletes this
term. Retaining a new non-Gaussian coupling would be a distinct closure,
requiring its own lawful source, mixed consumers, and full cost; parity
does not supply those values from the two covariances.

## 4. Complement / adjacent-layer identities retain the nonlinear unknown

For every scalar Z, `E[relu(Z)] = E[Z]/2 + E[|Z|]/2`. Computing E[|Z|]
is the original non-Gaussian one-sided integral in another spelling.
At an interior layer `relu(z)=z+relu(-z)` creates a linear skip through
the next weight, not a removal of the remaining rectifier. Exact moment
propagation must include the skip/negative-branch cross moments.

Nor can a general layer pair be collapsed to one ReLU affine map:
`relu(relu(x)-relu(y))` has different kink pieces on x=0 and x=y and is
not one `relu(a*x+b*y)`. Enumerating a general first layer's orthants
retains high-dimensional Gaussian cone integrals; the known weights do
not provide their probabilities and conditional first moments for free.
Special all-nonnegative or otherwise certified linear cases can simplify,
but no such whole-card structure was established in this source-only audit.

Conclusion: preserve the current bank and all closed receipts. The exact
symmetries above do not justify another sign/radius/basis/sampling sweep,
nor do they establish a new CP-free candidate. A genuinely new raw lever
must add an economical non-Gaussian joint-state approximation or a new
exact integration mechanism, rather than only rename existing moments.
