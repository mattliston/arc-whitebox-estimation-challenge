# Gaussian terminal skew v2: fixed scalar numerical repair

September8,2026. Preserve ALL gaussian_terminal_skew_v1 files and its failed
Gaussian acquisition42990ce7aba07f23ec92cecc23709452ae97dc6f54c28bb5329a25e2d4ee51bb.
The failure occurred before a complete prediction or any truth access: the
installed CDF computes .5*(1+erf), which loses the lower tail. At alpha=-9,
rounding CDF to0 while retaining positive pdf makes the direct r2 negative.
This is a source-identified numerical defect, not an adverse raw-score result.

The fixed model, both Gaussian/skew roles, B[:,i] ownership, all30/59 matrix
products, Gaussian6 polynomial, stored precision, birth/suffix equations,
output clipping and all score/resource gates are UNCHANGED. estimator.py
is mechanically identical to v1 except Gaussian scalar function replacement,
its import and module documentation; common.sources checks the exact inverse.
The pre-variance1e-10 floor is inherited unchanged. No negative moment is
clamped, rescaled or repaired. No fitted coefficient, parameter rescue or
new physical source is introduced. ROOT_DECISION.md from v1 remains pinned:
advance only if complete adjusted<4e-9 on BOTH35/100; no target code here.

## Public-FNP scalar implementation

stable_gaussian.py uses direct positive erfc rational coefficients from the
primary fdlibm source https://www.netlib.org/fdlibm/s_erf.c, preserving its
Sun Microsystems copyright/license. The coefficients are the primary source
literals, not an assertion that the installed copied coefficient file is
identical. Regions are the original0.84375,1.25,1/.35 boundaries in a/sqrt2.
The large-region exponent is split with an FP32-rounded high part. Returning
erfc directly avoids subtracting an already saturated erf. Values x>=28
underflow below FP64's subnormal range; this is not a moment floor. CDF is
q for alpha<0 and1-q for alpha>=0, with q=Phi(-abs(alpha)). Upper-tail q is
retained independently even where the positive CDF correctly rounds to1.

Let I_k(A)=integral_0^infinity t^k exp(-A*t-t*t/2)dt. Integration by parts
gives I_(k+1)=k I_(k-1)-A I_k and

    R_k=I_k/I_(k-1)=k/(A+R_(k+1)).

For the subset A>=5 ONLY, evaluate k64..1 backwards from R65=0 and form
r1=q R1,r2=r1 R2,r3=r2 R3. This is fixed64 continued-fraction evaluation,
not an adaptive order. Background: https://dlmf.nist.gov/7.9. Its finite
accuracy is qualified on the fixed scalar grid below; no global interval
certificate is claimed. Only nonempty tail subsets pay the64 loops.
For A<5, use r1=pdf-A*q,r2=q-A*r1,r3=2*r1-A*r2. This avoids using the
ill-conditioned large-A forward subtraction: direct r3 amplifies lower-tail
relative error at roughly O(A^6). The branch is numerical evaluation of the
same integrals, not a branch in the probabilistic model.

Negative-alpha mean/variance/k3 are r1,r2-r1^2,r3-3r1r2+2r1^3. For positive
alpha set loss2=2A*r1+r2, mean=A+r1, variance=1-loss2-r1^2, and
k3=3A^2*r1+3A*r2+r3-3r1*(1-loss2)+2r1^3. Scale by sigma,sigma²,sigma³.
These centered identities are algebraically equivalent to v1's raw moments;
they avoid cancellation of large positive-alpha raw cubes. The earlier local
reflected-marginal certificate is pinned as identity provenance, not borrowed
as this new FNP implementation's numerical qualification.

Every rational/poly/exp/CF operation, gather/scatter, cast, guard and constant
use is public FNP inside the participant budget. No private special-function
or host arithmetic is called on participant data. Helpers initialize no arrays
at import. The additional scalar work is measured, never presumed free.

## Fixed reference and scalar/tiny gates, no author execution

Grid exactly[-16,-12,-9,-8,-7,-6,-5,-2,0,2,5,9,16], unit input variance.
scalar_reference.py --execute runs only ordinary CPU SciPy adaptive quadrature
of POSITIVE lower-tail moment integrands and CPU64 math.erfc. It imports no
FNP estimator, Torch/MPS, network or truth. OFFPY has no SciPy, so this tiny
reference is a separate sequential process in root's SciPy environment.
Save its13 scalar rows, quadrature error estimates, source/version seals.
Both quadrature and libm are independent of the rational/continued-fraction
implementation. Fixed epsabs1e-100,epsrel2e-13,limit200; require each returned
quadrature error estimate <=1e-10 times its positive integral. No order/step search.

check_scalar.py --execute is OFFPY-only. Authenticate that sealed reference
and every source before reading its13 fixed scalars. Run the FNP helper under
BudgetContext at the grid. Require ALL finite, strictly positive means,
variances and third cumulants, lower-tail relative error<=1e-12, CDF absolute
error<=2e-14, and each mean/variance/k3 relative error<=1e-8. No floor conceals
tiny reference values. Record all per-point errors and actual scalar FLOPs.
These tolerances are fixed before any v2 numerical outcome.

Then check_small.py --execute requires that scalar PASS and delegates the
EXACT original n5/depth4/seed930915 explicit-tensor fixture. FP64 algebra
maxabs<=1e-11 and FP32 contraction BOTH absolute/relative<=2e-5, reference
norm reporting, Gaussian/skew prefix byte identity and matrix counts6/11
are unchanged. Its new outputs are in v2/small, not v1. All tiny stages use
nice10, BLAS1 environment,60s and<1GiB. No simultaneous numerical jobs.

Finally run_native.py --execute --role gaussian|skew delegates the frozen
official-input v1 harness with LOCAL v2 paths and both scalar/tiny prerequisite
seals. Each role35/100/35 runs in a fresh OFFPY process, initonce, exact repeat
bytes/F,120s and8,000,000,000B,130s call watchdog. Save attempt/bill before
resource assertion and public diagnostic reset; no reset in candidate/context.
Input copy billing remains separate and official FNP32 input ownership is
verified. Full returned16 means remain concrete FP32. Bind sources/inputs/
predictions before and after. Native residual is report-only. No truth/score,
fit, panel, package, server or board action is automatically authorized here.
