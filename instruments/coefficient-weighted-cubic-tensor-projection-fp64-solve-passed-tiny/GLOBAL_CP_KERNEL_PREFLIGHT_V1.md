# Global coefficient-weighted CP compression: prospective preflight

September 7, 2026 PDT. Phase 2 only. New source/algebra arm; no frozen
candidate, trained head, prior receipt or closed family is changed.

## Two routes and selection

1. **Joint coefficient-weighted CP dictionary.** Unlike the closed unweighted
   thinning, pair coalescence, coordinate projection and 512-scalar-query
   lookahead fits, this projects the complete all-distinct cubic tensor onto
   selected existing atoms. It restores every current repeated-index entry
   through the existing coordinate-star representation. Supporting evidence
   is an exact objective and a paid static cost margin, NOT raw accuracy.
   One structural example compresses old cohorts 0..6 after layer 7 to
   K=1024 atoms, leaving newborn 7 and its shared first push intact.

   Old P0/Q development geometries, ranks 35 and 100 respectively, give:

   | Quantity, FLOPs | rank 35 | rank 100 |
   |---|---:|---:|
   | Future CP product saving |178621361808|179942078968|
   | Three ordinary FP32 cross-Grams |73594109952|76588646400|
   | Both D4 repeated-plane restoration reads |14000145152|15065175316|
   | Supported FP64 LU solve, K=1024 |1435850068|1435850068|
   | Conservative assembly allowance |879854592|890275840|
   | Remaining modeled saving |88711402044|85962131344|
   | Fraction of OLD complete bill |18.08056%|17.55928%|

   Assembly allowance is 16*a*(M+K)+32*K*M+64*K^2+32*n*K+128*n^2+16*(M+K).
   No normalization is assumed. No exact M-by-M tensor-norm acquisition is
   included or free. Full FP64 Gram/restoration acquisition erases the
   benefit. The tiny qualification uses ordinary FP32 products, not D4
   acquisition. The old complete bills and masks are not current-response
   complete bills. Storage, persistent wall, coefficient conditioning and
   changed-mask effects are still unqualified.

   If the percentages transferred to the retained 4.83424098592912e-9
   panel at unchanged raw, adjusted would be about 3.96--3.99e-9. This is
   a conditional projection, not an expected measured score; only about
   0.4--1.0% raw worsening would then be tolerable for below 4e-9.
   The unresolved assumption is that selected atoms span enough of the
   current all-distinct tensor after signed cancellation, with a stable
   affordable solve and faithful current-star restoration. K >= n gives no
   guarantee: the feature space has dimension binomial(a,3).

2. **Pure Gaussian four-quadratic cycle source.** For
   q_i=d_i*(Z_i^2-C_ii)/2, its output fourth cumulant is
   3*trace((C*diag(d*w))^4). This differs from the closed two-quadratic /
   two-linear R1 source. The old block-feed test enumerated paths with a
   doubled edge, not four-cycles. Synthetic 4D receipts assigned 4.7%/8.7%
   of fourth cumulant to cycles, but do not quantify width-1024 value.
   Existing final-layer true-k4-error attribution captures only 4.6%/2.7%
   linearly; that is neither a broad ceiling nor support for a large gain.
   Two batched products per Gaussian trace probe cost 4.295B classical,
   8.590B FP64 billed, plus a valid PSD factor, draws, overlap subtraction,
   consumers and multi-layer third/covariance feedback. Scalar subtraction
   does not reduce probe variance; a payable matrix control is missing.
   Complete estimator cost and expected raw gain therefore remain unknown.
   Cheapest decisive evidence would jointly establish overlap-subtracted
   consumer alignment and controlled probe variance on a faithful state.
   No receipt supplies either. Do not force a network run on this route.

**Select route 1 only for a tiny synthetic algebra/API qualification.**
No Phase 2 network arrays, targets, checkpoint, fitting or bank execution
are admitted by this preflight. No candidate or milestone is claimed.

## Fixed tiny test, before execution

Script: qualify_global_cp_kernel_tiny_v1.py. Two fixed fixtures, n=8,
a=6 active coordinates, M=12 old atoms, K=6 selected atoms; seed 719.
One orthogonal distinct-feature dictionary and one correlated signed-pair
dictionary. Selection is the first six atoms, fixed by fixture definition.
This is not a proposed data-selection policy for width 1024.

For p,q define s_r=(p^r)^T(q^r). The kernel
k_D=s_1^3-3*s_1*s_2+2*s_3 equals the ordered all-distinct tensor inner
product. Three K-by-M FP32 products supply the FP64 polynomial, extracted
K-by-K Gram and signed RHS. Symmetrize once and call the supported FP64
fnp.linalg.solve once. No ridge, tuning, retry, alternative decomposition
or normalization. An unsupported/singular call fails this tiny test; it
does not license a silent fallback. Stored replacement weights are FP32.

Let R=plane(old)-plane(replacement), evaluated from actual stored factors
and weights. The additive star update is 3*R.T-2*diag(diag(R)). This must
restore all repeated entries; the residual tensor then has all indices
distinct. Check independently with explicit tiny tensors and five fixed
linear queries. Incoming nonzero star is included. No learned head used.

Prospective gates, both fixtures:

- Kernel versus explicit distinct tensor features: relative error <=1e-5.
- Native FP32 three-product charge equals 3*(2*a*K*M-K*M) exactly.
- Native FP64 solve charge equals 2*(2*K^3//3+2*K^2) exactly.
- Solved coefficients versus FP64 explicit-feature solve: relative <=1e-3;
  native FP64 normal-equation residual <=1e-10.
- Actual FP32-restored repeated plane error, divided by full old tensor
  norm, <=3e-6; five-query error matches all-distinct residual <=1e-5
  relative to full old contracted tensor norm.
- Finite outputs, process nice >=10, all three BLAS env controls exactly1.

Record all failures; no outcome-dependent fixture or tolerance edits.
Use the native FNP.12 environment, one nice10 CPU worker, no MPS, budget
2^28 per fixture. Synthetic NumPy reference work is separate, not a legal
estimator or a claimed complete deployment bill. A pass qualifies these
identities and APIs only. It cannot admit a candidate or panel without a
separate prospective current-state accuracy/cost protocol.

## Advancement objective for any later score-bearing arm

Actual complete per-net raw * max(0.1,F/2^41), averaged on the same nets,
is decisive. A raw increase is acceptable when its paid cost outweighs it.
Preserve any separately validated small gain as a component; below4e-9,
below3e-9 and below2e-9 are milestone banks, final goal strictly below1e-9.
The frozen response-row exception is complete and remains no-tune/no-refit.
Bank #329926 and all old receipts remain untouched. No forum posts.
