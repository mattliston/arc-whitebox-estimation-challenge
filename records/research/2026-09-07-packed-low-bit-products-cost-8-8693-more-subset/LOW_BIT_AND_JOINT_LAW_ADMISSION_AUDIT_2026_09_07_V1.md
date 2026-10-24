# Phase 2 admission audit: low-bit products and joint-law integration

Actual clock checked September 7, 2026 at 15:25:16 UTC / 08:25:16 PDT.
Source, metadata and exact-integer algebra only. No estimator execution,
target-array access, quantization experiment, fit, package or submission.
This records new source-price evidence, not a new raw or adjusted score.

## State and prospective decision

The previous goal turn was progress: it completed and qualified the FP32
pair-polynomial component. Official bank remains #329926 at
4.982231460183304e-9. Qualified independent-14 local component remains
4.788089831892398e-9, not an official score. Its complete record is
RETAINED_COMPONENT_PAIR_POLYNOMIAL_F32_V1.md. All completed checks and the
unsubmitted package remain intact; no rerun is needed.

At unchanged per-net raw and uniform cost scaling above the floor, reaching
4e-9 requires more than 16.459378573958805% cost reduction, equivalent to
77,016,052,025.08044 mean FLOPs from the present 467,915,916,017.2143.
This is planning arithmetic, not a claim that arbitrary mean raw and mean
cost can be multiplied to recover a panel score.

The two concrete directions examined before any substantial experiment:

1. **Cheaper CP contractions via low-bit multiplication.** A three-lane
   FP64 encoding nominally uses two-thirds the arithmetic of classical
   FP32 multiplication; a bit-pattern lookup alternative might appear
   cheaper still if gathers were incorrectly treated as unit-cost.
   Neither is a measured saving against the existing D4 implementation.
   Expected raw change is UNKNOWN: quantization can perturb signed-pair
   cancellations, inherited cumulants and frozen response-head features.
   Full cost must include ranges, quantization, packing/tables, unpacking,
   offsets, rescaling, accumulation, casts and copies. The cheapest decisive
   test is the installed-operation price and exact-mantissa audit below.
   **Selected and completed: the two specified circuits fail that screen.**
2. **Raw accuracy from an explicit joint characteristic-function supplier.**
   Current qualified rank35 oracle attribution gives next-mean response
   error ratios .3275057 for off-diagonal K3 and .0209139 for joint K3/K4.
   These are one-step source opportunities, NOT achievable final-score
   gains. A finer scalar readout has no supported raw magnitude; the causal
   joint-law constructor and its full feedback cost are missing. The
   cheapest source/equivalence test below finds no supplied constructor.
   **No numerical admission.**

The existing two agents independently reviewed these bounded questions.
No larger fit, precision sweep, old-family reopening or speculative model
run follows. The response-row exception remains completed, not open.

## 1. Three packed integer lanes: arithmetic is not cheap enough

Specify inner blocks of 32, unsigned A quantized to 8 bits, and unsigned B
to 4 bits. Pack three B columns with base beta=2^17. Each ordinary lane dot
is at most 32*255*15=122400<beta, and the packed nonnegative dot is below
2^51. This certifies integer arithmetic for the ordinary untransformed dot,
not quantization accuracy relative to the original floating operands.

Installed FNP.12 prices FP64 at twice FP32 and matmul by mn(2k-1).
For one m-by-32 by 32-by-n block, packed multiplication alone is

    2*m*ceil(n/3)*63.

Thus its divisible-dimension leading coefficient is 1.3125*m*k*n,
before every quantization/packing/unpacking/correction operation.

Concrete ARCHIVED same-shape comparison, not a current-network run:
bank_matmul_census_v1.json records the old P0/scalar-Q bank's
1024-by-1024 by 1024-by-4096 call at 5,180,428,288 FLOPs. Thirty-two
ordinary packed blocks cost at least 5,639,897,088 FLOPs: **8.8693% more**
for multiplication alone. The current response candidate retains the same
runtime source, but its current call geometry was not acquired here.
Do not transplant the old census's push fraction into today's full bill.

Adding D4 to the packed dots is not a free exactness assumption. A repeated
three-term Winograd transform has (3^4+1)/2=41 positive coefficients after
four levels. Valid unsigned inputs can give a two-term packed leaf dot

    2*255*41*15*41*(beta^2+beta+1)
      = 220928790304930050.

This lies above 2^57 and is 2 modulo 32, so it is not exactly representable
in FP64. Root verified this integer calculation with BigInt; no numerical
kernel was run. This invalidates transferring the ordinary-dot exactness
certificate to every D4 intermediate. It is NOT by itself a proof of a
wrong final decoded output: cancellation, controlled rounding or another
encoding would need a separately paid, qualified analysis.

The specified ordinary-dot construction is source-price rejected. No
claim that every low-bit/approximate multiplication algorithm is closed.

## 2. Five-bit, fourteen-input subset lookup: actual gathers cost four

The fixed alternative groups q=14 inner coordinates and uses b=5 bit
planes of B. For each block and output row, precompute the 2^q subset sums
of A. Each bit pattern selects one table entry per output, followed by
block summation and Horner combination. This is all intended to use FNP;
no host-numerical or unmetered data-dependent table construction is allowed.

The tempting 2b/q estimate is WRONG under installed official weights:

- take prices numel(output) with packaged weight **4.0**.
- take_along_axis has the same weight **4.0**.
- Advanced indexing explicitly charges 4*out_numel, plus boolean-mask
  scanning when applicable, under getitem weight 1.0.
- FP32 addition/reduction has rate/weight 1.0.

Therefore lookup and block accumulation alone have leading coefficient
5b/q=25/14=1.7857142857 times m*k*n. On the same archived
(m,k,n)=(1024,1024,4096), there are 74 inner blocks. A favorable lower
subtotal is

    [4*5*74 + 5*(74-1)]*1024*4096 = 7,738,490,880 FLOPs,

**49.3794% above** the archived D4 call, before tables, input ranges,
quantization, bit-pattern construction, Horner steps, scaling or copies.
More CP-push dominance cannot rescue an individually negative push saving.
There is no justification for a current trajectory recapture to price this
specified implementation. No bit-width/block-size sweep follows.

This is a source-priced lower subtotal for the specified circuit, NOT a
complete changed-estimator bill or a universal lookup-algorithm bound.
The failed source screen means no accuracy, runtime or score claim exists.

## 3. Raw-side identity: integration is not the missing constructor

For any real A with finite first moment,

    E[A_+] = E[A]/2 + (1/pi)*integral_0^infinity
             [1-Re(phi_A(t))]/t^2 dt.

For an actual affine receiver A=w^T h, phi_A(t)=phi_h(t*w). The unknown is
the joint characteristic function on those weight directions. Integrating
only the existing chosen moment closure more accurately does not supply it.
An O(n*Q) terminal quadrature subtotal excludes that constructor, all prior
layer work, covariance/K21/K22 feedback, and all sixteen response heads.

The active candidate's scalar map omits a kappa3-squared response, but no
test of that exact current-source splice was found. Do not call it a
universally or freshly tested numerical failure. The historical correlation
order extension changed unlearned raw by less than 0.1%, and higher-marginal
audits do not provide a strong current response-order gain. Existing incoming
joint-state errors are the better-supported accuracy target. No order or
amplitude sweep is admitted on novelty alone.

## 4. External joint-law method: useful scope test, not a new score route

[Adams et al., JMLR27 (2026), sections6-8](https://jmlr.org/papers/volume27/24-1199/24-1199.pdf)
uses signature quantization and Gaussian-mixture compression. With fixed
weights, Eq15c sends each atom c to delta_(W*ReLU(c)+b), with no new
parameter-induced covariance. Compression can nevertheless reconstruct a
Gaussian with the between-atom covariance. Gaussian-input initialization is
legitimate; the framework is not categorically inapplicable. Its VI/dropout
experiments do not establish our fixed-weight Phase2 accuracy or cost.

Our own information-theoretic scope check: for X~N(0,Sigma), Sigma SPD,
and any Y supported on at most N TOTAL atoms, entropy and maximum entropy
at a given distortion give

    log N >= I(X;Y)
          >= h(X) - (n/2)*log(2*pi*e*E||X-Y||^2/n),

hence W2^2 >= n*(det Sigma)^(1/n)*N^(-2/n). The accuracy agent independently
checked this derivation and its assumptions. It concerns full-state
Wasserstein approximation, NOT Gaussian-component count, arbitrary
post-ReLU laws, final-mean error, or a universal cubature lower bound.
A single atom at the exact final mean illustrates why no mean-error lower
bound follows. Thus the paper's distributional guarantee cannot substitute
for the missing mean-weighted performance evidence; do not force a more
expensive or stronger-than-needed distributional approximation experiment.

Primary technical searches yielded no new supported Phase2 scored method.
Phase1 write-up hits were not selected for new work. No forum post occurred.

## Provenance, limits and next decision

Principal installed source hashes, SHA256:

- flopscope/data/default_weights.json:
  f729bc47f07c080b8c7b9d7b045e5663f0af313669b4c2a962b5a9044a94cb7a
- flopscope/_array_ops.py:
  e45e927f18c54352503a4951d3d925a5f31f92ecea245e32360825d71a05da6c
- flopscope/_ndarray.py:
  fa5bbecb48b123ab77f31ced3da877d2e333aa5b403a324c33da591b967d3e74
- bank_matmul_census_v1.json:
  22479013b467f8e8c3fa8f4322915da5c2a9c352f9bbf78f1f4fc8823876e3a5
- census_bank_matmuls_v1.py:
  b97539b4f133b2c3d51c6e19f14f3d43240980723296a24c3eda1edef86638d5

The current candidate was rehashed unchanged:
0b84ab6fa96d32804c9adec350d41766e47d09a71a936b95652893bb1e364a96.
Installed whestbench.budget.PHASE2_ROUND also confirms lambda=0, C=F,
residual cap0.4s, wall cap120s and budget2^41. No Phase1 residual-time price
was added to a Phase2 score. Resource check found no Python/Whest numerical
process; swap remains784.75MiB of2048MiB. No MPS or numerical worker used.

The final goal remains strictly below1e-9, milestone banks below4/3/2e-9.
At present corrected raw, even the0.1 floor gives2.245315959e-9. Missing for
the next substantial raw experiment: an explicit causal joint-state source,
head-compatible consumed actions, favorable mean-weighted evidence and its
complete deployable cost. Missing for a major cost experiment: a genuinely
cheaper paid contraction circuit, not a unit-priced-gather assumption or
an unproved mantissa discount. No new substantial experiment is admitted.

All frozen work and old closeouts are preserved. No submission, forum post,
fit, commit, promotion, deletion or root-estimator edit occurred. The active
goal is neither achieved nor externally blocked; do not mark it complete
or blocked merely because these scientific routes failed admission.
