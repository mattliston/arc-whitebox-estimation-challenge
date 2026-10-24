# Full-budget sampling primary V1 preregistration

Status: `PREPARED_TARGET_FREE_TWO_ARM_FREEZE__NO_RELEASE_NO_SCORE`

This additive package freezes exactly two candidate classes before any target
opening:

1. `EstimatorScrambledSobolRaw`: six independent LMS+digital-shift Sobol
   replicates of sizes `32768,16384,8192,4096,2048,1024`, totaling 64,512;
2. `EstimatorIIDPairedWhitenedRadial`: 32,000 IID Gaussian base rows, their
   32,000 explicit negatives, exact sample whitening folded into the first
   weight, and the fixed leverage correction `0.9999921875305188`.

There is no rank, beta, ridge, clip, antithetic-Sobol, CP, response-control,
or whitening variant inside this frozen comparison.  The eventual decision is
the smaller mini100 final-layer raw MSE; ties within exact float64 comparison
select the paired-whitening arm because its meter has more memory headroom.
All 100 networks and both arms must complete before that metric may be opened.
There is no per-row, per-layer, or early-stopping selection.

## Exact scientific contracts

### Raw scrambled Sobol

The 1024x32 Joe--Kuo search-criterion-6 direction table is frozen as 131,072
little-endian bytes.  Each replicate independently draws a 32-bit digital
shift and a unit-diagonal random lower-triangular binary matrix.  The source's
Gray-code points and LMS+shift transform match SciPy's implementation on the
frozen reference fixture.

For the ideal continuous, unclipped, real-arithmetic randomized scramble,
each point is marginally uniform and the inverse-normal estimator is exactly
unbiased.  The executable clamps uniforms to `[2^-32,1-2^-32]` and casts the
normal values to float32.  It is therefore not claimed exactly unbiased for
the executed law.  If `a=Phi^-1(eps)`, the one-coordinate inverse-normal
perturbation has exact ideal expectation

`E|Z_clamp-Z| = 2*(eps*a + phi(a))`.

This local input bound is retained; no finite deep-network output-bias bound
is asserted.  There is no antithetic Sobol complement.

### IID paired whitening with leverage correction

Let `Z` be the 32000x1024 base Gaussian matrix and
`S=Z.T@Z/32000=L@L.T`.  The paired whitened bank is
`X=(+/-Z)@L^-T`; source orientation is exactly
`W1_eff=solve(L.T,W1)`.  Thus its ideal empirical covariance is identity.
For a Stiefel row, squared leverage is
`Beta(1024/2,(32000-1024)/2)`, independent of its uniform direction.  Degree-1
homogeneity of every biasless ReLU layer gives

`c*mean f(X)`, where `c=E[chi_32000]/sqrt(32000)`.

The 80-digit reference is
`0.9999921875305187702081151692378470446557362738627861719618318620646028733033937`;
its frozen float64 materialization is `0.9999921875305188`.  This proves exact
ideal-law unbiasedness despite whitening and pairing.  No ridge or centering
is used.  Executed float32 generation, Cholesky, solve, and output conversion
are qualified as numerical rounding rather than exact real arithmetic.

## Frozen resource gate

The official dataset-less whestbench/flopscope 0.12 meter produced:

| arm | paths | FLOPs | `2^41` margin | wall/MLP | residual wall | peak descendant RSS |
|---|---:|---:|---:|---:|---:|---:|
| raw Sobol | 64,512 | 2,183,394,289,696 | 15,628,965,856 | 6.706836 s | 0.134858 s | 7,634,501,632 B |
| paired whitening | 64,000 | 2,187,012,965,375 | 12,010,290,177 | 1.921763 s | 0.062490 s | 3,123,445,760 B |

Both are strictly below `2^41=2,199,023,255,552`, return finite float32
`(16,1024)` arrays, and passed with no estimator failure.  Meter truth used
only an 8-sample deterministic synthetic fixture and is explicitly unscored.

## Future seal and score boundary

No file in this package authorizes dataset iteration, prediction sealing,
target decoding, fitting, scoring, selection, submission, or graph mutation.
A future prediction sealer requires an independent exact-byte PASS and a
separate root prediction-only release.  A later joint scorer requires another
independent PASS and one root target-opening release.  Predictions for both
arms must be sealed before any target value is decoded.

