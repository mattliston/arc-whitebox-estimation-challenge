# First-layer quadratic RQMC: one target-blind information experiment

Prospective source-only protocol, September 9, 2026. Root source review,
an explicitly free numerical/MPS slot, and a separate launch precede any
execution. No estimator, target, fitted component, old failure, or bank is
changed. This is not a native-cost or raw-MSE experiment.

## Fixed scope and law

Use only canonical TRAIN observer-fit ranks **35, 36, 38, 100**, in that
order, and sixteen independent randomized rules per rank, in order 0..15.
Each rule has 4096 directions and their exact negatives: **8192 actual
paths**, no dense rotation, whitening, fitted coefficient, or resampling.
The seed for rank r / rule j is `9309100000 + 100*r + j`. The scramble uses
NumPy PCG64DXSM(seed); its tail-bit generator uses PCG64DXSM(seed XOR
0xA17C9E52D643B80F). These are fixed pseudorandom streams, not a claim of
physical randomness. All 64 seed pairs are sealed before weight decoding.

Reuse the frozen Joe--Kuo 1024x32 direction table and complete LMS/digital
shift/Gray-code integer circuit from
`../rqmc_fullbudget_rank1024_cv/estimator.py`, source SHA
`6801eb3edecfd12b1e7f74e51b5987111594a6857885492f2934f68f40f37e92`.
The only sampler arithmetic replacement is its final clipped-normal
conversion. Append independent twenty tail bits **per point/dimension**:

    k = uint64(residue32)*2^20 + uint64(tail20)
    u = (float64(k) + 0.5)*2^-52

The endpoints are exactly 2^-53 and 1-2^-53, never 0 or 1. The leading32
net bits are preserved. Use SciPy ndtri in FP64 without PPF clipping;
normalize each row in FP64, and multiply by rbar=E[chi_1024]. Cast the
positive directions to FP32, then concatenate their exact FP32 negatives.
The reference also retains the corresponding FP64 directions and exact
negatives. No sample-dependent rotation or normalization of the bank is
allowed. This is a **finite midpoint-grid law**, not exactly continuous
Gaussian or exactly uniform continuous sphere sampling.

## Sole coefficient circuit and ideal identity

W0..W15 are the authenticated stored FP32 weights, with row-vector paths
`relu(H @ W)`. Compute FP64 first-layer Gaussian m0 and raw moment M0 from
the actual W0 widened, using its Gram and the exact zero-mean arccos ReLU
kernel. Retain pristine C0=M0-m0*m0.T. Zero columns have exact zero moments;
known diagonal moments are v/2. Correlations may be clipped only within
64 FP64 eps of the mathematical endpoints, with the amount/count recorded;
anything beyond that fails. This is a fixed endpoint-roundoff convention.

The single-Gaussian coefficient provider reuses the exact original
`marginal`, `gaussian_map`, and `factor_state` source functions from
`positive_gmm4_split_merge_prototype_v1.py`, SHA
`16cc38b25bcb6ff0796b5630eb538bb60ccaba166c97be944994ce905d317a58`.
No mixture/split/merge/power iteration is executed. Starting from exact C0,
retain the original FP32 factor after one FP64 Cholesky. For layers1..14,
use original FP32 factor-push and Gram, symmetrize after widening, and the
original positive Hermite12 covariance map and marginal gates. Retain its
FP32 factor. Layer15 uses one factor-push and the FP64 column-square sum,
not a final covariance map. Thus there are15 factorizations and29 FP32
provider matrix products. Failed factor/domain checks stop, with no retry,
jitter, eigenvalue repair, alternate provider, or selected layer cutoff.
An additional O(n) marginal call obtains each gate; it is included in cost.

Form and **store A32**, with fourteen FP32 matrix products:

    A = W1 diag(g1) W2 ... W14 diag(g14) W15.
    ms = m0 @ float64(A32)
    vs = diag(float64(A32).T @ C0 @ float64(A32))
    M2 = vs + ms^2
    t = mu_G15/sqrt(vs)
    beta = phi(t)/(2*sqrt(vs)); ell = Phi(t)-2*beta*ms.

Negative/nonfinite vs fails; no variance floor. Exact zero vs requires
ms=M2=0 and receives ell=beta=0. The final gate uses **vs**, not the
provider's different final variance. Provider approximations affect
control quality but do not alter the ideal analytic center.

For X=rbar*U, H0=relu(X W0), s=H0 A32, evaluate the final residual

    f(X) - ell*(s-ms) - beta*((1024/rbar^2)*s^2-M2).

The actual network and s products are FP32 MPS; their outputs are widened
for FP64 host control arithmetic and reductions. Earlier layers use their
FP64 sample means, except layer0 uses analytic m0. Archive the raw FP64
rule mean, the uncorrected mean, and served FP32 max(mean,0), plus separate
cast/clamp defects. Never empirically recenter the control.

In the ideal continuous law, Gaussian G=R*U has R independent of U,
E[R]=rbar and E[R^2]=1024. Homogeneity gives E[s(rbar U)]=ms and
E[s(rbar U)^2]=(rbar^2/1024)M2. The displayed control has zero mean and
the residual is the radial conditional expectation of the centered
Gaussian quadratic. This identity holds for any deterministic stored A.
It does not prove an RQMC variance improvement or finite-arithmetic bias.

## Numerical admission, observations and information statistic

Before full-width acquisition, verify rbar by Decimal Machin-pi plus
factorial and half-integer recurrence identities at80 and120 digits; all
four FP64 roundings must agree. The **algorithm**, not the older approximate
sampler literal, fixes the constant before any samples. Record both
high-precision values and the alternating-series truncation bound. Tiny
checks cover exact zero/identical/opposite first-layer columns, Gaussian
half-normal moments, and n=1/n=2 radial conditional centering. These are
algebra/arithmetic checks, not sampled truth or a fitted center.

For each rank, the first rule's fixed first8 positive directions and their
antipodes (16 paths) are compared with CPU64 reference paths through all16
layers and the actual stored-A control. Compare CPU64 using X64, CPU64
using widened X32, and the actual same MPS32 paths. The hard absolute
fidelity bound is1e-5 for every layer/path coordinate and for s/control/
residual; save arrays and max/RMS errors before applying gates. Input-cast
and product/reduction errors are separate. Only this small CPU reference
is run: no full CPU replicate, weight refit, or model replay sweep.

For the sixteen **served final rule means** y_j per network, report

    V = sum_j ||y_j-mean(y)||^2 / (15*1024).

Also report the unclamped and uncorrected variances, and eight independent
pair-difference energies `||y_(2j)-y_(2j+1)||^2/(2*1024)`. Their spread and
standard error are descriptive; no Gaussian confidence bound or1024
independent-neuron sample count is assumed. **Do not divide V by16 or8192.**
All16 intermediate-layer means and rule diagnostics are retained.

The fixed information signal is whether mean(V across4 nets) times the
planning multiplier is below3e-9. It is report-only: passed/completed means
the entire64-rule numerical/integrity/resource acquisition completed,
not competitive raw MSE, admission to training, or advancement. A common
finite-grid/PPF/FP32/kernel/serving bias is invisible to scramble variance.
The tiny and16-path checks do not bound its population expectation.

## Planning cost, resources and seals

One8192-path cold circuit has leading planning cost403,726,925,824 F:
8.589934592B FP64 Gram/center,30.064771072B A products,
62.277025792B provider products,10.737418240B factorizations, and
292.057776128B true-network/control products. A25B auxiliary allowance
(including2B sampler) gives428,726,925,824 F, multiplier F/2^41 about.194963.
This is **not a metered bill or verified whole-source upper bound**.
All sampler operations, factor/map guards, scaling, inputs/copies/casts,
all outputs/reductions, and setup must be paid in any later native circuit.
The information experiment amortizes coefficients over16 rules for offline
speed, but the one-rule production estimate pays its own full cold setup.

Run exactly one AP Python3.11 / NumPy1.26.4 / SciPy1.11.1 / Torch2.7.0
process, nice>=10, five BLAS environment variables1 and Torch intra/inter-op
threads1. **Never import threadpoolctl in AP.** Bytecode is disabled, MPS fallback0
and low-watermark0.1. No gradients. Set the MPS allocator fraction to
min(1,8e9/recommended). Host+owned-tree RSS<8e9 B, sampled driver<8e9 B,
zero added swap; whole600s and no retired rule for120s, including admission
and final sealing. These are prospective stop limits, not timing claims.
An independent stdlib supervisor uses the previously qualified absolute
deadline monitor: one shared process snapshot and concurrent swap query,
completed sample interval<=1s including checks/logging, no cached sample or
late retry. Explicit free-slot attestation and foreign-worker inventory
are mandatory. Retire only owned worker/observational groups on failure.

CLI: watch.py --run --mps-slot-clear --source-sha256 WATCH
--worker-sha256 PROTOTYPE --protocol-sha256 PROTOCOL. The child receives
--run plus source/watcher/protocol hashes. Exclusive output folders are
information_v1 and information_watch_v1; never resume or overwrite.
Bind all source owners, canonical manifest/row/weight hashes, fixed seed
roster, coefficient and output arrays, rule markers, result/completion
before/after. No target paths are opened. All attempts survive failure;
no later stage auto-runs. Only a separate same-configuration native
complete-bill, bias/raw and holdout/release experiment could advance.
