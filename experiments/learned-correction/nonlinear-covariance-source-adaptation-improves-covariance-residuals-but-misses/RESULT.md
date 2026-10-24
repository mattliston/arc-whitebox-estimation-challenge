# Current nonlinear edge adaptation: NEGATIVE: fixed information gate failed

Both fits used the exact current official331684 frozen states, the same
coherent8M Gaussian-gain covariance-innovation labels, six TRAIN networks
and seven layers per network. VALID400/500 were excluded from both fits;
they have earlier research history and are not pristine holdouts. Forty
epochs and1680 updates completed for each arm, using identical cell orders,
optimizer, schedule, normalization and regularizer. Only final parameters
were evaluated; no validation checkpoint selection or fit extension.

The matched control trained16 output weights. The nonlinear arm trained
all528 parameters in the existing16-channel edge source. Input features,
normalizations, envelope and deployment architecture are unchanged. This
explicitly tests the encoder surface excluded by the earlier ridge study;
that study's source, failed gate and result remain unchanged.

|VALID rank|Arm|Covariance residual / baseline|Next-response residual / baseline|
|---|---|---:|---:|
|400|output_only|0.995642554374|0.986418747507|
|400|nonlinear|0.979719722266|0.981543255247|
|500|output_only|0.984627969438|0.999227123010|
|500|nonlinear|0.970613741842|0.991767418498|

Each nonlinear held network requires covariance<=.98, response<=.95, and
response<=.98 of the matched output-only control.

- Rank400: response/control=0.995057380780; passed=False.
- Rank500: response/control=0.992534525594; passed=False.

The proxy labels subtract the same declared Gaussian-gain transport of
previous mean/covariance errors. They are not exact full-program innovations
or derivatives, and retain coherent8M sampling/numerical error. The reverse
checks establish only the smooth frozen-input proxy's derivative, not a
native gradient or an autonomous improvement. No1B final-mean reference,
native candidate, complete bill, runtime or official score is measured.

All528 derivative coordinates pass independent scalar complex-step checks
on three fixed small fixtures, together with independent forward, symmetry,
permutation, null, directional-descent and FP32 checks. All56 wide bank
source reconstructions pass3e-6 before each network's targets are decoded.
Starting deltas are byte-zero. The copied innovation-transport function is
byte-identical to the old source, and all old covariance/response energies
reproduce within2e-10. FP32 target/response roundoff passes3e-6.
Both final models were sealed before VALID prediction. All28 held layer
deltas were sealed before scoring and pass an independent explicit-channel
FP64 reference. Direct and expanded quadratic losses agree within2e-10.

The first closeout verifier stopped on exact pooled-ratio equality: Python3.13
uses a different floating-point sum from the Python3.11 scoring runtime.
The preserved diagnostic shows differences of0--4 rounding units. Additive
finalize_v2 reproduces the producer's sequential sums exactly and independently
checks high-precision fsum within the existing2e-10 arithmetic tolerance.
No data, source model, fit, prediction, loss or scientific gate changed.
The original verifier, diagnostic and complete source inverse are retained.

All13 guarded groups retired; controller46086 terminal. Peak sampled RSS
1423769600 B, self-report
1423769600 B,zero added swap. Artifacts before
closeout total1296.225 MB. No new estimator calls, samples,
downloads,GPU or ordinary1B targets. Existing captured states and references
were reused, with all old and official source pins retained.

Official331684 stays adjusted3.5844144212297072e-9,raw1.949715441540434e-8,
100/100 complete,zero failures.4e-9 is banked;3/2/~1e-9 remain unmet.
The full goal remains active; qualified uploads remain authorized.

This fixed adaptation stops here. No alternate epoch, optimizer, penalty,
layer, label, feature or roster rescue, native candidate, panel or upload
follows this failed information test.
