# Current covariance-innovation replica audit completed

This measures the exact current331684 Gaussian-gain innovation target used by
checkpoint70/81, on rank35 layers8..14, using two existing independent8M
sample streams. It is distinct from the older oracle-state replica audit.
There is no new fit, sampling, estimator call or candidate.

|Channel|Pooled dispersion / observed|Stream cosine|
|---|---:|---:|
|covariance|0.068033424300|0.931967412570|
|diagonal|0.107753806874|0.892278973283|
|off_diagonal|0.067965363221|0.932035458304|
|response|0.108483609808|0.891539903521|

Dispersion is half the mean squared stream difference; observed energy is
the average of the two squared energies. The cross product plus dispersion
reproduces observed energy. Figures pool the seven equally weighted layer
energies before taking ratios. They are not averages of layer ratios.

|Fixed correction|Covariance residual / A|Covariance residual / B|Response residual / A|Response residual / B|
|---|---:|---:|---:|---:|
|output_only|0.982074148581|0.979831306455|0.996037144330|0.995199153108|
|nonlinear|0.966052377454|0.963730027171|0.990035893007|0.988933765979|

The original final16-output and528-parameter models are unchanged. Their
fourteen full current-state deltas were sealed before reference decoding in
this study and match independent FP64 formulas. The bank reconstruction is
byte-exact. These modest proxy gains also appear against streamB on this
training network; that does not establish native or cross-network benefit.
The nonlinear model's broader native panel remains failed at1.009582713004.
Its diagonal covariance residual worsens on both streams, despite improvement
in the full covariance measure. Complete per-layer results are in result.json.

The fixed decision threshold was at least20% pooled dispersion fraction in
covariance OR next response. Reaching it would support designing a separate
independent-label acquisition study; it never authorized an automatic new
stream or fit. Threshold reached: False.

This audit does not support sample noise as the main explanation for the
failed current fits. It supplies no reason to expand samples or rerun them.

These are two-stream point diagnostics, not a confidence-certified noise
ceiling. Both streams share the MPS/FP32 numerical implementation and its
unmeasured common bias. Rank35 is a TRAIN network; the parent and fitted
corrections have used streamA. Their dependence prevents an unbiased
fitted-arm cross-risk or held-out interpretation. No global sampling or
learning impossibility is asserted.

Both reference affine checks pass1e-4. Every original FP32 U byte and wide
covariance/response energy reproduces. The replica difference obeys the
independent linear-transport identity;16 fixed response columns per layer
and stream pass vector-matrix-vector checks. Synthetic indexed tangent and
noise/gain/partition identities pass, including zero and negative-cross cases.
An independent flattened-dot implementation and fsum verify all saved full
labels, both model losses, pooling and gain-gap identities within2e-10.

Four guarded groups retired; controller28123 terminal. Peak sampled RSS1231568896 B,self-report1256816640 B,zero added swap. Artifacts 176.307 MB before closeout; no new sampling,fit,native call,download or GPU. All five bank pins and checkpoint82 studies verified.

Official331684 remains adjusted3.5844144212297072e-9,raw1.949715441540434e-8,100/100 complete,zero failures.4e-9 banked;3/2/~1e-9 unmet. The full goal remains active; qualified uploads remain authorized.
