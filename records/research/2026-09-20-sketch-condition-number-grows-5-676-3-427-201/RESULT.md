# Current shared-sketch rounding diagnostic completed

Five rank38 cold replays reproduce all16 original output bytes and all full
bills after subtracting explicitly metered snapshots. Original base repeats
all saved trace arrays exactly. Three ownership/QR/Gram fixtures pass. All40
actual native QR results equal FP64 refactorization rounded toFP32 exactly.

Frozen-current-input FP64 sketch reconstruction differs from the native
sketch by3.91e-6..5.56e-6 relative on the base trajectory. Native condition
numbers grow from5,676 at layer5 to3,427,201 at layer14. Corresponding
relative Frobenius projector differences grow from0.000887 to0.212442.
At the first changed sketch (layer10), native centered projector response
norm is0.581818, versus0.000165167 for the independently reconstructed
same-input FP64 sketch. Their forward/center discrepancies are1.80812 and
0.0697395, respectively. Later FP64 references still inherit perturbed
native states; their remaining nonlinearity cannot isolate a root cause.

This establishes local sketch arithmetic sensitivity on this network and
these original head perturbations, not a causal end-to-end MSE improvement.
The diagnostic changes no estimator or parameters and opens no targets.
The previous canonical-order and original response experiments remain failed.
No fitting or submission is licensed by this diagnostic alone. A separate
full native sketch-precision intervention is needed to test response stability
and actual own cost. Merely declaring QR64 is not the proposed mechanism:
NumPy already widens QR internally, while the sketch producer isFP32.

Layers5..9 have zero cross-perturbation sketch response. In analysis.json the
field base_sketch_rounding_over_centered_response uses a1e-300 numerical
denominator floor; its large values at those layers are NOT finite response
ratios and must be interpreted as undefined. The response fields themselves
correctly use null for zero denominators. No scientific conclusion uses the
floored field.

Seven guarded groups retired;5 full calls,2.072718626895TF including
159,583,360 diagnostic copyF; peak sampled6.342GB, zero added swap.
About639MB artifacts; no new samples, downloads, GPU or training.
Official331684 stays3.5844144212297072e-9; remaining milestones unmet.
