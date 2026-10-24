# Explicit CPU ownership and independent interpretation

2026-09-09UTC, additive clarification; original RESULT.md unchanged.
CB's CPU arm is the native-G0-injected NG CPU model through layer1, not
the unmodified CO CPU trajectory used by the original whole-card BR test.
The exact result therefore diagnoses N-versus-native-G0-CPU at layer1.
It must not be attributed to BR's all16-layer N-versus-original-CO response.

fresh_accuracy and fresh_efficiency independently verified the result and
completion seals, all40 exact reconstructions and the scoped interpretation.
All clipping counts were zero. z removes only the final return cast;
upstream rounding and native FP32 head arithmetic remain. In the native
source the return cast occurs AFTER the recurrence, so this is not evidence
of quantization feeding back into hidden states.

The secondary native N-L covariance-circuit effects predominantly move the
output quantization phase at this first head, not the pre-return directional
response. This does not imply that covariance arithmetic is unimportant at
later layers. All original full-card gates and failed optimizer verdicts
remain unchanged. No new numerical acquisition follows from this addendum.
