# Actual signed-pair read roundoff is measurable and locally small

Both unchanged bank runs trigger the prospective numerical flag: some
effective read discrepancies exceed 3e-5 of the complete off-diagonal K21
norm. They are much smaller than the prior synthetic fixture's percent-scale
error. Their pooled formal covariance responses are about one part per
million of the original off-diagonal covariance. This is numerical evidence
on current states, not a measured improvement in final-layer accuracy.

| Network | Maximum error / full K21 | Pooled error / full K21 | Pooled formal covariance response | Maximum formal covariance response |
| --- | ---: | ---: | ---: | ---: |
| 35 | 5.74941407986e-05 | 3.51190087459e-05 | 1.04917298571e-06 | 1.56654710964e-06 |
| 100 | 5.33202585774e-05 | 3.4213247263e-05 | 1.0825297675e-06 | 1.45958136916e-06 |

Each network flags layers 2 and 4 through 14; layers 1 and 3 remain below
the fixed threshold. All 14 complete reads per network were observed,
including young-only, materialized-old and lazy-old paths. Relative to the
CP read itself, the maximum effective discrepancies are 7.85736010596e-5
and 7.51850321734e-5. The reference preserves the declared old factorization
and the supplied stored operands; it does not substitute rounded reconstructed
right factors, change a tensor, or isolate matrix-kernel error from the other
FP32 arithmetic within a read. Diagonal entries are excluded because the
actual consumer overwrites them through its independent FP64 scalar path.

The FP64 reference uses streamed 64-row products. A second reference sums
257-column contractions before the same old maps. Maximum relative reference
disagreements are 2.63105946172e-15 and 2.41320554950e-15, at most
3.71209273476e-11 of the measured read-error norms. Six tiny fixtures also
agree with independent reconstructed-right scalar math.fsum evaluation.
They cover all ownership modes, zero coefficients, no young columns, empty
width, strides, input preservation, exact repetition and snapshot lifetime
after caller mutation. Actual FNP parent-read output bytes and bills match
with and without the observer, and the observer returns the original object.

The first unit failed its ancillary covariance-affinity check. Source review
had selected an obsolete FunctionDef named nonlinear, overlooking subsequent
module assignments. The effective binding is _nonlinear_pair_batch, with
edge-correlation terms through degree2 and FP32 series arithmetic. The old
degree0 response was incorrect. RESPONSE_CORRECTION.md documents the exact
cause and additive repair; original files and the failure are preserved.
Textual inverse checks recover all four original scripts. Primary read
references, fixtures, native controls and the 3e-5 numerical flag did not
change, and no full-network call occurred before repair.

The corrected local covariance response uses the actual series' formal FP64
algebra with original scalar inputs, including correlation degrees0..2.
Its independent unbatched FP64 finite-difference certificate has maximum
absolute error 1.27393755267e-18, below the original 1e-12 algebra tolerance.
The actual rounded FP32 consumer's unit response differs by norm
2.19303139269e-9, below its separately declared 2.15110744451e-8 rounding
allowance. The recorded network response is the formal series quantity;
it is not an exact derivative of FP32 rounding or the learned autonomous
estimator. Scalar means and marginal diagonals are fixed in that calculation.

All 16 predictions and complete cold bills equal the original bank controls
exactly: 413448696909 and 416264792412 FLOPs. Original functions each run
once in the observed inference path; NumPy reference arithmetic is research
instrumentation, never feeds back, and does not change the paid bank bill.
It is not a free proposed production operation. Both wrappers restore,
both numeric caches retire, and no wide trace arrays were saved. An independent
scalar aggregate replay verifies the pooled ratios and layer flags.

The positive numerical flag alone does not supply an economically useful
candidate. It neither measures a final MSE gain nor bounds accumulated
autonomous effects; downstream states, basis construction, learned heads
and rounding can change under an intervention. No next candidate, precision
or kernel change is selected from this diagnostic. The failed polarized
read and the previous negative FP64 shared-sketch endpoint remain closed.
The receiver-direction result reviewed this turn was already executed and
yielded only 0.5570% old-host TRAIN raw improvement, failing its own gate;
it is not an unrun raw-error lead.

Five guarded groups retired; controllers 93145 and 67613 are terminal.
Two original native calls account for 829,713,489,321
measured FLOPs, excluding offline reference arithmetic. Peak sampled group
RSS was 6,234,374,144 bytes and maximum
reported process RSS 6,607,667,200 bytes.
All guards recorded zero added swap. Maximum snapshot size was
96,235,776 bytes, estimated reference workspace
347,529,984 bytes. About
0.300 MB was preserved before
closeout. No truths, fitting, new Monte Carlo samples, GPU work or downloads.

Official 331684 remains adjusted 3.5844144212297072e-9 and raw
1.949715441540434e-8, 100/100 complete with zero failures. The 4e-9 milestone
is banked; 3e-9, 2e-9 and approximately1e-9 remain unmet. The full Phase2
goal remains active. The separate qualified covariance/readout archive is
unchanged. Automatic approval review rejected its upload because it could
not verify trusted authorization for that exact private archive and AIcrowd
destination. The existing exact-upload question remains unanswered; no retry,
bypass, submission receipt or upload occurred in this study.
