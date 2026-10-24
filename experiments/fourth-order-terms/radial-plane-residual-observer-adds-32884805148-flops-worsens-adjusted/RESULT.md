# Radial-plane residual mean observer: fails the complete adjusted objective

The bank's nonlinear maps form a full postactivation fourth repeated plane Q
and reduce it to a radial scalar. This fixed observer subtracts its own
symmetric radial projection from Q, casts that residual D to FP32, and
contracts the coordinate-pair tensor with the next actual weight matrix.
It then propagates a separate mean correction through the original Gaussian
gains. The natural coefficient is one at every layer; there is no fit or
selection. Parent means, covariances, cumulant banks, selectors, learned
heads and physical state remain unchanged. The original output receives
only the observer's mean shift before the original nonnegative FP32 cast.

| Existing 1B network | Raw error ratio | Complete cold cost ratio | Adjusted error ratio |
| --- | ---: | ---: | ---: |
| 35 | 1.002188913427 | 1.079537813020 | 1.081900827833 |
| 100 | 1.004095817722 | 1.078999727451 | 1.083419113657 |

Pooled adjusted ratio: **1.082647443018**. The fixed
gate requires both adjusted ratios <1 and pooled <=.99. Passed: False.
Cold parent bills are 413448696909 /
416264792412 F; candidate bills are
446333502057 / 449149597560 F.
Every formation, cast, copy, matrix product and read is charged inside
predict. Extra observer work is 32884805148 F per call.
The cost multiplier is each full cold bill divided by 2^41, floored at .1.
Independent scalar-fsum and NumPy loss reductions agree. Predictions,
bills and qualification were sealed before ordinary reference target
decoding. These reused development networks are not held-out validation
and these ratios are not official scores.

Six tiny signed, zero and unequal-variance tensor cases pass, with explicit
dense fourth tensors, double-trace cancellation, coordinate rescaling,
permutation, independent mean recurrences, input preservation, and repeated
FNP bytes and bills. The zero double trace applies before FP32 rounding.
Three whole-function inverses and the complete parent byte prefix verify.
Disabled cold35 exactly reproduces all 16 parent rows and the complete bill.
Enabled35/100/35 and independent cold100 each preserve all 16 underlying
parent output rows, and subtracting measured helper work restores the
matching complete parent bill exactly. Own predictions, rays, sources and
shifts repeat exactly for repeated35 and warm/cold100. First output rows,
finite/nonnegative FP32 contract and numeric-cache retirement checks pass.

The independent diagnostic source, ray and full mean-recurrence comparisons
each cover 60 layer events. Their maximum relative errors are
2.90872266099e-08, 1.51001081694e-06 and
1.4913018349e-06, within frozen 2e-6 / 3e-5 / 3e-5 limits.
Research NumPy reference operations do not feed candidate results. Research
wall times are not production runtime qualification.

This tensor has only iiii and iijj entries. Its other collision sectors are
zero; the parent retains its complete existing baseline. The observer omits
covariance, learned-state and higher-cumulant feedback, and is not the full
native tangent or a complete fourth-state carrier. This is distinct from
the older fourth-plane-only chain, which replaced the fourth-state baseline.

Five full native calls total 2174583804499 F. Six serial
guarded groups retired; controller 77086 is terminal. Peak sampled group
RSS is 6718504960 B and self-report is
6718455808 B; added swap is zero. Fast
stages can understate sampled peaks. Artifacts before closeout total about
2.361 MB. No new samples, fits, downloads or GPU calls.
All five official source/archive/receipt/poll/evaluation pins are unchanged.

Close this fixed observer without amplitude, layer, source, precision, gain, fit or blend rescue. No wider panel, production qualification, package or upload follows. This result does not bound all fourth-state or joint-cumulant methods.

Official 331684 remains adjusted 3.5844144212297072e-9, with 100/100 complete and zero failures. The 4e-9 bank is retained; 3/2/~1e-9 remain unmet. The goal remains active.
