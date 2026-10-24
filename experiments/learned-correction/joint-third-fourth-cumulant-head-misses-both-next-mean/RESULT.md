# Joint incoming third/fourth head fails its fixed information gate

The single model finished its prescribed 32 epochs and 1,344 updates. Both
validation networks fail the next-layer mean response gate. The fixed
experiment is closed, with no extra epochs, model selection, amplitude
change, native candidate, final-score evaluation, package or submission.

| MINI ordinal | Weighted T21 | Weighted T31 | Weighted T22 | Joint covariance | Joint next mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| 14 | 0.721598902 | 0.794306777 | 0.853710691 | 0.737520421 | 0.917358747 |
| 71 | 0.711302170 | 0.783309956 | 0.876363142 | 0.726110964 | 0.931169619 |

Each network had to attain joint covariance AND next-mean ratios at most
0.90. The three plane ratios are auxiliary reports, not acceptance gates.
Ratios use frozen TRAIN per-layer energy normalizers. They are squared
errors in real-arithmetic responses on unchanged parent trajectories;
they are not actual final-layer MSE or FLOP-adjusted scores. Both covariance
gates pass, while next-mean ratios 0.917358747 and 0.931169619 fail. The
independent binomial centering and full matrix congruence replay preserves
the verdict. Its maximum target, layer-loss and aggregate-ratio relative
differences are 1.97220580418e-11,
2.85751245099e-14 and
1.68189903192e-15, respectively.

This is a new three-output model trained from scratch. It jointly predicts
normalized offdiagonal K21, K31 and K22 corrections at layers 8 through 14.
Its 140 causal node features include previous-parent H64 transported by W
and W squared; pair inputs are current R, S and S transpose. Only predicted
K22 is symmetrized. The exact directed response uses the same rho_ij for
both orientations and was certified on asymmetric and active-floor fixtures
before fitting. The earlier single-third model and its closed verdict are
unchanged. No complete fourth tensor or exact all-distinct emission was
implemented or tested by this experiment.

TRAIN MINI ordinals were 57, 68, 45, 59, 3 and 12. Validation ordinals 14
and 71 had already been used by the previous experiment and are explicitly
reused validation, not pristine holdouts. Their new fourth-moment fields
were admitted only after this final model was hash-frozen. All eight
existing 100M public payloads were reused without downloads or sampling.
Newly decoded fields were pre_m4, pre_M31 and pre_M22, alongside previously
admitted preactivation moments. No 1B target field was decoded.

All eight source checks passed. Maximum Gaussian first-layer standardized
fourth RMS was 1.00262565459, maximum entry
5.60232540131; limits were 1.5 and 12.
Maximum raw fourth diagonal discrepancy was
0.499999050051 FP32 ULPs (limit 2).
The maximum late-layer FP32 rounding bound was
0.0224672651498 of the missing-plane RMS
(limit 0.05). These checks do not remove late-layer Monte Carlo noise or
shared numerical bias. Source credits and pinned revisions are recorded
in DATA_PROVENANCE.md and the source seals.

Centering, Wick influence variance, directed source response, analytic
gradient, FNP equation, permutation, zero-output and repeat checks passed.
One complete standalone head and physical correction with cold array
parameter ingress measured 1,343,427,584 weighted FLOPs; the conservative
seven-layer sum was 9,403,993,088, below the 15-billion component cap. There
is no full changed-estimator bill. The unexecuted build.py adapter draft
is preserved as preparatory work; no native source seal or candidate exists.

All 12 guarded groups retired successfully. Maximum sampled group
and process peak RSS were 5,443,518,464 and
5,443,518,464 bytes, respectively. Added
swap was zero. New artifacts before closeout total
1,409,565,720 bytes, within the 2.5GB cap;
free disk was 485.144GB. No new parent or changed
native trajectory was run. This is a scoped negative for this model,
features, fixed loss and optimizer, not an impossibility result for joint
moment prediction.

Official submission 331684 remains adjusted 3.5844144212297072e-9 and raw
1.949715441540434e-8, with 100/100 cases and zero failures. The 4e-9 milestone
is banked; 3e-9, 2e-9 and approximately 1e-9 remain unmet. The goal is active.

Automatic approval review previously rejected uploading the separate
qualified covariance/readout archive because it could not verify trusted
authorization for the exact private payload and AIcrowd destination. The
existing approval question is unanswered; this experiment made no retry.
