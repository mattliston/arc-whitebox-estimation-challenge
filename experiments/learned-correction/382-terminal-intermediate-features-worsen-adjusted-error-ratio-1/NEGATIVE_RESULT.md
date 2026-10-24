# Terminal intermediate representation: fixed information gate failed

The predeclared20-network, five-fold information test completed and failed.
No full-data readout fit, CHECK opening, feature/penalty sweep, candidate
extension or deployment follows this result.

| Exploratory TRAIN-network CV arm | Pooled adjusted error |
| --- | ---: |
|Unchanged Read5|3.989137621921596e-9|
|Affine [1,H15] control|3.974845830335309e-9|
|Intermediate-feature extension|4.090361903053607e-9|

Extended/unchanged1.0253749784 (+2.53750%); extended/control1.0290617744
(+2.90618%). The extension is worse in ALL five folds. The required gates
were <=.95 of unchanged, <=.97 of control, and4/5 fold wins against both.

The382 extra channels are the actual final encoder input126 and its two
SiLU128 intermediates, before the final64-channel compression. Their
TRAIN-only weighted SVD projection removes the existing [1,H15] span.
Every fold retains control rank65 and369 residual columns; maximum
normalized training cross-correlation is2.33344e-13. Thus duplicate-column
changes to the control ridge penalty do not explain the comparison.
This closes this fixed span/fit, not all nonlinear predictors or a universal
claim that intermediate representations contain no information.

The synthetic duplicate-column check leaves the control exactly unchanged.
An independent combined normal-equation solve agrees with the staged fit
to maximum absolute2.665e-15. These certify the fitting algebra, not efficacy.

All20 complete predictions, existing intermediate arrays and own-cold bills
were sealed before the new target access. Original1B FP32 mc_post means
were widened for scoring, joined by the original target manifest and exact
weight hashes. All OOF served FP32 predictions were sealed before aggregate
scoring. The parent head has historical exposure to these networks, so this
is exploratory network-CV, not independent validation or an official score.
Its1B TRAIN20 score must not replace Read5's separate4M-NC14 panel result.

The capture introduces zero numerical operations. Source removal restores
the original gnn_step bytes. Qualification35/100/35 preserves every parent
output and its FLOP bill exactly; all observed arrays repeat byte exactly.
The score includes1MF/3MF readout reserves beyond each own-cold parent for
control/extension. There is no qualified deployment bill or new payload.

Resources:22 full parent calls,8,922,656,554,727 metered parent FLOPs;
one numerical group at a time. Peak monitored process-tree RSS6,403,293,184B,
zero additional swap, every owned group retired. The small information
fit took2.06s with483.6MB maximum self-reported RSS. Saved research artifacts
occupy about43.1MB; no downloads or large covariance histories were written.

Key evidence:

- information.py SHA256:27a4686eb487625cd41c0f152cbb4a4639105e180098f19160ddc321ede0b319
- train_capture/manifest.json SHA256:5c49e3f22217c597b0702497da93e5d7d034d115de762a00c9923ee2fb94e339
- Full scalar/row/fold result:information/result.json
- Source, resource and artifact closeout:closeout.json

Read5 candidate/archive remain unchanged and unsubmitted, pending explicit
approval of that exact archive. The earlier automatic Cost75 upload
rejection is not bypassed. Official best remains graded#331584,
4.015535762022727e-9; the approximately1e-9 goal remains active and unfinished.
