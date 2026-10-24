# Fixed nonlinear memory capacity round: closed

Written September 7, 2026 PDT (machine clock September 8 UTC). Phase 2
only. All eight current-bank captures, both fixed fits, prediction evaluation
and current-head consumer evaluation completed. Existing sources, receipts,
models and protocols remain frozen. No parameter sweep, refit, port or board
submission follows from this failed scientific gate.

## Ownership and qualification

The captured teacher is current candidate0b84ab6fa96d32804c9adec350d41766e47d09a71a936b95652893bb1e364a96.
Every capture matched its archived predictions and complete metered cost,
including a same-source warm control. Train35/100/150/200/250/300;
held-out400/500. Two exactly specified16->8->8->2 models, seed9062047,
100epochs each, identical settings, no held-out model selection.

Training completed893.2896578s, peak host RSS2,521,317,376B. All final
models were sealed before held-out arrays were decoded. Predictions for
both arms on all eight nets were sealed before action targets were opened.
Independent learned-parameter NumPy64/MPS32 fixture passed2e-5 absolute
tolerance (worst observed4.0222e-8). No MC truth or current-head output was
used as a model input.

Consumer evaluation completed57.1559758s, peak RSS2,503,573,504B. Native
teacher features, head deltas, corrected means and masked covariance were
byte-identical to captures at every tested layer/net. Predictions changed
only the third-action input. This is teacher-forced local discrepancy,
NOT endogenous final-layer raw MSE or adjusted score.

## Held-out result

All plane numbers are standardized full-plane MSE, equal mean over
query layers1..14. Consumer numbers are squared physical discrepancies
from the current teacher, not errors against independent truths.

| Quantity | rank400 control | rank400 memory | rank500 control | rank500 memory |
| --- | ---: | ---: | ---: | ---: |
| Plane MSE |9.221599507e-5|9.323458288e-5|8.855818896e-5|8.952584915e-5|
| Immediate post-head mean discrepancy |6.731541796e-8|6.967474055e-8|6.259395949e-8|6.476906494e-8|
| Next physical variance discrepancy |1.106560880e-6|1.178090560e-6|1.046178668e-6|1.135685558e-6|

Memory/control plane ratios1.011045674 and1.010926829; mean-consumer
ratios1.035048770 and1.034749447; variance-consumer ratios1.064641432
and1.085556026. Pooled held-out plane MSE9.138021601e-5 versus control
9.038709202e-5 (ratio1.010987454). Memory/zero-decoder ratio.368283964;
this relative improvement does not establish superiority to the matched
control or an endogenous score.

The prospectively fixed gate required at least20% plane improvement from
H over D-only on EACH held-out net, and smaller immediate mean AND
next-variance discrepancies. All three tests fail on both nets. Therefore
the fixed nonlinear H spelling is closed. The gate was an information
screen, not an assertion of a necessary raw-error percentage. This does
not prove that every nonlinear tensor compression is impossible. It also
does not license an after-the-fact rescue of the control or extra epochs.

## Score status

Official bank329926 remains4.982231460183304e-9 adjusted. Best qualified
local0b84 remains4.788089831892398e-9 on the independent14-net panel.
No new adjusted-score evidence was produced by this capacity round.
The estimated full-student279.846B cost was never a measured deployment
bill and must not be combined with these teacher-forced errors.

## Frozen SHA256 bindings

- TRAINING_PROTOCOL.md: df12da9dd2179e7442e9ddcb60e9f4debbddd00bdf7042e8b36bac6dc3212cef
- CONSUMER_PROTOCOL.md: 47adf1ce9b6f9bdf8cbee7f7004af4015444042565ed2b9aea46fd70fae6e764
- train_capacity.py: f53d3c577cb8c198faccdc3c14bd4689828cef4600471194962ad32eefed9cc6
- evaluate_predictions.py: 869b2ff15078f42cbf9223eae5ddb426b3ef6997fdccb13733bee391d94aa12e
- evaluate_consumers.py: d36ed7e5554eff4c8902b8440eef66fc6eba065f2d9ceaf363174711da44b387
- training/result.json: bdbf47a842093cbe4194da2bddf6328aedb88d7923bdb302d7b5be6d0352e383
- predictions/result.json: e0663bf0202c4bc1bb31e00cf6726539d4f916c7cc8bf7433539c8266d008cbd
- consumers/result.json: 6bfe09d6abe5b8d2493f699684b9a8b4c42e6399eb3369d7a04acd94d6de2fea

These results close only this fixed round; they do not change the user's
active goal or create a new approval requirement for other in-scope work.
