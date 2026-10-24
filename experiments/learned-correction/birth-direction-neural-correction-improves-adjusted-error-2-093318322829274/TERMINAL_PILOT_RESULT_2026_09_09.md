# Fixed node-head pilot: small measured gain, advancement gate missed

2026-09-09 UTC (2026-09-08 Pacific), Phase2 only. The one fixed20-TRAIN/4-VALID, six-epoch30-update fit finished. No extension, refit, checkpoint selection or hyperparameter selection.

Training result SHA256 `99175824b72910818d445c74109ff0236d803bd608181aeaf0bad11108891d4f`; terminal candidate `f072ad28a70698c152fa9a9ce31ba78191d9aece4b6928087d9b99c49a8dd30a`; terminal build `3efa491ce8c2aa4e7725346b87e33602a8f384e4ae9aa3fc9bb4246c0e193f9a`.

Final score result `terminal_score_v2/result.json`, SHA256 `ffb2ed5a3386674d707a37918ce454c20c16f6ca7f906e111adfd335b9934930`. The original v1 metadata-reader failure is preserved. See SCORER_METADATA_REPAIR_V2.md; only the unsupported nonempty instrumentation requirement was corrected. All numerical gates and scientific thresholds remained unchanged.

Frame: fixed reused-development VALID37/39/42/48, observed existing1e9 `mc_post` final-layer MSE multiplied by each candidate's own complete native scoring multiplier. Parent is exact d50; its original first-four/cold37 prefix is reused after verifying all41 archived records. New zero and terminal roles each ran all four plus exact repeat with their own input/lifecycle/bills. No estimated-cost substitution or repeat averaging.

| Role | Mean raw MSE | Mean adjusted | Mean complete FLOPs |
| --- | ---: | ---: | ---: |
| Parent d50 |2.026176015404889e-8|4.367945309309118e-9|475111388300.25|
| Zero head |2.026176015404889e-8|4.3680821740128575e-9|475126242316.25|
| Terminal head |1.984349169123722e-8|4.276510309818188e-9|475126242316.25|

Terminal adjusted improves2.093318322829274% against parent and2.096386023583985% against zero. Three of the same four nets beat BOTH controls; rank37 is slightly worse. The prospective mean-gain threshold was3% against each control, so **pilot_gate_passed=false**. The score itself completed validly: passed/completed/integrity/numerical_fidelity true.

All16 zero outputs are byte-identical to parent. Terminal native/MPS maximum differences2.3842e-6–2.6226e-6 are below1e-5; final-MSE ratios0.9992619094580636–1.0003822472662278 are within1%. All four predictions/bills and terminal A were sealed before validation truths. No production or independent14 claim is made.

Preserve the terminal artifact and this small positive development signal as a possible future combination component, not a fully independent-panel validated bank. No automatic larger fit, retraining, standalone panel or submission under this pilot. The current official bank #330390 remains the earlier d50 candidate at4.53886349950395e-9; this node-head extension is not on the board.
