# On-policy readout

The second-step linear readout fitted by ridge regression on captures of the estimator's own trajectory.

| Date | Experiment | Programs | Outcome |
|---|---|---:|---|
| 2026-09-13 | [Terminal quadratic readout loses every fold: adjusted ratio 1.0784432697076234 on 20 research networks](terminal-quadratic-readout-loses-every-fold-adjusted-ratio-1/README.md) | 181 | rejected |
| 2026-09-27 | [Boosted-tree graph head: development raw error rises 0.083935%](boosted-tree-graph-head-development-raw-error-rises-0/README.md) | 3 | rejected |
| 2026-09-27 | [Covariance actions on 64 hidden channels: cross-fitted raw error rises 8.078767%](covariance-actions-64-hidden-channels-cross-fitted-raw-error/README.md) | 8 | rejected |
| 2026-09-27 | [Current terminal 65-feature ridge refit on six networks: adjusted error rises 1.136865%](current-terminal-65-feature-ridge-refit-six-networks-adjusted/README.md) | 12 | rejected |
| 2026-09-27 | [Eight-coefficient reference-trained readout: native adjusted error rises 2.145536%](eight-coefficient-reference-trained-readout-native-adjusted-error-rises/README.md) | 14 | rejected |
| 2026-09-27 | [Native final-error graph head: fitted gain reverses to 2.97073% raw loss on 32 check networks](native-final-error-graph-head-fitted-gain-reverses-2/README.md) | 17 | rejected |
| 2026-09-27 | [Network-level five-mode calibration: feature head loses and oracle gain is only 1.548859%](network-level-five-mode-calibration-feature-head-loses-oracle/README.md) | 4 | rejected |
| 2026-09-27 | [Same-layer covariance and third-plane response features: cross-fitted raw error rises 8.60760%](same-layer-covariance-third-plane-response-features-cross-fitted/README.md) | 9 | rejected |
| 2026-09-27 | [Terminal covariance contraction actions: 15-coefficient head worsens check-network raw error 0.009766%](terminal-covariance-contraction-actions-15-coefficient-head-worsens-check/README.md) | 6 | rejected |
| 2026-09-30 | [Equal averaging of eight- and sixteen-update outputs fails the fixed gain and win gates](equal-averaging-eight-sixteen-update-outputs-fails-fixed-gain/README.md) | 4 | rejected |
| 2026-09-30 | [Graph-feature linear augmentation increases cross-validated raw error 0.066533%](graph-feature-linear-augmentation-increases-cross-validated-raw-error/README.md) | 9 | rejected |
| 2026-09-30 | [Pooled-hidden-state terminal interactions worsen fixed cross-validation relative to the additive head](pooled-hidden-state-terminal-interactions-worsen-fixed-cross-validation/README.md) | 8 | rejected |
| 2026-09-30 | [Terminal 65-parameter ridge fit on 160 cached training examples fails eight-network confirmation](terminal-65-parameter-ridge-fit-160-cached-training-examples/README.md) | 25 | rejected |
| 2026-09-30 | [Terminal learned-correction scalar calibration improves raw error 0.07041732418%, below the fixed gate](terminal-learned-correction-scalar-calibration-improves-raw-error-0/README.md) | 34 | rejected |
| 2026-09-30 | [Two-scalar final affine fit: grouped leave-one-out gain is negative](two-scalar-final-affine-fit-grouped-leave-one-out/README.md) | 2 | rejected |
| 2026-10-17 | [Saved-state final-head variants using reflection, Hermite features and nonlinear models all stay within ±0.08%](saved-state-final-head-variants-using-reflection-hermite-features/README.md) | 1 | rejected |
| 2026-10-21 | [First-128 hidden readout: -.367205943% Mac32 and -.4015029588562216% Linux32 versus #337466](first-128-hidden-readout-367205943-mac32-4015029588562216-linux32-versus/README.md) | 15 | adopted |
| 2026-10-21 | [Gate times first-128 readout: -0.148839615% cross-validation misses the fixed -0.15% threshold](gate-times-first-128-readout-0-148839615-cross-validation/README.md) | 49 | rejected |
| 2026-10-21 | [Full-hidden on-policy second fit: -0.5898238461% cross-validation becomes +.102475306% native adjusted](full-hidden-policy-second-fit-0-5898238461-cross-validation/README.md) | 20 | rejected |
| 2026-10-21 | [First-128 on-policy refit strength: zero added FLOPs and about -0.3% adjusted across native cohorts](first-128-policy-refit-strength-zero-added-flops-about/README.md) | 44 | adopted |
| 2026-10-21 | [Second-step readout with cap 4: -0.190% Mac, -0.220% Linux and -0.018% fresh adjusted](second-step-readout-cap-4-0-190-mac-0/README.md) | 4 | rejected |
| 2026-10-21 | [Squared and gate readout features: -0.733% and -0.836% cross-validation at 577 and 705 columns](squared-gate-readout-features-0-733-0-836-cross/README.md) | 44 | adopted |
| 2026-10-21 | [705-column gate readout with cap 3: strength 1.5 gives -0.176% pooled adjusted across 76 observations](705-column-gate-readout-cap-3-strength-1-5/README.md) | 7 | adopted |
| 2026-10-21 | [705-column gate readout with cap 4: -0.378% pooled adjusted, held for limited runtime margin](705-column-gate-readout-cap-4-0-378-pooled/README.md) | 3 | rejected |
| 2026-10-21 | [A further on-policy iteration adds about 0.04% raw gain on Mac and the ridge override is ineffective](further-policy-iteration-adds-about-0-04-raw-gain/README.md) | 44 | rejected |
| 2026-10-21 | [Gate ridge and damping: 0.03 with damping 0.5 improves proxy fit but misses the submission threshold](gate-ridge-damping-0-03-damping-0-5-improves/README.md) | 3 | rejected |
| 2026-10-21 | [Gate-readout learning curve: held-out change reaches -0.826% with 72 training networks](gate-readout-learning-curve-held-out-change-reaches-0/README.md) | 4 | informational |
| 2026-10-21 | [Hidden times gate features improve native score; ridge 0.01 is worse than 0.03](hidden-times-gate-features-improve-native-score-ridge-0/README.md) | 1 | adopted |
| 2026-10-21 | [Hidden times gate plus first-128 times hidden: -0.995% held-fold proxy at 961 columns](hidden-times-gate-plus-first-128-times-hidden-0/README.md) | 8 | adopted |
| 2026-10-21 | [Nonuniform layer strengths for the hidden-times-gate readout do not improve on uniform 1.5](nonuniform-layer-strengths-hidden-times-gate-readout-do-not/README.md) | 5 | rejected |
| 2026-10-21 | [961-column readout with cap 4 and size threshold: -0.620% Linux and -0.423% Mac before accounting repair](961-column-readout-cap-4-size-threshold-0-620/README.md) | 3 | adopted |
| 2026-10-21 | [Final readout strength 1.35 improves Mac and Linux panels; strengths 1.05 through 1.65 give a broad optimum](final-readout-strength-1-35-improves-mac-linux-panels/README.md) | 5 | adopted |
| 2026-10-21 | [Strength 1.28 and late-layer strength 1.0 tie the final source within measured Linux variation](strength-1-28-late-layer-strength-1-0-tie/README.md) | 3 | inconclusive |
