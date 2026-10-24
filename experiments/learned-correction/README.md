# Learned per-layer correction

The learned per-layer correction network (observer, edge module, per-layer heads), its training and its refits.

| Date | Experiment | Programs | Outcome |
|---|---|---:|---|
| 2026-08-24 | [688-network trajectory scan: all 11 tail failures fell below 5e-7 after the variance-floor repair](688-network-trajectory-scan-all-11-tail-failures-fell/README.md) | 18 | informational |
| 2026-08-24 | [Exposure-trained own-rollout residual state: measured and rejected](exposure-trained-own-rollout-residual-state-measured-rejected/README.md) | 1 | rejected |
| 2026-08-24 | [Exposure-trained two-auxiliary-state successor: measured and rejected](exposure-trained-two-auxiliary-state-successor-measured-rejected/README.md) | 1 | rejected |
| 2026-08-24 | [Learned backward-adjoint residual observer: rejected in full-width cross-validation](learned-backward-adjoint-residual-observer-rejected-full-width-cross/README.md) | 9 | rejected |
| 2026-08-24 | [Mean-and-variance residual state: endpoint error 6.65959% worse than the dynamic-state observer](mean-variance-residual-state-endpoint-error-6-65959-worse/README.md) | 3 | rejected |
| 2026-08-24 | [Pooled-Gram residual observer: 0.3378137% worse than the transport-energy observer](pooled-gram-residual-observer-0-3378137-worse-than-transport/README.md) | 4 | rejected |
| 2026-08-24 | [State-independent Duhamel innovation cross-validation: endpoint error 18.5369% worse](state-independent-duhamel-innovation-cross-validation-endpoint-error-18/README.md) | 3 | rejected |
| 2026-08-25 | [Carrier-host refit ceiling: measured and rejected](carrier-host-refit-ceiling-measured-rejected/README.md) | 8 | rejected |
| 2026-08-25 | [Composed shell and fourth-order carrier exported deterministically at 1954.3e9 FLOPs](composed-shell-fourth-order-carrier-exported-deterministically-1954-3e9/README.md) | 29 | informational |
| 2026-08-25 | [Corrector retraining on the joint host: adjusted 2.833e-8 lost to the 2.2843e-8 submission](corrector-retraining-joint-host-adjusted-2-833e-8-lost/README.md) | 26 | informational |
| 2026-08-25 | [Frozen corrector plus fourth-cumulant carrier: 34% worse than the deployed baseline](frozen-corrector-plus-fourth-cumulant-carrier-34-worse-than/README.md) | 10 | rejected |
| 2026-08-25 | [Held-fold ensembles of five residual observers: best error 3.221703035370728e-8 missed the material-gain gate](held-fold-ensembles-five-residual-observers-best-error-3/README.md) | 2 | rejected |
| 2026-08-25 | [Host-uncertainty observable tape augmentation: measured and rejected](host-uncertainty-observable-tape-augmentation-measured-rejected/README.md) | 35 | rejected |
| 2026-08-25 | [Joint cubic shell and fourth-order carrier: corrected error 3.1339e-8 versus 3.1334e-8 baseline](joint-cubic-shell-fourth-order-carrier-corrected-error-3/README.md) | 6 | rejected |
| 2026-08-25 | [Network-adaptive rank-16 cross-unit residual covariance observer: measured and rejected](network-adaptive-rank-16-cross-unit-residual-covariance-observer/README.md) | 11 | rejected |
| 2026-08-25 | [Rank-4 fourth-cumulant two-pair residual carrier: host error fell 35.6%, corrected error stayed flat](rank-4-fourth-cumulant-two-pair-residual-carrier-host/README.md) | 16 | informational |
| 2026-08-26 | [Additional supplied-host readout features: honest all-200 error fell 1.43%](additional-supplied-host-readout-features-honest-all-200-error/README.md) | 2 | rejected |
| 2026-08-27 | [688-network endpoint-action head refit: accepted relative to its parent but worse than the native observer](688-network-endpoint-action-head-refit-accepted-relative-parent/README.md) | 13 | informational |
| 2026-08-27 | [688-network global shrinkage diagnostic: 0.07894% gain missed the acceptance threshold](688-network-global-shrinkage-diagnostic-0-07894-gain-missed/README.md) | 17 | rejected |
| 2026-08-27 | [Endpoint-action correction head: local projection failed to transfer to the public grade](endpoint-action-correction-head-local-projection-failed-transfer-public/README.md) | 70 | rejected |
| 2026-08-27 | [Exact layer-7 scalar residual-state reset: endpoint error 3.3897455206496917e-8 missed the target](exact-layer-7-scalar-residual-state-reset-endpoint-error/README.md) | 2 | rejected |
| 2026-08-27 | [Exact-weight autonomous suffix state: all four held-out folds regressed](exact-weight-autonomous-suffix-state-all-four-held-out/README.md) | 6 | rejected |
| 2026-08-27 | [Fold-pure exact-weight suffix state initialized from the nonlinear head: native baseline still better](fold-pure-exact-weight-suffix-state-initialized-nonlinear-head/README.md) | 7 | rejected |
| 2026-08-27 | [Global context for the exact-weight suffix state: 0.102505% gain missed the 0.25% gate](global-context-exact-weight-suffix-state-0-102505-gain/README.md) | 9 | rejected |
| 2026-08-27 | [Nonlinear transported 20-field observation head: 0.402903% cross-validated improvement](nonlinear-transported-20-field-observation-head-0-402903-cross/README.md) | 5 | rejected |
| 2026-08-27 | [Third-cumulant cohort retirement: every tested age window worsened endpoint error](third-cumulant-cohort-retirement-every-tested-age-window-worsened/README.md) | 2 | rejected |
| 2026-08-30 | [Cross-network fourth-cumulant times mean correction worsened held-out error capture](cross-network-fourth-cumulant-times-mean-correction-worsened-held/README.md) | 2 | informational |
| 2026-08-31 | [82-input signed/absolute/squared message head: measured and rejected](82-input-signed-absolute-squared-message-head-measured-rejected/README.md) | 7 | rejected |
| 2026-08-31 | [86-input endpoint-action head: 0.09354760270741128% of oracle-gap recovery missed the 8% gate](86-input-endpoint-action-head-0-09354760270741128-oracle-gap/README.md) | 11 | rejected |
| 2026-08-31 | [Collective injection worsened bare host error from 8.593e-8 to 2.218e-7](collective-injection-worsened-bare-host-error-8-593e-8/README.md) | 2 | informational |
| 2026-08-31 | [Native-frame collective-feature refit: 1.40% gain failed the preregistered halving requirement](native-frame-collective-feature-refit-1-40-gain-failed/README.md) | 5 | rejected |
| 2026-09-01 | [State assimilation improved correlation, with optimum shrinkage near 0.75](state-assimilation-improved-correlation-optimum-shrinkage-near-0-75/README.md) | 13 | informational |
| 2026-09-01 | [Streaming state prototype below parity had omitted its planned assimilation step](streaming-state-prototype-below-parity-had-omitted-planned-assimilation/README.md) | 13 | informational |
| 2026-09-03 | [Newly generated fourth-order block-path feed had negligible value at width 1024](newly-generated-fourth-order-block-path-feed-had-negligible/README.md) | 8 | rejected |
| 2026-09-04 | [Engine-derived skew shifts worsened the second development network](engine-derived-skew-shifts-worsened-second-development-network/README.md) | 4 | informational |
| 2026-09-04 | [Fourth-order plane-chain integration passed parity; its refitted correction remained worse than baseline](fourth-order-plane-chain-integration-passed-parity-refitted-correction/README.md) | 18 | informational |
| 2026-09-04 | [Removing coherent engine-skew bias improved alignment but failed cross-network endpoint transfer](removing-coherent-engine-skew-bias-improved-alignment-but-failed/README.md) | 5 | informational |
| 2026-09-05 | [Rank, covariance, skew and factor-carrier screens on the independently rebuilt closure](rank-covariance-skew-factor-carrier-screens-independently-rebuilt-closure/README.md) | 13 | informational |
| 2026-09-05 | [Newborn-only pruning: bounded branch closed](newborn-only-pruning-bounded-branch-closed/README.md) | 43 | informational |
| 2026-09-05 | [Radial fourth-to-third cumulant interaction with exact pair batching: locally validated candidate](radial-fourth-third-cumulant-interaction-exact-pair-batching-locally/README.md) | 43 | informational |
| 2026-09-05 | [Third-tensor directional features: bounded branch closed](third-tensor-directional-features-bounded-branch-closed/README.md) | 43 | informational |
| 2026-09-05 | [Correlation-graph features: both closed-loop development networks worsened](correlation-graph-features-both-closed-loop-development-networks-worsened/README.md) | 18 | rejected |
| 2026-09-05 | [Joint-plane information probe: closed](joint-plane-information-probe-closed/README.md) | 13 | rejected |
| 2026-09-05 | [Third-moment recentering: measured correction rejected](third-moment-recentering-measured-correction-rejected/README.md) | 9 | rejected |
| 2026-09-05 | [Anisotropic fourth source closed](anisotropic-fourth-source-closed/README.md) | 11 | rejected |
| 2026-09-05 | [Centered-feedback convention closed after matched closed-loop test](centered-feedback-convention-closed-after-matched-closed-loop-test/README.md) | 17 | rejected |
| 2026-09-05 | [Compact local complement averaging closed](compact-local-complement-averaging-closed/README.md) | 15 | rejected |
| 2026-09-05 | [Fixed suffix training pilot: closed](fixed-suffix-training-pilot-closed/README.md) | 13 | rejected |
| 2026-09-05 | [Conditional skew energy information probe: closed](conditional-skew-energy-information-probe-closed/README.md) | 3 | rejected |
| 2026-09-05 | [Late rank-8 covariance congruence correction: fixed pilot rejected](late-rank-8-covariance-congruence-correction-fixed-pilot-rejected/README.md) | 17 | rejected |
| 2026-09-05 | [Complete heavy-path completed-square birth: closed](complete-heavy-path-completed-square-birth-closed/README.md) | 14 | rejected |
| 2026-09-05 | [Conditional quadratic full-tensor information: closed](conditional-quadratic-full-tensor-information-closed/README.md) | 15 | rejected |
| 2026-09-05 | [Coherent state rescaling: fidelity qualified for the bounded response test](coherent-state-rescaling-fidelity-qualified-bounded-response-test/README.md) | 23 | informational |
| 2026-09-06 | [Final-layer variance-source pilot: measured and rejected](final-layer-variance-source-pilot-measured-rejected/README.md) | 2 | rejected |
| 2026-09-06 | [Final-layer cohort-source pilot: measured and rejected](final-layer-cohort-source-pilot-measured-rejected/README.md) | 22 | rejected |
| 2026-09-06 | [Learned endpoint-history covariance decoder: fixed information test failed](learned-endpoint-history-covariance-decoder-fixed-information-test-failed/README.md) | 2 | rejected |
| 2026-09-06 | [Complete prefix and endpoint co-adaptation toolchain: validation passed](complete-prefix-endpoint-co-adaptation-toolchain-validation-passed/README.md) | 28 | informational |
| 2026-09-07 | [Learned 10-parameter recurrent pair memory fails all four held-out gates against zero and reset controls](learned-10-parameter-recurrent-pair-memory-fails-all-four/README.md) | 26 | rejected |
| 2026-09-07 | [Input-gradient Gram observer fails all three gates: pooled raw ratio 1.00093491602506 versus the reference](input-gradient-gram-observer-fails-all-three-gates-pooled/README.md) | 9 | rejected |
| 2026-09-07 | [Frozen response-row reduction improves two-network adjusted score by 3.63221%](frozen-response-row-reduction-improves-two-network-adjusted-score/README.md) | 17 | adopted |
| 2026-09-07 | [Frozen response-row reduction improves independent14 adjusted score by 3.25073% and is retained](frozen-response-row-reduction-improves-independent14-adjusted-score-3/README.md) | 21 | adopted |
| 2026-09-08 | [Cubic-free decoder-only estimator gives adjusted ratio 43.94708516 versus the matched reference](cubic-free-decoder-only-estimator-gives-adjusted-ratio-43/README.md) | 19 | rejected |
| 2026-09-08 | [Full-FP64 response shadow fails current-width primal compatibility at the correlation-row RMS feature](full-fp64-response-shadow-fails-current-width-primal-compatibility/README.md) | 25 | rejected |
| 2026-09-08 | [Nonlinear pair memory worsens held-out plane error: pooled memory/control ratio 1.010987454](nonlinear-pair-memory-worsens-held-out-plane-error-pooled/README.md) | 25 | rejected |
| 2026-09-09 | [65-coefficient terminal ridge gains 0.75198181029575% on native40 but fails the independent public group](65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/README.md) | 69 | rejected |
| 2026-09-09 | [Birth-direction neural correction improves adjusted error 2.093318322829274% but misses its 3% gate](birth-direction-neural-correction-improves-adjusted-error-2-093318322829274/README.md) | 53 | rejected |
| 2026-09-09 | [Frozen birth-direction and mean-head neural composition passes native40 and independent14 at ratio 0.9741291234422259](frozen-birth-direction-mean-head-neural-composition-passes-native40/README.md) | 117 | adopted |
| 2026-09-09 | [Native directional search gains 0.8624937% on fit networks but worsens checks by 2.0150121%](native-directional-search-gains-0-8624937-fit-networks-but/README.md) | 39 | rejected |
| 2026-09-09 | [One-pass mean-head refit passes native40 at ratio 0.9918044807401466 and independent14 at .991026496598678](one-pass-mean-head-refit-passes-native40-ratio-0/README.md) | 82 | adopted |
| 2026-09-10 | [Joint covariance and mean fitting loses every check network: adjusted ratio 1.772556056561821](joint-covariance-mean-fitting-loses-every-check-network-adjusted/README.md) | 116 | rejected |
| 2026-09-10 | [Joint mean and birth-direction neural fit loses to mean-only control; mean-only adjusted gain 0.3463% is retained](joint-mean-birth-direction-neural-fit-loses-mean-only/README.md) | 91 | inconclusive |
| 2026-09-10 | [Two extra mean-head passes give mixed-backend raw ratio 0.9980865177902755, 23/40 wins](two-extra-mean-head-passes-give-mixed-backend-raw/README.md) | 1 | rejected |
| 2026-09-12 | [Fixed temporal observer gains 1.95462% training error but fails its 10% usefulness gate](fixed-temporal-observer-gains-1-95462-training-error-but/README.md) | 199 | rejected |
| 2026-09-13 | [Exact native two-layer forward passes but its secant misses signal and curvature gates; adjoint remains untested](exact-native-two-layer-forward-passes-but-secant-misses/README.md) | 188 | inconclusive |
| 2026-09-19 | [382 terminal intermediate features worsen adjusted error: ratio 1.0253749784, every fold loses](382-terminal-intermediate-features-worsen-adjusted-error-ratio-1/README.md) | 6 | rejected |
| 2026-09-19 | [Five-update response-head refit on holdout networks 37/39/42/48: pooled adjusted ratio 1.0002074896](five-update-response-head-refit-holdout-networks-37-39/README.md) | 90 | rejected |
| 2026-09-19 | [Five-update response-head refit on the wall-repaired estimator fails: adjusted ratio 1.0195443375 on network 37](five-update-response-head-refit-wall-repaired-estimator-fails/README.md) | 88 | rejected |
| 2026-09-19 | [Four-member terminal nonlinear ensemble worsens adjusted error: ratio 1.036374342900148, 0/5 folds win](four-member-terminal-nonlinear-ensemble-worsens-adjusted-error-ratio/README.md) | 5 | rejected |
| 2026-09-19 | [Scalar final readout distilled from the head's predictions: 14-net noise-corrected adjusted ratio .9870967012 (held)](scalar-final-readout-distilled-head-s-predictions-14-net/README.md) | 88 | inconclusive |
| 2026-09-19 | [Sixteen rank-8 correlation-energy terminal features, five-fold ridge: .9988638603 vs unchanged, 1.0024666616 vs affine](sixteen-rank-8-correlation-energy-terminal-features-five-fold/README.md) | 7 | rejected |
| 2026-09-19 | [Ten two-step covariance and third-plane features worsen adjusted error: ratio 1.012422254768379](ten-two-step-covariance-third-plane-features-worsen-adjusted/README.md) | 9 | rejected |
| 2026-09-20 | [All-layer recurrent mean observer fails five-fold gate: adjusted ratio 1.00422797606219](all-layer-recurrent-mean-observer-fails-five-fold-gate/README.md) | 20 | rejected |
| 2026-09-20 | [Direct native 16-coefficient covariance-head optimization fails training: adjusted ratio 1.00078780578](direct-native-16-coefficient-covariance-head-optimization-fails-training/README.md) | 22 | rejected |
| 2026-09-20 | [Final recurrent observer on Gaussian propagation reaches training MSE 1.0376e-6, above the 2.5e-8 limit](final-recurrent-observer-gaussian-propagation-reaches-training-mse-1/README.md) | 12 | rejected |
| 2026-09-20 | [Frozen local mean-innovation correction worsens pooled adjusted error 4.281639440%](frozen-local-mean-innovation-correction-worsens-pooled-adjusted-error/README.md) | 18 | rejected |
| 2026-09-20 | [Local mean-innovation network with 102-32-1 layers has normalized training MSE 0.8766 versus the .50 gate](local-mean-innovation-network-102-32-1-layers-has/README.md) | 13 | rejected |
| 2026-09-20 | [Recurrent observer (44,210 scalars) on the frozen lazy-rotation trajectory: TRAIN20 adjusted ratio 1.00353 vs <=0.95](recurrent-observer-44-210-scalars-frozen-lazy-rotation-trajectory/README.md) | 12 | rejected |
| 2026-09-20 | [Shared cross-network 65-feature final covariance tangent readout: extended/parent 1.0253, 0/5 folds, 2/20 nets improve](shared-cross-network-65-feature-final-covariance-tangent-readout/README.md) | 15 | rejected |
| 2026-09-21 | [Copying 65 readout coefficients onto the covariance-edge estimator gives eight-network adjusted ratio 0.995858110143](copying-65-readout-coefficients-onto-covariance-edge-estimator-gives/README.md) | 11 | adopted |
| 2026-09-21 | [Covariance-edge fit completes 160 graphs and 40 updates over 528 parameters with fixed mean-head bytes](covariance-edge-fit-completes-160-graphs-40-updates-over/README.md) | 14 | informational |
| 2026-09-21 | [Covariance-edge refit passes native40 but fails independent public8: adjusted ratio 1.000701542110908](covariance-edge-refit-passes-native40-but-fails-independent-public8/README.md) | 43 | rejected |
| 2026-09-21 | [Cubic hidden-query ridge transfer fails: adjusted ratio 1.030768487997, 1/20 networks improve](cubic-hidden-query-ridge-transfer-fails-adjusted-ratio-1/README.md) | 27 | rejected |
| 2026-09-21 | [Directed third-cumulant head fails one held-network gate: next-mean ratio 0.916243773 versus 0.90](directed-third-cumulant-head-fails-one-held-network-gate/README.md) | 30 | rejected |
| 2026-09-21 | [Eight complete final-plane summaries add only about 1% beyond the affine oracle and miss both gates](eight-complete-final-plane-summaries-add-only-about-1/README.md) | 13 | rejected |
| 2026-09-21 | [Frozen 528-parameter nonlinear covariance correction passes two-network gate at ratio 0.993554950949](frozen-528-parameter-nonlinear-covariance-correction-passes-two-network/README.md) | 18 | inconclusive |
| 2026-09-21 | [Frozen joint third/fourth-moment correction worsens native adjusted error: pooled ratio 1.530216168430](frozen-joint-third-fourth-moment-correction-worsens-native-adjusted/README.md) | 33 | rejected |
| 2026-09-21 | [Frozen nonlinear covariance correction fails 14-network panel: ratio 1.009582713004, 5/14 wins](frozen-nonlinear-covariance-correction-fails-14-network-panel-ratio/README.md) | 18 | rejected |
| 2026-09-21 | [Frozen recurrent covariance decoder raises pooled adjusted error 19.7306%](frozen-recurrent-covariance-decoder-raises-pooled-adjusted-error-19/README.md) | 30 | rejected |
| 2026-09-21 | [Joint third/fourth-cumulant head misses both next-mean gates: ratios 0.917358747 and 0.931169619](joint-third-fourth-cumulant-head-misses-both-next-mean/README.md) | 23 | rejected |
| 2026-09-21 | [Nonlinear covariance-source adaptation improves covariance residuals but misses next-mean gates](nonlinear-covariance-source-adaptation-improves-covariance-residuals-but-misses/README.md) | 18 | rejected |
| 2026-09-21 | [Privileged cubic hidden-query fit yields adjusted ratios 0.824954528953 and 0.813778888455, above 0.80 gate](privileged-cubic-hidden-query-fit-yields-adjusted-ratios-0/README.md) | 12 | rejected |
| 2026-09-21 | [Sign-reversed terminal hidden features lose five-fold comparison: extended/reference ratio 1.01396566079](sign-reversed-terminal-hidden-features-lose-five-fold-comparison/README.md) | 39 | rejected |
| 2026-09-21 | [Signed-message fit fails native40 consistency gate: ratio 0.999179297214, 23/40 wins versus 24 required](signed-message-fit-fails-native40-consistency-gate-ratio-0/README.md) | 54 | rejected |
| 2026-09-21 | [Sixteen-coefficient covariance-innovation ridge fit misses both held-network response gates](sixteen-coefficient-covariance-innovation-ridge-fit-misses-both-held/README.md) | 12 | rejected |
| 2026-09-21 | [Refitting 45200 mean-encoder parameters on the covariance-edge estimator completes 160 graphs and 40 updates](refitting-45200-mean-encoder-parameters-covariance-edge-estimator-completes/README.md) | 33 | adopted |
| 2026-09-22 | [Adding learned graph-network capacity gives native adjusted ratio 1.0003 with 15/40 wins](adding-learned-graph-network-capacity-gives-native-adjusted-ratio/README.md) | 50 | rejected |
| 2026-09-22 | [Additional message encoder reduces raw error 0.05% but raises work 2.55%](additional-message-encoder-reduces-raw-error-0-05-but/README.md) | 51 | rejected |
| 2026-09-22 | [Final-objective readout on frozen pair features reduces raw error 1.03% but worsens adjusted error](final-objective-readout-frozen-pair-features-reduces-raw-error/README.md) | 53 | rejected |
| 2026-09-22 | [Dropping cohorts aged at least nine plus a joint refit still leaves 2.57-fold raw error](dropping-cohorts-aged-least-nine-plus-joint-refit-still/README.md) | 23 | rejected |
| 2026-09-22 | [Learned 64-by-48 source basis worsens native adjusted error: ratio 1.065 and 0/40 wins](learned-64-48-source-basis-worsens-native-adjusted-error/README.md) | 54 | rejected |
| 2026-09-23 | [Joint retuning of the unchanged closure worsens forty-network raw error by 2.1%](joint-retuning-unchanged-closure-worsens-forty-network-raw-error/README.md) | 22 | rejected |
| 2026-09-23 | [Per-unit cumulant heads explain only 5–9% of in-sample error variance and fail transfer](per-unit-cumulant-heads-explain-only-59-sample-error/README.md) | 25 | rejected |
| 2026-09-23 | [Whole-tier rank 256 or 288 with joint refit does not recover the lost accuracy](whole-tier-rank-256-or-288-joint-refit-does/README.md) | 25 | rejected |
| 2026-09-23 | [Joint refit with mean and edge outputs initially zero has no recorded validation verdict](joint-refit-mean-edge-outputs-initially-zero-has-no/README.md) | 22 | inconclusive |
| 2026-09-23 | [Refitting only learned heads after the plane and fourth-cumulant changes improves raw error 0.55%](refitting-only-learned-heads-after-plane-fourth-cumulant-changes/README.md) | 22 | rejected |
| 2026-09-23 | [Heads-only refit after gain and plane changes worsens sixteen-network validation error 0.13%](heads-only-refit-after-gain-plane-changes-worsens-sixteen/README.md) | 24 | rejected |
| 2026-09-23 | [Full closure refit after mathematical changes improves native raw error 0.50%](full-closure-refit-after-mathematical-changes-improves-native-raw/README.md) | 2 | adopted |
| 2026-09-23 | [Training message and encoder weights improves native raw error 1.001% with identical work](training-message-encoder-weights-improves-native-raw-error-1/README.md) | 1 | adopted |
| 2026-09-24 | [Auxiliary intermediate-mean loss ties final-only training: -0.006% on 40 networks](auxiliary-intermediate-mean-loss-ties-final-only-training-0/README.md) | 32 | rejected |
| 2026-09-24 | [Rank-320 trunk refit absorbs essentially none of the 3.35% raw-error loss](rank-320-trunk-refit-absorbs-essentially-none-3-35/README.md) | 29 | rejected |
| 2026-09-24 | [Expanded-data trunk refit follows the previous parameter direction more slowly](expanded-data-trunk-refit-follows-previous-parameter-direction-more/README.md) | 1 | informational |
| 2026-09-24 | [Twice-wide learned trunk and fourfold training data fail to beat the previous fit materially](twice-wide-learned-trunk-fourfold-training-data-fail-beat/README.md) | 29 | rejected |
| 2026-09-25 | [Half-strength born-unit plane feed plus refit improves 40-network raw error 2.05%](half-strength-born-unit-plane-feed-plus-refit-improves/README.md) | 30 | adopted |
| 2026-09-25 | [Leading final-covariance directions contain error but their coefficients are not predictable](leading-final-covariance-directions-contain-error-but-their-coefficients/README.md) | 28 | rejected |
| 2026-09-25 | [Born-unit feed strength 0.3 plus refit gains 1.65% but underperforms strength 0.5](born-unit-feed-strength-0-3-plus-refit-gains/README.md) | 33 | rejected |
| 2026-09-25 | [Second birth-feed refit epoch improves sixteen-network validation error 0.78% with 16/16 wins](second-birth-feed-refit-epoch-improves-sixteen-network-validation/README.md) | 31 | adopted |
| 2026-09-25 | [Refitting frozen learned-off physics dials remains 0.69% worse than the prior learned-on model](refitting-frozen-learned-off-physics-dials-remains-0-69/README.md) | 30 | rejected |
| 2026-09-25 | [Second physical-feed refit epoch improves all 40 held-out networks by 0.69%](second-physical-feed-refit-epoch-improves-all-40-held/README.md) | 30 | adopted |
| 2026-09-25 | [Refit on the new trajectory transfers to native validation: adjusted error improves 1.259%](refit-new-trajectory-transfers-native-validation-adjusted-error-improves/README.md) | 37 | adopted |
| 2026-09-25 | [Further refit after removing the residual-plane cross term gains only 0.331% and fails the 0.5% gate](further-refit-after-removing-residual-plane-cross-term-gains/README.md) | 43 | rejected |
| 2026-09-25 | [Joint learned-parameter and physics-dial GPU refit passes: CPU validation improves 0.525%](joint-learned-parameter-physics-dial-gpu-refit-passes-cpu/README.md) | 42 | adopted |
| 2026-09-25 | [Second joint refit epoch improves only 0.180% and fails its 0.5% gate](second-joint-refit-epoch-improves-only-0-180-fails/README.md) | 37 | rejected |
| 2026-09-25 | [Weight averaging between joint-refit epochs fails: all gains below 0.3%](weight-averaging-between-joint-refit-epochs-fails-all-gains/README.md) | 40 | rejected |
| 2026-09-25 | [Joint refit passes forty-network validation: raw error improves 0.543%](joint-refit-passes-forty-network-validation-raw-error-improves/README.md) | 39 | adopted |
| 2026-09-25 | [Doubling learned width adds only about 0.08%; extra training improves forty-network error 0.377%](doubling-learned-width-adds-only-about-0-08-extra/README.md) | 1 | adopted |
| 2026-09-25 | [Longer joint training disproves the earlier 0.06-0.12% remaining-gain projection](longer-joint-training-disproves-earlier-0-06-0-12/README.md) | 1 | informational |
| 2026-09-25 | [A further joint-refit block improves deployment-structure validation 0.430% but is superseded before submission](further-joint-refit-block-improves-deployment-structure-validation-0/README.md) | 39 | informational |
| 2026-09-25 | [Final plain joint-refit block improves deployment validation 0.206% and is selected before rank adaptation](final-plain-joint-refit-block-improves-deployment-validation-0/README.md) | 39 | adopted |
| 2026-09-25 | [Training at a reduced join rank fails to show specific co-adaptation](training-reduced-join-rank-fails-show-specific-co-adaptation/README.md) | 41 | rejected |
| 2026-09-25 | [Fresh cosine learning-rate blocks contribute more than extra epochs within each block](fresh-cosine-learning-rate-blocks-contribute-more-than-extra/README.md) | 40 | adopted |
| 2026-09-26 | [Refitted correction and final-layer pair-mean truncation: held-out adjusted score improves 2.3550%](refitted-correction-final-layer-pair-mean-truncation-held-out/README.md) | 10 | adopted |
| 2026-09-27 | [Current-state covariance-supervised pair model: both covariance and next-response gates fail](current-state-covariance-supervised-pair-model-both-covariance-next/README.md) | 10 | rejected |
| 2026-09-27 | [Deployed-edge off-diagonal mean projection: adjusted error rises 0.334747%](deployed-edge-off-diagonal-mean-projection-adjusted-error-rises/README.md) | 8 | rejected |
| 2026-09-27 | [Exact global pair-Gram innovation model: native adjusted error rises 9.049217%](exact-global-pair-gram-innovation-model-native-adjusted-error/README.md) | 65 | rejected |
| 2026-09-27 | [Final-objective nonlinear covariance pair head: training gain reverses on eight check networks](final-objective-nonlinear-covariance-pair-head-training-gain-reverses/README.md) | 22 | rejected |
| 2026-09-27 | [Fresh covariance-innovation learner: native adjusted error rises 6.5392%](fresh-covariance-innovation-learner-native-adjusted-error-rises-6/README.md) | 21 | rejected |
| 2026-09-27 | [Frozen covariance observer with physical feedback: native adjusted error rises 13.6731%](frozen-covariance-observer-physical-feedback-native-adjusted-error-rises/README.md) | 9 | rejected |
| 2026-09-27 | [Frozen joint head restricted to covariance: adjusted error rises 6.99703%](frozen-joint-head-restricted-covariance-adjusted-error-rises-6/README.md) | 11 | rejected |
| 2026-09-27 | [Frozen joint-moment head transferred to native recurrence: adjusted error rises 105.5486%](frozen-joint-moment-head-transferred-native-recurrence-adjusted-error/README.md) | 8 | rejected |
| 2026-09-27 | [Learned fourth-order matrix supplier: native adjusted error rises 9.763717%](learned-fourth-order-matrix-supplier-native-adjusted-error-rises/README.md) | 25 | rejected |
| 2026-09-27 | [Nonlinear covariance response-input fit: native held-out training subset worsens 0.143173%](nonlinear-covariance-response-input-fit-native-held-out-training/README.md) | 46 | rejected |
| 2026-09-27 | [Recurrent covariance-error observer: history worsens check-network raw error relative to reset](recurrent-covariance-error-observer-history-worsens-check-network-raw/README.md) | 10 | rejected |
| 2026-09-30 | [Eight continuation updates on 32 additional examples increase adjusted error 1.2985424828%](eight-continuation-updates-32-additional-examples-increase-adjusted-error/README.md) | 66 | rejected |
| 2026-09-30 | [Eight updates with detached star bases fail 40-network component confirmation](eight-updates-detached-star-bases-fail-40-network-component/README.md) | 81 | rejected |
| 2026-09-30 | [Eighty continuation updates with an unfrozen terminal head fail independent native confirmation](eighty-continuation-updates-unfrozen-terminal-head-fail-independent-native/README.md) | 76 | rejected |
| 2026-09-30 | [Four-update learned correction with native residual anchors: adjusted error rises 0.5246001601%](four-update-learned-correction-native-residual-anchors-adjusted-error/README.md) | 121 | rejected |
| 2026-09-30 | [Hidden mean projection after learned correction increases adjusted error 0.2580932%](hidden-mean-projection-after-learned-correction-increases-adjusted-error/README.md) | 29 | rejected |
| 2026-09-30 | [Learned edge bias hoisting lowers compute but increases adjusted error 0.0615985%](learned-edge-bias-hoisting-lowers-compute-but-increases-adjusted/README.md) | 31 | rejected |
| 2026-09-30 | [Nearest-grid edge activation lookup: adjusted error rises 0.1832807%](nearest-grid-edge-activation-lookup-adjusted-error-rises-0/README.md) | 132 | rejected |
| 2026-09-30 | [Rational degree-five edge activation: adjusted error rises 0.4237349499%](rational-degree-five-edge-activation-adjusted-error-rises-0/README.md) | 102 | rejected |
| 2026-09-30 | [Sixteen updates with detached star bases improve the mean but fail both confidence bounds](sixteen-updates-detached-star-bases-improve-mean-but-fail/README.md) | 48 | rejected |
| 2026-09-30 | [Three learned-correction energy mean closures: every native adjusted score worsens](three-learned-correction-energy-mean-closures-every-native-adjusted/README.md) | 107 | rejected |
| 2026-09-30 | [Ties-to-even edge activation lookup: adjusted error rises 0.12044676158629031%](ties-even-edge-activation-lookup-adjusted-error-rises-0/README.md) | 138 | rejected |
| 2026-09-30 | [Two updates with partial old-tail alignment increase native adjusted error 0.613160902283%](two-updates-partial-old-tail-alignment-increase-native-adjusted/README.md) | 72 | rejected |
| 2026-09-30 | [Two-update warm continuation: eight-network adjusted gain has confidence crossing zero](two-update-warm-continuation-eight-network-adjusted-gain-has/README.md) | 68 | rejected |
| 2026-09-30 | [Uniform learned-correction gain of 1.02 improves fitting error but regresses on eight checks](uniform-learned-correction-gain-1-02-improves-fitting-error/README.md) | 12 | rejected |
| 2026-10-01 | [Matched terminal-readout refits fail to recover the row-sum and diagonal kurtosis correction](matched-terminal-readout-refits-fail-recover-row-sum-diagonal/README.md) | 3 | rejected |
| 2026-10-01 | [Refitting the angular propagation’s learned correction improves teacher loss but fails native execution](refitting-angular-propagations-learned-correction-improves-teacher-loss-but/README.md) | 6 | rejected |
| 2026-10-01 | [Transferring the angular-trained mean correction to Gaussian propagation worsens native error](transferring-angular-trained-mean-correction-gaussian-propagation-worsens-native/README.md) | 1 | rejected |
| 2026-10-01 | [Adding 64 terminal hidden-state directions worsens every cross-validation fold](adding-64-terminal-hidden-state-directions-worsens-every-cross/README.md) | 3 | rejected |
| 2026-10-01 | [Predicted per-unit variance corrections worsen final-layer error by 1.2–5.3%](predicted-per-unit-variance-corrections-worsen-final-layer-error/README.md) | 25 | rejected |
| 2026-10-01 | [Continued training lowers held-out error by only 0.24–0.43%](continued-training-lowers-held-out-error-only-0-240/README.md) | 4 | rejected |
| 2026-10-01 | [Least-squares correction refits improve unseen error by 0.17–0.39%; richer readouts fit noise](least-squares-correction-refits-improve-unseen-error-0-170/README.md) | 44 | rejected |
| 2026-10-02 | [Scaling the learned mean correction to 0.8 lowers fresh-network error by 0.84%](scaling-learned-mean-correction-0-8-lowers-fresh-network/README.md) | 41 | adopted |
| 2026-10-02 | [Masked learned covariance correction lowers charged work by 0.69% for 0.02% additional error](masked-learned-covariance-correction-lowers-charged-work-0-69/README.md) | 44 | adopted |
| 2026-10-02 | [Skipping learned mean-correction messages for dead units raises error by 1.17%](skipping-learned-mean-correction-messages-dead-units-raises-error/README.md) | 43 | rejected |
| 2026-10-02 | [Refitting after fourth-order changes yields no reproducible improvement](refitting-after-fourth-order-changes-yields-no-reproducible-improvement/README.md) | 46 | rejected |
| 2026-10-02 | [Norm-coupling diagonal as a learned input lowers native MSE 2.15% on ten networks, all ten improved](norm-coupling-diagonal-learned-input-lowers-native-mse-2/README.md) | 4 | adopted |
| 2026-10-02 | [Left-factor row-size input rejected: 0.16% bill for a proxy of already weak learned-input signals](../../records/research/2026-10-02-left-factor-row-size-input-rejected-0-16-bill/README.md) | result only | rejected |
| 2026-10-02 | [Joint third step of three fourth-order input columns gives only 0.20% predicted gain, below noise](joint-third-step-three-fourth-order-input-columns-gives/README.md) | 2 | rejected |
| 2026-10-02 | [Learned-array step after adding the diagonal input lowers MSE 0.160%; larger step raises it 5.847%](learned-array-step-after-adding-diagonal-input-lowers-mse/README.md) | 44 | inconclusive |
| 2026-10-02 | [Learned-array step on the pair-law estimator lowers MSE 0.158%; larger step raises it 0.654%](learned-array-step-pair-law-estimator-lowers-mse-0/README.md) | 44 | inconclusive |
| 2026-10-10 | [Row-gain, carried-weight and radial readouts rejected after line searches](row-gain-carried-weight-radial-readouts-rejected-after-line/README.md) | 5 | rejected |
| 2026-10-10 | [Layer-15 weight refit rejected: +1.69% held-out MSE despite -1.77% in sample](layer-15-weight-refit-rejected-1-69-held-out/README.md) | 40 | rejected |
| 2026-10-10 | [Previous-state per-unit fourth- and third-cumulant readouts show no zero-weight signal](previous-state-per-unit-fourth-third-cumulant-readouts-show/README.md) | 7 | rejected |
| 2026-10-10 | [Continuation refit: -0.455% versus the preceding point on 48 clean ranks; a second step is rejected](continuation-refit-0-455-versus-preceding-point-48-clean/README.md) | 8 | adopted |
| 2026-10-10 | [Banded continuation refit rejected: -0.313% sizing MSE but -0.121% leave-one-network-out](banded-continuation-refit-rejected-0-313-sizing-mse-but/README.md) | 8 | rejected |
| 2026-10-10 | [Propagated learned correction is not a useful head input: error-fit R^2 <= 0.024](propagated-learned-correction-not-useful-head-input-error-fit/README.md) | 37 | rejected |
| 2026-10-10 | [Joint refit accepts three independent-check steps; fourth step rejected at +0.016% MSE](joint-refit-accepts-three-independent-check-steps-fourth-step/README.md) | 8 | adopted |
| 2026-10-10 | [Banded head weights and additional columns rejected: -0.018% on 48 clean ranks; quartic fold costs +1.343%](banded-head-weights-additional-columns-rejected-0-018-48/README.md) | 6 | rejected |
| 2026-10-11 | [First cut-simulated refit rejected: independent MSE change -0.235%, standard error 0.340](first-cut-simulated-refit-rejected-independent-mse-change-0/README.md) | 108 | rejected |
| 2026-10-11 | [All-layer-loss refit rejected: late-layer residual prediction removes less than 0.1% of error power](all-layer-loss-refit-rejected-late-layer-residual-prediction/README.md) | 1 | rejected |
| 2026-10-11 | [Early-layer covariance edge and a 16-network refit rejected: +0.831% FLOPs and +0.197% independent MSE](early-layer-covariance-edge-16-network-refit-rejected-0/README.md) | 1 | rejected |
| 2026-10-12 | [Re-sizing previously fitted arms gives no fresh-network signal; two-stage pruning fails its stability rule](re-sizing-previously-fitted-arms-gives-no-fresh-network/README.md) | 1 | rejected |
| 2026-10-13 | [Small refit rounds rejected for native insignificance or independent-check failure](small-refit-rounds-rejected-native-insignificance-or-independent-check/README.md) | 1 | rejected |
| 2026-10-14 | [Learned energy-message replacements: fresh-48 adjusted -0.183%/-0.340%, native delivery parity fails](learned-energy-message-replacements-fresh-48-adjusted-0-183/README.md) | 6 | rejected |
| 2026-10-14 | [Correction-only learned factorization: import failure followed by displacement RMS 0.0320, rejected](correction-only-learned-factorization-import-failure-followed-displacement-rms/README.md) | 10 | rejected |
| 2026-10-14 | [Signed second-message conditional mean with optional rank-two residual: native parity delivery fails](signed-second-message-conditional-mean-optional-rank-two-residual/README.md) | 3 | rejected |
| 2026-10-14 | [Standalone conditional energy mean: fresh-32 adjusted -0.156%, native parity delivery fails](standalone-conditional-energy-mean-fresh-32-adjusted-0-156/README.md) | 4 | rejected |
| 2026-10-14 | [Refit-after-drop round 57: held-out adjusted -0.033% on 15 of 32 networks, rejected](refit-after-drop-round-57-held-out-adjusted-0/README.md) | 57 | rejected |
| 2026-10-14 | [Refit-after-drop round 58 with reduced arms: held-out adjusted +0.086% on 11 of 32 networks, rejected](refit-after-drop-round-58-reduced-arms-held-out/README.md) | 57 | rejected |
| 2026-10-14 | [Refit-after-drop round 59 from the round-57 point: held-out adjusted -0.008%, rejected](refit-after-drop-round-59-round-57-point-held/README.md) | 57 | rejected |
| 2026-10-14 | [Principal-component compression of learned contractions to rank two: bill -0.31%, fresh-16 adjusted -0.245%](principal-component-compression-learned-contractions-rank-two-bill-0/README.md) | 1 | inconclusive |
| 2026-10-14 | [Three-coefficient output calibration (y, mean y, y^2/mean y) fitted on 16 networks: LOO -0.78% but fresh +0.88%, closed](three-coefficient-output-calibration-y-mean-y-y-2/README.md) | 2 | rejected |
| 2026-10-14 | [Rank-two learned-message compression with a shared second message: two-network adjusted -0.29%](rank-two-learned-message-compression-shared-second-message-two/README.md) | 25 | inconclusive |
| 2026-10-14 | [Antipodal projection of the final-layer learned head (odd component onto W^T state): two-network adjusted +0.14%, closed](antipodal-projection-final-layer-learned-head-odd-component-onto/README.md) | 1 | rejected |
| 2026-10-14 | [Rank-two output projection of the learned correction: fresh-16 adjusted -0.577%, out-of-sample -0.560%](rank-two-output-projection-learned-correction-fresh-16-adjusted/README.md) | 18 | adopted |
| 2026-10-14 | [Refit-after-drop round 60: held-out adjusted -0.003%; two unproductive rounds end the line](refit-after-drop-round-60-held-out-adjusted-0/README.md) | 57 | inconclusive |
| 2026-10-14 | [One subspace iteration on the rank-two learned output projection: adjusted +0.28%](one-subspace-iteration-rank-two-learned-output-projection-adjusted/README.md) | 1 | rejected |
| 2026-10-14 | [Output-sum (mass) restoration on the rank-2 output projection: two-network -0.07% but pilot-8 adjusted +0.03%, closed](output-sum-mass-restoration-rank-2-output-projection-two/README.md) | 2 | rejected |
| 2026-10-14 | [Conditional-energy log correction on rank-two output projection: adjusted +0.23%](conditional-energy-log-correction-rank-two-output-projection-adjusted/README.md) | 1 | rejected |
| 2026-10-14 | [Short refit of the learned correction on 8 networks after the output projection: held-out MSE -0.014% (4/8), parked](short-refit-learned-correction-8-networks-after-output-projection/README.md) | 1 | rejected |
| 2026-10-14 | [Encoder-weighted output projection and triangle terms: adjusted +0.04%/+0.18%](encoder-weighted-output-projection-triangle-terms-adjusted-0-04/README.md) | 2 | rejected |
| 2026-10-15 | [Linear (94-feature ridge) and nonlinear (94-16-1) fits of the terminal variance correction fail held-out networks](linear-94-feature-ridge-nonlinear-94-16-1-fits/README.md) | 1 | rejected |
| 2026-10-15 | [Refit-after-drop rounds 61 and 62 select zero by the declared leave-one-out criterion](refit-after-drop-rounds-61-62-select-zero-declared/README.md) | 64 | rejected |
| 2026-10-15 | [Native terminal linear ridge on 32 networks: blocked cross-validation selects zero](native-terminal-linear-ridge-32-networks-blocked-cross-validation/README.md) | 29 | rejected |
| 2026-10-15 | [Four terminal physical-interaction ridge features: cross-validation selects zero](four-terminal-physical-interaction-ridge-features-cross-validation-selects/README.md) | 1 | rejected |
| 2026-10-15 | [Terminal 16-hidden-unit nonlinear fit: all-24 adjusted -0.036487% but out-of-sample +0.005041%](terminal-16-hidden-unit-nonlinear-fit-all-24-adjusted/README.md) | 5 | inconclusive |
| 2026-10-15 | [Terminal global context plus nonlinear fit: native pilot adjusted +0.541490% despite cross-validation -0.373141%](terminal-global-context-plus-nonlinear-fit-native-pilot-adjusted/README.md) | 30 | rejected |
| 2026-10-15 | [Native terminal 254-feature hidden ridge: cross-validation -0.0337414%, retained without export](native-terminal-254-feature-hidden-ridge-cross-validation-0/README.md) | 27 | inconclusive |
| 2026-10-15 | [Warm-started terminal nonlinear adaptation: cross-validation -0.024432%, no native export](warm-started-terminal-nonlinear-adaptation-cross-validation-0-024432/README.md) | 1 | inconclusive |
| 2026-10-15 | [Old-read projection distilled to the parent predictions: cross-validation adjusted +1.30503%](old-read-projection-distilled-parent-predictions-cross-validation-adjusted/README.md) | 28 | rejected |
| 2026-10-15 | [Intermediate mean ridge at layers 7-14: local cross-validation -0.395022%, rollout adjusted +0.900108%](intermediate-mean-ridge-layers-7-14-local-cross-validation/README.md) | 34 | rejected |
| 2026-10-15 | [Intermediate mean translation preserving centered state: pilot-8 adjusted +1.130840%](intermediate-mean-translation-preserving-centered-state-pilot-8-adjusted/README.md) | 29 | rejected |
| 2026-10-15 | [Innovation-target and joint propagated-feature ridge fits fail final-objective cross-validation](innovation-target-joint-propagated-feature-ridge-fits-fail-final/README.md) | 2 | rejected |
| 2026-10-15 | [Nonlinear output-only fit: cross-validation -0.074624% adjusted, only two of eight folds improve](nonlinear-output-only-fit-cross-validation-0-074624-adjusted/README.md) | 1 | rejected |
| 2026-10-15 | [Pair-aware residual-energy feature 23: pilot adjusted +0.473696%](pair-aware-residual-energy-feature-23-pilot-adjusted-0/README.md) | 39 | rejected |
| 2026-10-15 | [Rank-four terminal encoder adaptation: cross-validation MSE +0.035696%](rank-four-terminal-encoder-adaptation-cross-validation-mse-0/README.md) | 1 | rejected |
| 2026-10-15 | [Terminal hidden rank-one direction adaptation: cross-validation -0.000052626%](terminal-hidden-rank-one-direction-adaptation-cross-validation-0/README.md) | 1 | rejected |
| 2026-10-15 | [Read-tail feature 24 restoration: pilot adjusted +0.346987%](read-tail-feature-24-restoration-pilot-adjusted-0-346987/README.md) | 36 | rejected |
| 2026-10-15 | [Terminal sign-half averaging: saved MSE +0.031673%, contrast oracle below 0.16%](terminal-sign-half-averaging-saved-mse-0-031673-contrast/README.md) | 4 | rejected |
| 2026-10-15 | [Exact signed-pair row-energy correction costs +0.297006% bill](exact-signed-pair-row-energy-correction-costs-0-297006/README.md) | 10 | rejected |
| 2026-10-15 | [Shared native innovation target: 0.20914% oracle gain shrinks to 0.008689% with the terminal surrogate](shared-native-innovation-target-0-20914-oracle-gain-shrinks/README.md) | 1 | rejected |
| 2026-10-15 | [Paid-prefix row-energy feature 22: all-24 adjusted -0.162470%, out-of-sample -0.145096%](paid-prefix-row-energy-feature-22-all-24-adjusted/README.md) | 31 | adopted |
| 2026-10-16 | [Native final-layer head ridge refit: held-out 49-network adjusted +0.016%, control-24 +0.504%](native-final-layer-head-ridge-refit-held-out-49/README.md) | 8 | rejected |
| 2026-10-16 | [Final-aware layer-14 head refit: held-out -0.088%, control-24 +0.173%](final-aware-layer-14-head-refit-held-out-0/README.md) | 5 | rejected |
| 2026-10-16 | [Learned covariance edge cut deeper (off, hard clip, tight rosters): each cut costs 1.4-2x its bill; off +1.31% MSE](learned-covariance-edge-cut-deeper-off-hard-clip-tight/README.md) | 1 | rejected |
| 2026-10-16 | [Learned edge restricted to layers 10-12: -0.23% net on 8 seeds, +0.62% MSE on 21 store networks, killed](learned-edge-restricted-layers-10-12-0-23-net/README.md) | 2 | rejected |
| 2026-10-18 | [Early capacity growth with source calibration: all three fixed variants lost adjusted score](early-capacity-growth-source-calibration-all-three-fixed-variants/README.md) | 3 | rejected |
| 2026-10-18 | [Halfway early source-coefficient shrinkage: initial -0.801293% shrank to -0.28319665% on seven](halfway-early-source-coefficient-shrinkage-initial-0-801293-shrank/README.md) | 3 | rejected |
| 2026-10-18 | [Shared-array gradient step improved training by 1.04% but only 0.049587% on untouched checks](shared-array-gradient-step-improved-training-1-04-but/README.md) | 56 | rejected |
| 2026-10-18 | [Late-layer encoder gradients on 32 training networks failed the prespecified global signal gate](late-layer-encoder-gradients-32-training-networks-failed-prespecified/README.md) | 57 | rejected |
| 2026-10-18 | [Layer-7 rank-cut distillation failed both a fixed gradient point and four-direction least squares](layer-7-rank-cut-distillation-failed-both-fixed-gradient/README.md) | 62 | rejected |
| 2026-10-18 | [Learned gate response: +0.124420% adjusted with zero wins on seven native networks](learned-gate-response-0-124420-adjusted-zero-wins-seven/README.md) | 3 | rejected |
| 2026-10-18 | [Fixed retained-cumulant gate: -0.1448% adjusted on seven, below the continuation threshold](fixed-retained-cumulant-gate-0-1448-adjusted-seven-below/README.md) | 3 | inconclusive |
| 2026-10-18 | [Quadratic retained-cumulant gate lost 0.100038% adjusted on seven networks](quadratic-retained-cumulant-gate-lost-0-100038-adjusted-seven/README.md) | 2 | rejected |
| 2026-10-19 | [Learned covariance edge off at layers 8, 9 or 14: +0.085% to +0.133% adjusted on 25-28 ranks; it stays on at every layer](learned-covariance-edge-off-layers-8-9-or-14/README.md) | 3 | rejected |
| 2026-10-20 | [Terminal 64-state ridge readout refit: leave-one-network-out MSE +0.06..+0.33%](terminal-64-state-ridge-readout-refit-leave-one-network/README.md) | 4 | rejected |
| 2026-10-20 | [Thirty per-unit features cannot predict late pre-variance errors across networks](thirty-per-unit-features-cannot-predict-late-pre-variance/README.md) | 1 | rejected |
| 2026-10-20 | [Forty-five cheap readout features across layers 10-15: best leave-one-network-out gain -0.022%](forty-five-cheap-readout-features-across-layers-10-15/README.md) | 2 | rejected |
| 2026-10-20 | [Joint native readout refit at layers 12-15: -0.66% in sample, nonnegative leave-one-network-out error](joint-native-readout-refit-layers-12-15-0-66/README.md) | 1 | rejected |
| 2026-10-20 | [Variance head inputs give leave-one-network-out R^2 <= +0.01 at every layer and ridge](variance-head-inputs-give-leave-one-network-out-r/README.md) | 1 | rejected |
| 2026-10-20 | [Radical learned-edge approximation: -0.1676% adjusted on four fresh networks](radical-learned-edge-approximation-0-1676-adjusted-four-fresh/README.md) | 5 | adopted |
| 2026-10-20 | [Learned-edge channels 9 and 14 pruned: fresh gain exceeds full-32 gain; pruning channel 9 alone fails](learned-edge-channels-9-14-pruned-fresh-gain-exceeds/README.md) | 2 | adopted |
| 2026-10-20 | [Cheaper square-root activation: +.294248% adjusted versus its exact pruned-edge parent](cheaper-square-root-activation-294248-adjusted-versus-exact-pruned/README.md) | 2 | rejected |
| 2026-10-20 | [Eight-channel message distillation with loader repair: -.6774163425% Mac32 versus #337393](eight-channel-message-distillation-loader-repair-6774163425-mac32-versus/README.md) | 6 | adopted |
| 2026-10-20 | [Message width six and nonlinear encoder compression fail to improve the eight-channel parent](message-width-six-nonlinear-encoder-compression-fail-improve-eight/README.md) | 13 | rejected |
| 2026-10-20 | [Second hidden width 64 distillation: +.2093449737% Mac32; old trajectory invalidates self-distillation premise](second-hidden-width-64-distillation-2093449737-mac32-old-trajectory/README.md) | 6 | rejected |
| 2026-10-21 | [Learned edge hidden width eight: -.150665691% Mac and -.2338808% fresh adjusted versus #337422](learned-edge-hidden-width-eight-150665691-mac-2338808-fresh/README.md) | 4 | adopted |
| 2026-10-21 | [Learned edge hidden width six: -.169319149% Mac adjusted with less consistent fresh improvement](learned-edge-hidden-width-six-169319149-mac-adjusted-less/README.md) | 4 | rejected |
| 2026-10-21 | [Current-trajectory rank-16/24 affine compression: small gains vanish in the width-eight edge stack](current-trajectory-rank-16-24-affine-compression-small-gains/README.md) | 3 | rejected |
| 2026-10-21 | [Restoring original sixteen-channel projectors after message compression worsens Mac32 +0.2620721%](restoring-original-sixteen-channel-projectors-after-message-compression-worsens/README.md) | 5 | rejected |
| 2026-10-21 | [Seven-dimensional final-teacher scale repair is rejected before native validation](seven-dimensional-final-teacher-scale-repair-rejected-before-native/README.md) | 21 | rejected |
| 2026-10-21 | [Layer-specific first-affine teacher fits overfit; shared fit passes a prespecified fidelity gate](layer-specific-first-affine-teacher-fits-overfit-shared-fit/README.md) | 7 | rejected |
| 2026-10-21 | [Higher projector ranks, shared first affine and energy-balanced join selection fail full-32 tests](higher-projector-ranks-shared-first-affine-energy-balanced-join/README.md) | 11 | rejected |
| 2026-10-21 | [Per-layer edge hidden width four: +5.393798% adjusted; width two already +10.315041% on seven networks](per-layer-edge-hidden-width-four-5-393798-adjusted/README.md) | 7 | rejected |
| 2026-10-21 | [Four-network shared local-error readout: mixed host results and no useful gain in edge stack](four-network-shared-local-error-readout-mixed-host-results/README.md) | 10 | rejected |
| 2026-10-21 | [Four-network terminal-response fit: -1.467658% training proxy but +0.873410% leave-one-out](four-network-terminal-response-fit-1-467658-training-proxy/README.md) | 17 | rejected |
| 2026-10-21 | [Mean-only energy carriers trade -.111633% FLOPs for +.107325% raw MSE](mean-only-energy-carriers-trade-111633-flops-107325-raw/README.md) | 4 | rejected |
| 2026-10-21 | [Eight-dimensional true-terminal edge fit: leave-one-out +0.029263%, no candidate exported](eight-dimensional-true-terminal-edge-fit-leave-one-out/README.md) | 5 | rejected |
| 2026-10-21 | [Hidden-feature variance screen: leave-one-out variance R^2 .01094 but mean error +1.06234%](hidden-feature-variance-screen-leave-one-out-variance-r/README.md) | 5 | rejected |
| 2026-10-21 | [Authentic hidden-width-64 teacher compression saves .167264% projected FLOPs but fails all five folds](authentic-hidden-width-64-teacher-compression-saves-167264-projected/README.md) | 5 | rejected |
| 2026-10-21 | [Ninety-network terminal-response proxy fits: shared +0.040428%, three-band +0.109454% leave-one-out](ninety-network-terminal-response-proxy-fits-shared-0-040428/README.md) | 4 | rejected |
| 2026-10-21 | [Current-message edge-context refit and current-consumer projector metric give no useful native gain](current-message-edge-context-refit-current-consumer-projector-metric/README.md) | 8 | rejected |
| 2026-10-21 | [Per-layer readout fitted on 90 networks: -.371029209% Mac32 and -.235745959% Linux32](per-layer-readout-fitted-90-networks-371029209-mac32-235745959/README.md) | 15 | adopted |
| 2026-10-21 | [Width-eight edge plus per-layer 90-network readout: -.533265988% Mac32 and -.441676584% fresh](width-eight-edge-plus-per-layer-90-network-readout/README.md) | 23 | adopted |
| 2026-10-21 | [Readout intercept and physical-loss alternatives miss the fixed -.15% proxy improvement threshold](readout-intercept-physical-loss-alternatives-miss-fixed-15-proxy/README.md) | 3 | rejected |
| 2026-10-21 | [Current Gram metric for energy features: +.065519076% Mac32, no promotion](current-gram-metric-energy-features-065519076-mac32-no-promotion/README.md) | 5 | rejected |
| 2026-10-21 | [Hardclip and quadratic edge approximations fail fixed 1% teacher-fidelity gates](hardclip-quadratic-edge-approximations-fail-fixed-1-teacher-fidelity/README.md) | 8 | rejected |
| 2026-10-21 | [Hidden complement readout adds .001660192% FLOPs and improves both host judges](hidden-complement-readout-adds-001660192-flops-improves-both-host/README.md) | 14 | adopted |
| 2026-10-21 | [Four exact native readout directions fail curvature; half-radius successor misses leave-one-out gate](four-exact-native-readout-directions-fail-curvature-half-radius/README.md) | 9 | rejected |
| 2026-10-21 | [Readout adaptation after pair contraction fails conditional local-error recovery gates](readout-adaptation-after-pair-contraction-fails-conditional-local-error/README.md) | 10 | rejected |
| 2026-10-21 | [Width-eight edge plus hidden complement: -.191409749% Mac32 incremental over #337466](width-eight-edge-plus-hidden-complement-191409749-mac32-incremental/README.md) | 8 | adopted |
| 2026-10-21 | [Forty-nine reconstructed gate/message/energy skip features: -.045224116% incremental proxy, below gate](forty-nine-reconstructed-gate-message-energy-skip-features-045224116/README.md) | 11 | rejected |
| 2026-10-21 | [Squared second-hidden readout: -.258929085% Mac32 and -.317014687% Linux32 versus #337466](squared-second-hidden-readout-258929085-mac32-317014687-linux32-versus/README.md) | 19 | rejected |
| 2026-10-21 | [Twenty-nine physical readout features miss the fixed proxy gate; first-128 reconstruction passes](twenty-nine-physical-readout-features-miss-fixed-proxy-gate/README.md) | 11 | rejected |
| 2026-10-21 | [First-128 plus squared second-hidden readout: -.068997650% proxy improvement, below the fixed threshold](first-128-plus-squared-second-hidden-readout-068997650-proxy/README.md) | 9 | rejected |
| 2026-10-21 | [First-hidden squared readout: -.084706254% proxy improvement, below fixed -.15% threshold](first-hidden-squared-readout-084706254-proxy-improvement-below-fixed/README.md) | 9 | rejected |
| 2026-10-21 | [First-hidden width 128 to 64 compression fails every teacher fold with 15.7-30.3% RMS errors](first-hidden-width-128-64-compression-fails-every-teacher/README.md) | 6 | rejected |
| 2026-10-21 | [Encoder row scales and offsets: +43.0785360494% cross-validation error, zero of 90 wins](encoder-row-scales-offsets-43-0785360494-cross-validation-error/README.md) | 6 | rejected |
