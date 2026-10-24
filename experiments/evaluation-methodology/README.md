# Evaluation methodology

How candidates were measured locally and how local numbers transfer to the hosted grader: paired stores, cross-half truth, fresh cohorts, torch versus native.

| Date | Experiment | Programs | Outcome |
|---|---|---:|---|
| 2026-09-09 | [First-head covariance attribution fails two of four fixed vector/scalar response cells](first-head-covariance-attribution-fails-two-four-fixed-vector/README.md) | 52 | rejected |
| 2026-09-12 | [Shared rank-384 old basis: independent-14 adjusted 3.793391755502277e-9; native/client parity failed](shared-rank-384-old-basis-independent-14-adjusted-3/README.md) | 269 | inconclusive |
| 2026-09-16 | [Partial cumulant corrections break closure cancellation: rank1 truth correction makes final MSE 6x worse](partial-cumulant-corrections-break-closure-cancellation-rank1-truth-correction/README.md) | 1 | rejected |
| 2026-09-19 | [Exact-cost depth selection passes 14-network panel: adjusted ratio .954075238, score 3.9806136484e-9](exact-cost-depth-selection-passes-14-network-panel-adjusted/README.md) | 7 | adopted |
| 2026-09-21 | [Refit mean encoder on the 40-network validation split: adjusted ratio 0.9848676992236052, 32/40 wins at equal compute](refit-mean-encoder-40-network-validation-split-adjusted-ratio/README.md) | 39 | adopted |
| 2026-09-21 | [Refit mean encoder passed all downstream gates: cold14 ratio 0.9787744692559065, independent public8 0.9719722866477218](refit-mean-encoder-passed-all-downstream-gates-cold14-ratio/README.md) | 17 | adopted |
| 2026-09-24 | [Dead-side cuts, exact column reuse and deeper reads pass the relaxed adjusted-score gate](dead-side-cuts-exact-column-reuse-deeper-reads-pass/README.md) | 6 | informational |
| 2026-09-25 | [Per-network choice among existing variants has oracle gain at most 0.19%](per-network-choice-among-existing-variants-has-oracle-gain/README.md) | 1 | rejected |
| 2026-09-25 | [Native held-out evaluation confirms the single intermediate-rank cut](native-held-out-evaluation-confirms-single-intermediate-rank-cut/README.md) | 4 | adopted |
| 2026-09-25 | [Depth-seven arithmetic with the latest refit passes a forty-network candidate gate](depth-seven-arithmetic-latest-refit-passes-forty-network-candidate/README.md) | 2 | informational |
| 2026-09-26 | [Combined join-core and star-diagonal changes: all 40 held-out networks improve adjusted score](combined-join-core-star-diagonal-changes-all-40-held/README.md) | 3 | adopted |
| 2026-09-27 | [Native final-objective innovation-readout response: all four stability cells fail](native-final-objective-innovation-readout-response-all-four-stability/README.md) | 8 | rejected |
| 2026-09-28 | [Birth-strength selection and star residual repair combined: validation reverses training gain](birth-strength-selection-star-residual-repair-combined-validation-reverses/README.md) | 12 | rejected |
| 2026-09-28 | [Rank-32 star bundle: 40-network adjusted gain is 1.793775% with 39 wins](rank-32-star-bundle-40-network-adjusted-gain-1/README.md) | 21 | adopted |
| 2026-09-29 | [Training/native forward qualification: Jacobian response discrepancy 0.523 fails the gate](training-native-forward-qualification-jacobian-response-discrepancy-0-523/README.md) | 35 | rejected |
| 2026-09-30 | [Native-style training join alignment leaves the final prediction gap unchanged](native-style-training-join-alignment-leaves-final-prediction-gap/README.md) | 38 | rejected |
| 2026-09-30 | [Partial old-tail training alignment reduces the surrogate-to-native gap to 0.783927093427](partial-old-tail-training-alignment-reduces-surrogate-native-gap/README.md) | 41 | inconclusive |
| 2026-10-13 | [Third-order row-factor gradient fails out of sample because gradient replay freezes carrier states](third-order-row-factor-gradient-fails-out-sample-because/README.md) | 8 | rejected |
| 2026-10-21 | [Paid first-128 candidate confirms -.591104203% adjusted on 12 known fresh networks](paid-first-128-candidate-confirms-591104203-adjusted-12-known/README.md) | 10 | adopted |
