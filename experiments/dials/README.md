# Dials

Scalar coefficients set by finite-difference line search (GSK, BFIX gamma, SS beta, NP strength, K4L_C, k4 bands, row-cut z and others).

| Date | Experiment | Programs | Outcome |
|---|---|---:|---|
| 2026-09-10 | [Refitting birth coefficients yields 3/4 wins but worsens mean adjusted score by 0.1128042679%](refitting-birth-coefficients-yields-3-4-wins-but-worsens/README.md) | 117 | rejected |
| 2026-09-23 | [Raising the response-row floor saves arithmetic but worsens adjusted error](raising-response-row-floor-saves-arithmetic-but-worsens-adjusted/README.md) | 1 | rejected |
| 2026-09-23 | [Covariance error value flows through the pair map; zero-cost Gauss-Newton knobs improve held-out error](covariance-error-value-flows-through-pair-map-zero-cost/README.md) | 24 | adopted |
| 2026-09-23 | [Extra kink-row carrier scales add no useful gain beyond the covariance knobs](extra-kink-row-carrier-scales-add-no-useful-gain/README.md) | 26 | rejected |
| 2026-09-23 | [Layer-banded covariance knobs fail: only 5/12 held-out wins and -0.37% pooled error](layer-banded-covariance-knobs-fail-only-5-12-held/README.md) | 27 | rejected |
| 2026-09-23 | [Refitting covariance knobs after learned coefficients gives only 0.09% pooled gain](refitting-covariance-knobs-after-learned-coefficients-gives-only-0/README.md) | 27 | rejected |
| 2026-09-25 | [Joint physics fit with learned corrections off gains 10.06% but is mostly redundant when restored](joint-physics-fit-learned-corrections-off-gains-10-06/README.md) | 29 | inconclusive |
| 2026-09-25 | [Learned-on joint physics dials improve held-out raw error 2.23% without a refit](learned-joint-physics-dials-improve-held-out-raw-error/README.md) | 30 | adopted |
| 2026-09-25 | [Consistent-semantics physics refit improves raw error 1.98% on forty low-rank residual-plane networks](consistent-semantics-physics-refit-improves-raw-error-1-98/README.md) | 40 | adopted |
| 2026-09-25 | [Alternating learned-on dial fit and refit improves held-out error 0.89% in the first round](alternating-learned-dial-fit-refit-improves-held-out-error/README.md) | 30 | adopted |
| 2026-09-25 | [Second alternating dial fit fails: validation +0.106% and leave-one-out +0.44%](second-alternating-dial-fit-fails-validation-0-106-leave/README.md) | 38 | rejected |
| 2026-09-25 | [Third dial-fit round on the Nyström structure is null: sixteen-network validation +0.032%](third-dial-fit-round-nystrom-structure-null-sixteen-network/README.md) | 29 | rejected |
| 2026-09-25 | [Isotropic (3,1)-plane coefficient scan is flat near 0.65-0.69; no new dial change](isotropic-3-1-plane-coefficient-scan-flat-near-0/README.md) | 35 | rejected |
| 2026-09-26 | [Old-cohort carry multiplier contrast: nonlinear response fails the fixed opportunity gate](old-cohort-carry-multiplier-contrast-nonlinear-response-fails-fixed/README.md) | 5 | rejected |
| 2026-09-27 | [Covariance convention and correction strength: three short screens stay below the 1% gate](covariance-convention-correction-strength-three-short-screens-stay-below/README.md) | 9 | rejected |
| 2026-09-27 | [Trace-vector depth tilt: finite responses are nonlinear and predicted gain is below threshold](trace-vector-depth-tilt-finite-responses-nonlinear-predicted-gain/README.md) | 6 | rejected |
| 2026-09-28 | [Birth-fix gamma raised to 0.125: initial gain vanishes after eight-network expansion](birth-fix-gamma-raised-0-125-initial-gain-vanishes/README.md) | 28 | rejected |
| 2026-09-28 | [Eleven physical dials fitted from 144 native forwards: leave-one-out gain is negative](eleven-physical-dials-fitted-144-native-forwards-leave-one/README.md) | 47 | rejected |
| 2026-09-28 | [Five-scalar native Gauss–Newton fit: leave-one-out gain is negative](five-scalar-native-gaussnewton-fit-leave-one-out-gain/README.md) | 9 | rejected |
| 2026-09-30 | [Early three-one coefficient changes of ±5%: adjusted error rises in both directions](early-three-one-coefficient-changes-5-adjusted-error-rises/README.md) | 83 | rejected |
| 2026-09-30 | [Late three-one coefficient scaled to 95% or 90%: neither fixed screen qualifies](late-three-one-coefficient-scaled-95-or-90-neither/README.md) | 77 | rejected |
| 2026-09-30 | [Native three-dial joint fit: predicted gain 0.14259635011146798% misses the 0.5% gate](native-three-dial-joint-fit-predicted-gain-0-14259635011146798/README.md) | 40 | rejected |
| 2026-09-30 | [Negative trace-vector depth contrast fails 40-network component confirmation after eight-check gains](negative-trace-vector-depth-contrast-fails-40-network-component/README.md) | 45 | rejected |
| 2026-09-30 | [Three archived fitted correction dials improve eight checks but regress on 40-network confirmation](three-archived-fitted-correction-dials-improve-eight-checks-but/README.md) | 42 | rejected |
| 2026-10-01 | [Fitting one kurtosis-correction amplitude fails to realize its predicted error reduction](fitting-one-kurtosis-correction-amplitude-fails-realize-predicted-error/README.md) | 17 | rejected |
| 2026-10-01 | [A fitted mean-annihilating covariance component improves four training networks but fails eight held-out networks](fitted-mean-annihilating-covariance-component-improves-four-training-networks/README.md) | 16 | rejected |
| 2026-10-01 | [Fitting the young negative-tail trace repair amplitude gives no held-out improvement](fitting-young-negative-tail-trace-repair-amplitude-gives-no/README.md) | 13 | rejected |
| 2026-10-01 | [Reducing recentering strength below 0.92 worsens held-out error](reducing-recentering-strength-below-0-92-worsens-held-out/README.md) | 24 | rejected |
| 2026-10-01 | [Joint fitting of 91 per-layer dials improves training error but fails leave-one-network-out validation](joint-fitting-91-per-layer-dials-improves-training-error/README.md) | 25 | rejected |
| 2026-10-02 | [Birth-source scale 0.85 helps without fourth-cumulant shifts and harms with them](birth-source-scale-0-85-helps-without-fourth-cumulant/README.md) | 41 | rejected |
| 2026-10-02 | [Off-diagonal fourth-order covariance scale is too sensitive for a usable dial](off-diagonal-fourth-order-covariance-scale-too-sensitive-usable/README.md) | 43 | rejected |
| 2026-10-02 | [Birth-dial perturbations show strong sensitivity to new-state scale and little gain elsewhere](birth-dial-perturbations-show-strong-sensitivity-new-state-scale/README.md) | 40 | rejected |
| 2026-10-02 | [Gradient-directed dial line search lowers error by 1.44% on 50 held-out networks](gradient-directed-dial-line-search-lowers-error-1-44/README.md) | 2 | adopted |
| 2026-10-02 | [Joint least-squares fit of five dials improves held-out error by 0.70%](joint-least-squares-fit-five-dials-improves-held-out/README.md) | 5 | adopted |
| 2026-10-02 | [Network features fail to predict useful dial gradients beyond a weak permutation-null result](network-features-fail-predict-useful-dial-gradients-beyond-weak/README.md) | 2 | rejected |
| 2026-10-02 | [Joint norm-coupling dial solve lowers MSE 1.48% on 25 ranks and enters submission #333587](joint-norm-coupling-dial-solve-lowers-mse-1-48/README.md) | 46 | adopted |
| 2026-10-02 | [Second norm-coupling dial solve lowers research MSE 1.03% on 25 ranks; native benefit disappears](second-norm-coupling-dial-solve-lowers-research-mse-1/README.md) | 1 | rejected |
| 2026-10-02 | [Young- and old-plane read scales jointly fitted: final MSE falls 0.226% on 25 ranks](young-old-plane-read-scales-jointly-fitted-final-mse/README.md) | 46 | inconclusive |
| 2026-10-02 | [Second joint solve of estimator dials lowers MSE 0.403% on 25 ranks, below two standard errors](second-joint-solve-estimator-dials-lowers-mse-0-403/README.md) | 45 | rejected |
| 2026-10-02 | [Separate norm-coupling strengths on kink and other rows have no improvement over strength one](separate-norm-coupling-strengths-kink-other-rows-have-no/README.md) | 38 | rejected |
| 2026-10-10 | [Global dials near optimum after refit; remaining late-band signals require per-layer constants](global-dials-near-optimum-after-refit-remaining-late-band/README.md) | 1 | rejected |
| 2026-10-10 | [Fixed-constant audit: centers and scales are near noise; edge-array gradients fail independent acceptance](fixed-constant-audit-centers-scales-near-noise-edge-array/README.md) | 38 | rejected |
| 2026-10-13 | [Kink-, dead-unit and top-mode features fail to predict per-network dial gradients](kink-dead-unit-top-mode-features-fail-predict-per/README.md) | 2 | rejected |
| 2026-10-13 | [Carrier and second-order re-screens rejected; a -0.019% plane-strength result remains unconfirmed](carrier-second-order-re-screens-rejected-0-019-plane/README.md) | 2 | rejected |
| 2026-10-16 | [Transport-gate kurtosis strength 0.6: fresh-eight noise-free MSE -0.072% at identical FLOPs](transport-gate-kurtosis-strength-0-6-fresh-eight-noise/README.md) | 7 | adopted |
| 2026-10-16 | [Learned correction damping after the rank-384 join: four-network MSE +0.556/+0.171%](learned-correction-damping-after-rank-384-join-four-network/README.md) | 12 | rejected |
| 2026-10-16 | [Increasing learned covariance and higher-order strengths worsens the current native estimator](increasing-learned-covariance-higher-order-strengths-worsens-current-native/README.md) | 11 | rejected |
| 2026-10-16 | [Reduced strength and kink-row third-moment dials all worsen final accuracy](reduced-strength-kink-row-third-moment-dials-all-worsen/README.md) | 13 | rejected |
| 2026-10-16 | [Nonlinear-pair and birth/output scalar dials: no eight-network result clears the -0.15% criterion](nonlinear-pair-birth-output-scalar-dials-no-eight-network/README.md) | 8 | rejected |
| 2026-10-16 | [Nonlinear-pair strength reduced by 8%: 24-network adjusted +0.092%](nonlinear-pair-strength-reduced-8-24-network-adjusted-0/README.md) | 5 | rejected |
| 2026-10-16 | [Covariance and fourth-order dial steps from an earlier refit fail on the current submitted source](covariance-fourth-order-dial-steps-earlier-refit-fail-current/README.md) | 5 | rejected |
| 2026-10-16 | [Birth-fix and learned-edge dials on submission #336412: neither improved the confirmed final error](birth-fix-learned-edge-dials-submission-336412-neither-improved/README.md) | 1 | rejected |
| 2026-10-16 | [Store dial sweep of 39 zero-bill arms (gate, skew, kurtosis, birth, plane scales, late grows): nothing beyond noise](store-dial-sweep-39-zero-bill-arms-gate-skew/README.md) | 5 | rejected |
| 2026-10-17 | [Four fixed coefficients for gate, birth, skew and star terms improve 32-network raw error by 0.295%](four-fixed-coefficients-gate-birth-skew-star-terms-improve/README.md) | 4 | adopted |
| 2026-10-18 | [Six-scalar gradient direction had only 0.06984% noise-corrected quadratic upside](six-scalar-gradient-direction-had-only-0-06984-noise/README.md) | 55 | rejected |
| 2026-10-19 | [Old and young read scales, core scale and star coefficient: small perturbations do not improve error](old-young-read-scales-core-scale-star-coefficient-small/README.md) | 7 | rejected |
| 2026-10-19 | [Combined zero-cost dial optima: -0.19% raw error on 28 networks](combined-zero-cost-dial-optima-0-19-raw-error/README.md) | 3 | adopted |
| 2026-10-20 | [15 fitted zero-bill dials re-centred by finite differences (pair-law strength, star beta, row factors): all within 0.05%](15-fitted-zero-bill-dials-re-centred-finite-differences/README.md) | 19 | rejected |
| 2026-10-20 | [Six-network late-term screen: all fourth-order, terminal and variance-floor changes rejected](six-network-late-term-screen-all-fourth-order-terminal/README.md) | 1 | rejected |
| 2026-10-20 | [Mean-consistency strength and late fourth-order row factors: full-32 tests reject apparent six-network gains](mean-consistency-strength-late-fourth-order-row-factors-full/README.md) | 4 | rejected |
| 2026-10-20 | [Terminal variance scale shifts of 0.5% increase MSE +210..+290%; cumulant scales remain near optimum](terminal-variance-scale-shifts-0-5-increase-mse-210/README.md) | 1 | rejected |
| 2026-10-20 | [Late isotropic correction x1.2 / x0.8: +0.019% / -0.033% adjusted on 32 networks](late-isotropic-correction-x1-2-x0-8-0-019/README.md) | 2 | rejected |
| 2026-10-20 | [Combined late learned gain, isotropic scale and terminal skew scale: +0.055% adjusted](combined-late-learned-gain-isotropic-scale-terminal-skew-scale/README.md) | 2 | rejected |
| 2026-10-20 | [Learned mean-shift covariance coefficient x1.2 / x0.8: +0.082% / +0.010% adjusted](learned-mean-shift-covariance-coefficient-x1-2-x0-8/README.md) | 2 | rejected |
| 2026-10-20 | [Late mean-shift covariance coefficient x0.8: -0.0843% adjusted; x0.6 reverses the gain](late-mean-shift-covariance-coefficient-x0-8-0-0843/README.md) | 3 | rejected |
| 2026-10-20 | [Joint early covariance profile and eight scalar corrections: corrected six-network baseline +0.02763%](joint-early-covariance-profile-eight-scalar-corrections-corrected-six/README.md) | 12 | rejected |
| 2026-10-20 | [Early third/fourth-order scalar pair and fourth-order age coefficient give only small full-32 gains](early-third-fourth-order-scalar-pair-fourth-order-age/README.md) | 3 | rejected |
| 2026-10-20 | [Single scalar gain reverses inside both exact-copy and fourth-order stacks](single-scalar-gain-reverses-inside-both-exact-copy-fourth/README.md) | 2 | rejected |
| 2026-10-20 | [Prespecified fourth-order scalar 1.14: -.263896% fresh increment but worse than parent on 32 networks](prespecified-fourth-order-scalar-1-14-263896-fresh-increment/README.md) | 7 | rejected |
| 2026-10-21 | [Fourth-cumulant strength gains on the base estimator disappear on the fitted readout](fourth-cumulant-strength-gains-base-estimator-disappear-fitted-readout/README.md) | 8 | rejected |
| 2026-10-21 | [Late plane strengths and fourth-order band changes remain near zero or worsen the deployed source](late-plane-strengths-fourth-order-band-changes-remain-near/README.md) | 4 | rejected |
| 2026-10-21 | [Late coefficient retuning fails to add to the final readout; the extra submission is withheld](late-coefficient-retuning-fails-add-final-readout-extra-submission/README.md) | 4 | rejected |
