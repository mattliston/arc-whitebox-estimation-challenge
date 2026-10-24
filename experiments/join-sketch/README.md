# Join sketch

Randomised low-rank sketches of the joined plane columns: the Nystrom join sketch, the power-step sketch, kc, join truncation.

| Date | Experiment | Programs | Outcome |
|---|---|---:|---|
| 2026-09-12 | [Shared rank384 old-factor basis on network 37 improves adjusted error: ratio 0.905199269759271](shared-rank384-old-factor-basis-network-37-improves-adjusted/README.md) | 167 | adopted |
| 2026-09-12 | [Shared rank-384 old basis on four networks: adjusted ratio 0.9249351388935633, all four improve](shared-rank-384-old-basis-four-networks-adjusted-ratio/README.md) | 175 | adopted |
| 2026-09-12 | [Shared rank-384 old basis on 40 networks: adjusted ratio 0.9191502236702563, 40/40 improve](shared-rank-384-old-basis-40-networks-adjusted-ratio/README.md) | 179 | adopted |
| 2026-09-25 | [Value-weighted Nyström row selection loses to the existing row rule](value-weighted-nystrom-row-selection-loses-existing-row-rule/README.md) | 35 | rejected |
| 2026-09-25 | [Increasing the final join rank beyond 384 costs more than its accuracy gain](increasing-final-join-rank-beyond-384-costs-more-than/README.md) | 40 | rejected |
| 2026-09-26 | [Eigenspace Gram compression: raw error rises 3.616% or 18.598% at reduced ranks](eigenspace-gram-compression-raw-error-rises-3-616-or/README.md) | 31 | rejected |
| 2026-09-26 | [Future-basis Nyström core approximation: native bill falls 0.7069% and 0.7196%](future-basis-nystrom-core-approximation-native-bill-falls-0/README.md) | 2 | adopted |
| 2026-09-26 | [Inherited coefficient scale restored in the basis-selection core increases adjusted error 0.0253%](inherited-coefficient-scale-restored-basis-selection-core-increases-adjusted/README.md) | 5 | rejected |
| 2026-09-26 | [Interpolatory Nyström joins at original ranks: raw error rises 5.352%](interpolatory-nystrom-joins-original-ranks-raw-error-rises-5/README.md) | 31 | rejected |
| 2026-09-26 | [Interpolatory Nyström joins at rank 384: compute savings are outweighed by raw error](interpolatory-nystrom-joins-rank-384-compute-savings-outweighed-raw/README.md) | 4 | rejected |
| 2026-09-26 | [Interpolatory Nyström ranks 448/384/448: wider joins consume the arithmetic savings](interpolatory-nystrom-ranks-448-384-448-wider-joins-consume/README.md) | 4 | rejected |
| 2026-09-26 | [Residual-diagonal pivot blocks of 32 or 64: no gain over original row selection](residual-diagonal-pivot-blocks-32-or-64-no-gain/README.md) | 31 | rejected |
| 2026-09-26 | [Symmetric blocks at earlier join cores save about 0.07% compute with rounding changes](symmetric-blocks-earlier-join-cores-save-about-0-07/README.md) | 2 | inconclusive |
| 2026-09-27 | [Earlier three-join schedules: adjusted error rises 4.438081% and 4.285327%](earlier-three-join-schedules-adjusted-error-rises-4-438081/README.md) | 8 | rejected |
| 2026-09-27 | [Final join shifted to layer 12 or 14: early gain reverses after expansion](final-join-shifted-layer-12-or-14-early-gain/README.md) | 8 | rejected |
| 2026-09-28 | [First join allocates 96 rows to signed-pair difference energy: gain fails two-win gate](first-join-allocates-96-rows-signed-pair-difference-energy/README.md) | 29 | rejected |
| 2026-09-28 | [Layer-10 shared join rank 256: adjusted error rises 9.132025366%](layer-10-shared-join-rank-256-adjusted-error-rises/README.md) | 4 | rejected |
| 2026-09-28 | [Precision bundle plus core carry: slightly worse than the precision parent](precision-bundle-plus-core-carry-slightly-worse-than-precision/README.md) | 17 | rejected |
| 2026-09-30 | [Exact weighted Gram core at join seven: adjusted error rises 0.2107815947%](exact-weighted-gram-core-join-seven-adjusted-error-rises/README.md) | 98 | rejected |
| 2026-09-30 | [First join at layer six widened to rank 448: adjusted error still rises 0.44565962%](first-join-layer-six-widened-rank-448-adjusted-error/README.md) | 80 | rejected |
| 2026-09-30 | [First join moved to layer six at rank 384: adjusted error rises 0.65277694%](first-join-moved-layer-six-rank-384-adjusted-error/README.md) | 77 | rejected |
| 2026-09-30 | [Layer-ten tail Ritz basis: fixed 256 common and 64 tail directions increase adjusted error 0.5415039616%](layer-ten-tail-ritz-basis-fixed-256-common-64/README.md) | 32 | rejected |
| 2026-09-30 | [Nyström columns kept in energy order: adjusted error rises 0.8603447002%](nystrom-columns-kept-energy-order-adjusted-error-rises-0/README.md) | 79 | rejected |
| 2026-09-30 | [Nyström core ridge at 1e-6 and 1e-5: both native screens fail](nystrom-core-ridge-1e-6-1e-5-both-native/README.md) | 79 | rejected |
| 2026-09-30 | [Nyström selected-block symmetry with reuse: eight-network confidence and win gates fail](nystrom-selected-block-symmetry-reuse-eight-network-confidence-win/README.md) | 83 | rejected |
| 2026-09-30 | [One-pivot Schur residual join selection increases adjusted error 1.1359041476%](one-pivot-schur-residual-join-selection-increases-adjusted-error/README.md) | 30 | rejected |
| 2026-09-30 | [Rank-64 response basis before the join loses its two-network gain on eight networks](rank-64-response-basis-before-join-loses-two-network/README.md) | 34 | rejected |
| 2026-10-02 | [Rank-32 sketched norm-coupling carrier: two passes lower MSE 0.41% on 50 ranks; one pass raises it 4.50%](rank-32-sketched-norm-coupling-carrier-two-passes-lower/README.md) | 40 | rejected |
| 2026-10-10 | [Layer-10 join rank 384 with refitting rejected: accuracy gain approximately equals FLOP cost](layer-10-join-rank-384-refitting-rejected-accuracy-gain/README.md) | 43 | rejected |
| 2026-10-10 | [Fixed-rank join-sketch refinements rejected: exact eigenvectors give -0.80% MSE for about 9% more FLOPs](fixed-rank-join-sketch-refinements-rejected-exact-eigenvectors-give/README.md) | 38 | rejected |
| 2026-10-10 | [Join-sketch eigenvalue weighting rejected: best change -0.17% misses the -0.4% acceptance threshold](join-sketch-eigenvalue-weighting-rejected-best-change-0-17/README.md) | 38 | rejected |
| 2026-10-11 | [Join schedules rejected in both directions: early joins add 10.67% MSE; later joins cost more than they gain](join-schedules-rejected-both-directions-early-joins-add-10/README.md) | 42 | rejected |
| 2026-10-12 | [Native capacity retests reject rank 384 and rank-8 star: adjusted +0.378% and +0.454%](native-capacity-retests-reject-rank-384-rank-8-star/README.md) | 6 | rejected |
| 2026-10-13 | [Rank redistribution rejected: restoring layer-10 rank 384 gives +0.148% net out-of-selection change](rank-redistribution-rejected-restoring-layer-10-rank-384-gives/README.md) | 121 | rejected |
| 2026-10-13 | [Early-read rank variant costs +1.225% native FLOPs versus the copy baseline; screen remains pending](early-read-rank-variant-costs-1-225-native-flops/README.md) | 79 | inconclusive |
| 2026-10-14 | [Common-u basis reordered by within-span eigen-energy at layer 10: two-network -0.96% but fresh-16 net +0.37%, closed](common-u-basis-reordered-within-span-eigen-energy-layer/README.md) | 2 | rejected |
| 2026-10-14 | [Residual Nystrom join-core repair: fresh-16 adjusted -0.12%, out-of-sample-16 +0.44%](residual-nystrom-join-core-repair-fresh-16-adjusted-0/README.md) | 4 | rejected |
| 2026-10-14 | [Exact core diagonal at the first join with ranks 320/320/384: first network adjusted +3.20% (bill -1.20%), closed early](exact-core-diagonal-first-join-ranks-320-320-384/README.md) | 1 | rejected |
| 2026-10-15 | [Join rows ordered by descending carrier energy before the QR at layers 10/13 (zero bill): fixed-2 -0.26%, pilot-8 -0.12%](join-rows-ordered-descending-carrier-energy-before-qr-layers/README.md) | 1 | inconclusive |
| 2026-10-15 | [Energy-ordered join rows with rank-32 common tail: pilot-8 incremental adjusted +0.11%](energy-ordered-join-rows-rank-32-common-tail-pilot/README.md) | 1 | rejected |
| 2026-10-15 | [Join rows from 16 top anchors plus conditional Gram-residual selection (layers 10/13): pilot-8 -0.42%, fresh-16 +0.25%](join-rows-16-top-anchors-plus-conditional-gram-residual/README.md) | 3 | rejected |
| 2026-10-15 | [Exact paired cubic mode-1 metric for the join rows: pilot-8 adjusted +0.23% (MSE +0.16%, bill +0.07%), closed](exact-paired-cubic-mode-1-metric-join-rows-pilot/README.md) | 2 | rejected |
| 2026-10-15 | [Exact diagonal of the first rank-384 join core: pilot adjusted +0.002635%](exact-diagonal-first-rank-384-join-core-pilot-adjusted/README.md) | 27 | rejected |
| 2026-10-15 | [First-join residual-row selector: pilot adjusted +0.096659%](first-join-residual-row-selector-pilot-adjusted-0-096659/README.md) | 33 | rejected |
| 2026-10-15 | [Layer-10 join rank 384: initial single-network adjusted +2.64766%, later evidence requires reconsideration](layer-10-join-rank-384-initial-single-network-adjusted/README.md) | 40 | rejected |
| 2026-10-16 | [Layer-10 join rank 384: 24-network noise-free adjusted -0.461%, final eight +0.173%](layer-10-join-rank-384-24-network-noise-free/README.md) | 2 | adopted |
| 2026-10-16 | [Layer-7 join rank 448: four-network adjusted +0.74% despite MSE -0.91%](layer-7-join-rank-448-four-network-adjusted-0/README.md) | 12 | rejected |
| 2026-10-16 | [Rank-384 join plus kurtosis strength 0.6: fresh-24 noise-free adjusted -0.199%, aggregate -0.096%](rank-384-join-plus-kurtosis-strength-0-6-fresh/README.md) | 7 | adopted |
| 2026-10-16 | [Join rank moved from layer 13 to layer 10: adjusted -0.073% with +400 requests](join-rank-moved-layer-13-layer-10-adjusted-0/README.md) | 12 | rejected |
| 2026-10-16 | [One subspace iteration at all joins: MSE -1.370% but adjusted +3.885%](one-subspace-iteration-all-joins-mse-1-370-but/README.md) | 7 | rejected |
| 2026-10-16 | [Block-greedy join pivots: pilot adjusted -0.760%, but request risk 30.95%](block-greedy-join-pivots-pilot-adjusted-0-760-but/README.md) | 8 | adopted |
| 2026-10-16 | [One-shot residual-diagonal pivot refinement: adjusted +0.162% on base and +1.520% on rank-384 stack](one-shot-residual-diagonal-pivot-refinement-adjusted-0-162/README.md) | 7 | rejected |
| 2026-10-16 | [Static-plus-one adaptive 128-column join block: pilot adjusted -0.475%, request risk 19.64%](static-plus-one-adaptive-128-column-join-block-pilot/README.md) | 8 | adopted |
| 2026-10-16 | [Rank cuts under greedy joins: layer 7 adjusted +0.01%, layer 13 +0.43%](rank-cuts-under-greedy-joins-layer-7-adjusted-0/README.md) | 8 | rejected |
| 2026-10-16 | [Static plus one 256-column greedy block: eight-network adjusted -0.655%](static-plus-one-256-column-greedy-block-eight-network/README.md) | 6 | adopted |
| 2026-10-16 | [Extra or wider join schedules under adaptive pivots fail: late absorption adjusted +1.62%](extra-or-wider-join-schedules-under-adaptive-pivots-fail/README.md) | 3 | rejected |
| 2026-10-16 | [Per-join adaptive pivot budgets: best tested mix adjusted -0.14%, below the -0.25% criterion](per-join-adaptive-pivot-budgets-best-tested-mix-adjusted/README.md) | 3 | rejected |
| 2026-10-16 | [Oversampled Nystrom with eigen truncation: +1.875/+3.075% bill for 64/128 extra pivots](oversampled-nystrom-eigen-truncation-1-875-3-075-bill/README.md) | 4 | rejected |
| 2026-10-16 | [Rank 384 on the adaptive-256 join with own prebind refit: adjusted +0.187%](rank-384-adaptive-256-join-own-prebind-refit-adjusted/README.md) | 7 | rejected |
| 2026-10-16 | [Density-weighted greedy pivots: MSE -0.141/-0.217% with standard errors above 1%](density-weighted-greedy-pivots-mse-0-141-0-217/README.md) | 4 | rejected |
| 2026-10-16 | [Static pivots restored at layer 7 on the block-greedy join: 24-network adjusted +0.111%](static-pivots-restored-layer-7-block-greedy-join-24/README.md) | 4 | rejected |
| 2026-10-16 | [Wide-candidate 32-block greedy join: 24-network adjusted +0.919% versus block-greedy reference](wide-candidate-32-block-greedy-join-24-network-adjusted/README.md) | 5 | rejected |
| 2026-10-16 | [Wide-candidate 128-block greedy join: adjusted +0.46%; column ordering changes the common read](wide-candidate-128-block-greedy-join-adjusted-0-46/README.md) | 5 | rejected |
| 2026-10-16 | [Layer-13 join ranks 416 and 448: adjusted +0.148% and +0.232%](layer-13-join-ranks-416-448-adjusted-0-148/README.md) | 2 | rejected |
| 2026-10-16 | [Warm-started join subspace from the old basis and heavy young columns fails saved-Gram accuracy](warm-started-join-subspace-old-basis-heavy-young-columns/README.md) | 12 | rejected |
| 2026-10-16 | [Oversampled join with eigen truncation: raw gains -0.356 to -1.298% are canceled by bill](oversampled-join-eigen-truncation-raw-gains-0-356-1/README.md) | 12 | rejected |
| 2026-10-16 | [Block-32 oversampled joins: 24-network MSE -0.582%, adjusted +0.068%](block-32-oversampled-joins-24-network-mse-0-582/README.md) | 16 | rejected |
| 2026-10-16 | [Earlier cohort absorption remains harmful with near-exact join subspace: MSE +1.407%](earlier-cohort-absorption-remains-harmful-near-exact-join-subspace/README.md) | 12 | rejected |
| 2026-10-16 | [Block-32 oversampling only at layer 7: pooled adjusted -0.042%, independent confirmation +0.021%](block-32-oversampling-only-layer-7-pooled-adjusted-0/README.md) | 4 | rejected |
| 2026-10-16 | [Eigen-truncating the 384-row sketch to rank 352 loses more energy at every captured join](eigen-truncating-384-row-sketch-rank-352-loses-more/README.md) | 12 | rejected |
| 2026-10-16 | [Interpolative old-tier representation: three to 3.8 times the lost join energy despite 2.5% bill saving](interpolative-old-tier-representation-three-3-8-times-lost/README.md) | 1 | rejected |
| 2026-10-16 | [Energy-ordered join rows pass 24-network retest: adjusted -0.236% at zero bill](energy-ordered-join-rows-pass-24-network-retest-adjusted/README.md) | 6 | inconclusive |
| 2026-10-16 | [Join core numerics not a lever: float64 solve byte-identical, float64 QR rounding only, Rayleigh-Ritz core +0.87% bill](join-core-numerics-not-lever-float64-solve-byte-identical/README.md) | 6 | rejected |
| 2026-10-16 | [Task-aware Nystrom join rows (kink-weighted energy): -0.455% on 8 seeds, +0.213% on 32 store networks, removed](task-aware-nystrom-join-rows-kink-weighted-energy-0/README.md) | 9 | rejected |
| 2026-10-16 | [Join rank cuts and grows at layers 7, 10, 13 (320, 256, 448): cuts lose more accuracy than bill, grows pay under 1:1](join-rank-cuts-grows-layers-7-10-13-320/README.md) | 3 | rejected |
| 2026-10-17 | [Join landmark selectors on saved joins (QR interpolation, one-pivot, core diagonal, pair rotation): no consistent gain](join-landmark-selectors-saved-joins-qr-interpolation-one-pivot/README.md) | 5 | rejected |
| 2026-10-17 | [Power-step join sketches failed under fast-kernel rounding despite better offline bases](power-step-join-sketches-failed-under-fast-kernel-rounding/README.md) | 3 | rejected |
| 2026-10-18 | [Power-step sketch only at layer 7 gave +0.01% adjusted; later errors reversed the local improvement](power-step-sketch-only-layer-7-gave-0-01/README.md) | 3 | rejected |
| 2026-10-18 | [Join sketch on 3072/2048/2048 heavy columns: 32-network adjusted -1.431% with unchanged raw error](join-sketch-3072-2048-2048-heavy-columns-32-network/README.md) | 2 | informational |
| 2026-10-18 | [Oversampled Nyström eigen-truncation on heavy-column sketches bought no mean final accuracy](oversampled-nystrom-eigen-truncation-heavy-column-sketches-bought-no/README.md) | 2 | rejected |
| 2026-10-18 | [Join sketch on 2048/1536/1536 heavy columns: 32-network adjusted and bill both -1.653%](join-sketch-2048-1536-1536-heavy-columns-32-network/README.md) | 2 | adopted |
| 2026-10-18 | [Light-column factors on 320 of 384 frame coordinates increased plane error by 33%](light-column-factors-320-384-frame-coordinates-increased-plane/README.md) | 1 | rejected |
| 2026-10-18 | [Heavy-column sketch squeezed to 1536/1024/1024: +0.04% adjusted versus the submitted plan](heavy-column-sketch-squeezed-1536-1024-1024-0-04/README.md) | 2 | rejected |
| 2026-10-18 | [Rank-64 join residual correction with old-core reuse: +0.351% adjusted on two native networks](rank-64-join-residual-correction-old-core-reuse-0/README.md) | 6 | rejected |
| 2026-10-18 | [Pair-mean-only join selection retained full signed content but lost 1.520% adjusted](pair-mean-only-join-selection-retained-full-signed-content/README.md) | 3 | rejected |
| 2026-10-18 | [Both-sided signed rank-64 join repair: +0.585% adjusted on two native networks](both-sided-signed-rank-64-join-repair-0-585/README.md) | 6 | rejected |
| 2026-10-18 | [Restoring omitted inherited-core trace had negligible and inconsistent signed-read benefit](restoring-omitted-inherited-core-trace-had-negligible-inconsistent-signed/README.md) | 2 | rejected |
| 2026-10-18 | [Join-sketch tail completion by energy strata or omitted-row diagonal worsened all six saved joins](join-sketch-tail-completion-energy-strata-or-omitted-row/README.md) | 1 | rejected |
| 2026-10-18 | [Layer-7 join rank 320 saved 0.836096% bill but added cold requests and 0.608336% raw error](layer-7-join-rank-320-saved-0-836096-bill/README.md) | 29 | rejected |
| 2026-10-18 | [Nyström landmark-Gram symmetry cost only 0.01315% bill savings after actual kernel pricing](nystrom-landmark-gram-symmetry-cost-only-0-01315-bill/README.md) | 1 | rejected |
| 2026-10-18 | [Layer-13 tensor-mass landmark selection improved saved fidelity but lost 0.292914% adjusted natively](layer-13-tensor-mass-landmark-selection-improved-saved-fidelity/README.md) | 18 | rejected |
| 2026-10-18 | [Symmetrizing the Nyström landmark block before QR improved the held stack to -0.248556% adjusted](symmetrizing-nystrom-landmark-block-before-qr-improved-held-stack/README.md) | 1 | inconclusive |
| 2026-10-19 | [Nystrom join core solved in float64 or row-equilibrated float32: adjusted +0.41% and +0.01%; core condition 4.5-5.9](nystrom-join-core-solved-float64-or-row-equilibrated-float32/README.md) | 3 | rejected |
| 2026-10-19 | [Join-sketch columns reduced to 1536/1024/1024: -0.390% adjusted; smaller counts lose raw accuracy](join-sketch-columns-reduced-1536-1024-1024-0-390/README.md) | 5 | adopted |
| 2026-10-19 | [Join rank at layer 13 held at 384: ranks 320, 352, 448 and 512 all worse adjusted (+0.50% to +0.97%)](join-rank-layer-13-held-384-ranks-320-352/README.md) | 4 | rejected |
| 2026-10-20 | [Join rank at layer 10 increased to 512: +3.05% adjusted for -0.93% MSE and +4.00% FLOPs](join-rank-layer-10-increased-512-3-05-adjusted/README.md) | 1 | rejected |
| 2026-10-20 | [Per-row join-frame energy repair increases MSE +0.19% / +0.04% and adds about 1.0% FLOPs](per-row-join-frame-energy-repair-increases-mse-0/README.md) | 14 | rejected |
| 2026-10-20 | [Layer-7 join rank 320: +1.183% adjusted; rank 352 saves effectively no FLOPs](layer-7-join-rank-320-1-183-adjusted-rank/README.md) | 3 | rejected |
| 2026-10-20 | [Additional first join at layer 5: -0.95% adjusted on one fresh network and +2.7% on another](additional-first-join-layer-5-0-95-adjusted-one/README.md) | 12 | rejected |
| 2026-10-20 | [Join-sketch oversampling at 32/128 and 64/128: +0.5735% / +0.8255% adjusted on 32 networks](join-sketch-oversampling-32-128-64-128-0-5735/README.md) | 5 | rejected |
| 2026-10-20 | [Join-sketch oversampling 64/192: first-seven +1.627310% versus #337393; full-32 test reprioritized](join-sketch-oversampling-64-192-first-seven-1-627310/README.md) | 8 | rejected |
| 2026-10-21 | [Signed pair-energy join selection: -.063740% full-32 adjusted, too weak and uncertain for adoption](signed-pair-energy-join-selection-063740-full-32-adjusted/README.md) | 4 | rejected |
| 2026-10-21 | [Join-sketch column counts around the deployed settings: +0.175%, +0.011% and +0.110% adjusted](join-sketch-column-counts-around-deployed-settings-0-175/README.md) | 3 | rejected |
