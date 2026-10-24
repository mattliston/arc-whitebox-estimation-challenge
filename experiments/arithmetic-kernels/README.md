# Arithmetic kernels

Exact kernels that lower the metered FLOP count: Strassen and Winograd products, grouped (Smirnov) reads, symmetric Gram, float32 transports, padding rules.

| Date | Experiment | Programs | Outcome |
|---|---|---:|---|
| 2026-08-26 | [Cached unchecked depth-1 third-cumulant arena: measured and rejected](cached-unchecked-depth-1-third-cumulant-arena-measured-rejected/README.md) | 10 | rejected |
| 2026-08-26 | [Depth-1 third-cumulant arena deployed in submission #328586](depth-1-third-cumulant-arena-deployed-submission-328586/README.md) | 1 | adopted |
| 2026-08-26 | [Preallocated depth-4 Winograd arena: 0.04961721133440733 s extra residual exceeded the 0.010 s gate](preallocated-depth-4-winograd-arena-0-04961721133440733-s-extra/README.md) | 46 | rejected |
| 2026-08-26 | [Unpacked depth-5 arena memory boundary: rejected in this form](unpacked-depth-5-arena-memory-boundary-rejected-this-form/README.md) | 23 | rejected |
| 2026-08-27 | [Splitting arena aliases saved only 0.683 ms while allocating 2,320,941,056 extra bytes](splitting-arena-aliases-saved-only-0-683-ms-while/README.md) | 9 | rejected |
| 2026-08-29 | [Late-layer error attribution vs offline Monte Carlo truth: verified](late-layer-error-attribution-offline-monte-carlo-truth-verified/README.md) | 2 | informational |
| 2026-08-30 | [Third-cumulant rank cuts sharply worsened raw error while fixed arena costs persisted](third-cumulant-rank-cuts-sharply-worsened-raw-error-while/README.md) | 5 | informational |
| 2026-09-01 | [Compute decomposition found fixed arena shapes obstructed proposed rank-cap schedules](compute-decomposition-found-fixed-arena-shapes-obstructed-proposed-rank/README.md) | 8 | informational |
| 2026-09-03 | [Derived plane and residual-carry host reached raw 2.17e-8 at a larger compute bill](derived-plane-residual-carry-host-reached-raw-2-17e/README.md) | 4 | rejected |
| 2026-09-03 | [Spherical cubature at width 1024 was evaluated against the analytic closure](spherical-cubature-width-1024-evaluated-against-analytic-closure/README.md) | 1 | informational |
| 2026-09-04 | [Age-dependent low-rank cohort schedule doubled raw error without saving compute against the arena](age-dependent-low-rank-cohort-schedule-doubled-raw-error/README.md) | 1 | rejected |
| 2026-09-04 | [Billed low-rank source-cohort carrier preserved full-rank parity but compressed forms lost accuracy](billed-low-rank-source-cohort-carrier-preserved-full-rank/README.md) | 4 | rejected |
| 2026-09-04 | [Direct third-cumulant products confirmed rank-cap error was physical; float16 offered no billing saving](direct-third-cumulant-products-confirmed-rank-cap-error-physical/README.md) | 1 | informational |
| 2026-09-04 | [Fourth-order plane-only transport missed 25–32% of the two-pair and about 95% of triple-single content](fourth-order-plane-only-transport-missed-2532-two-pair/README.md) | 7 | informational |
| 2026-09-04 | [Frozen skew-scale table recovered 34–53% of the true-skew correction value](frozen-skew-scale-table-recovered-3453-true-skew-correction/README.md) | 5 | informational |
| 2026-09-04 | [Low-rank third-cumulant plane carry lost the distinct tensor content required by transport](low-rank-third-cumulant-plane-carry-lost-distinct-tensor/README.md) | 8 | rejected |
| 2026-09-04 | [Organizer cumulant propagation at width 1024 exposed precision and memory differences](organizer-cumulant-propagation-width-1024-exposed-precision-memory-differences/README.md) | 1 | informational |
| 2026-09-04 | [Pure Gaussian closure remained inaccurate after co-adapted learning](pure-gaussian-closure-remained-inaccurate-after-co-adapted-learning/README.md) | 8 | rejected |
| 2026-09-04 | [Retaining only young third-cumulant cohorts increased host error 3.6 to 7.5 times](retaining-only-young-third-cumulant-cohorts-increased-host-error/README.md) | 4 | informational |
| 2026-09-04 | [Third- and fourth-order rank cuts: rank-256 quartic carrier saved compute after co-adaptation](third-fourth-order-rank-cuts-rank-256-quartic-carrier/README.md) | 19 | informational |
| 2026-09-04 | [Variance correction required fine fourth-order plane structure and self-consistent application](variance-correction-required-fine-fourth-order-plane-structure-self/README.md) | 4 | informational |
| 2026-09-04 | [With true fourth-order planes, accumulated skew error remained 4.5–6.5% at layers 6–14](true-fourth-order-planes-accumulated-skew-error-remained-4/README.md) | 1 | informational |
| 2026-09-05 | [Bounded radial and factor-thinning variants failed their local accuracy gates](bounded-radial-factor-thinning-variants-failed-their-local-accuracy/README.md) | 33 | informational |
| 2026-09-05 | [Compact-neuron path: measured useful cost lever](compact-neuron-path-measured-useful-cost-lever/README.md) | 77 | informational |
| 2026-09-05 | [Compact closure cost tests: factor thinning and contraction scheduling](compact-closure-cost-tests-factor-thinning-contraction-scheduling/README.md) | 37 | informational |
| 2026-09-05 | [Exact square-multiplication schedule: cost savings measured; runtime margin remained unresolved](exact-square-multiplication-schedule-cost-savings-measured-runtime-margin/README.md) | 43 | informational |
| 2026-09-05 | [Shared difference-leg compression: cost screen closed](shared-difference-leg-compression-cost-screen-closed/README.md) | 1 | rejected |
| 2026-09-06 | [Mixed rectangular multiplication: optimistic envelope, not a submission](mixed-rectangular-multiplication-optimistic-envelope-not-submission/README.md) | 6 | informational |
| 2026-09-06 | [Exact circuit simplification: insufficient cost gain](exact-circuit-simplification-insufficient-cost-gain/README.md) | 8 | rejected |
| 2026-09-06 | [Six-program rectangular catalog round: cost gate failed](six-program-rectangular-catalog-round-cost-gate-failed/README.md) | 11 | rejected |
| 2026-09-06 | [Inherited third-cumulant gain: branch diagnosis preserved the failed local qualification](inherited-third-cumulant-gain-branch-diagnosis-preserved-failed-local/README.md) | 20 | informational |
| 2026-09-06 | [Expanded cyclic catalog: arithmetic gate still fails](expanded-cyclic-catalog-arithmetic-gate-still-fails/README.md) | 9 | rejected |
| 2026-09-06 | [Tractable full-prefix query schedule: cost screen closed](tractable-full-prefix-query-schedule-cost-screen-closed/README.md) | 8 | rejected |
| 2026-09-06 | [Symmetric-output triangle compiler: prerequisite not met](symmetric-output-triangle-compiler-prerequisite-not-met/README.md) | 1 | rejected |
| 2026-09-06 | [Fixed-count cohort retention: closed on the information prerequisite](fixed-count-cohort-retention-closed-information-prerequisite/README.md) | 15 | rejected |
| 2026-09-06 | [Trace-dependent mean/variance mixture: algebra and numerical stage](trace-dependent-mean-variance-mixture-algebra-numerical-stage/README.md) | 16 | informational |
| 2026-09-07 | [Float32 pair-polynomial component passes independent14: 4.83424098592912e-9 to 4.788089831892398e-9](float32-pair-polynomial-component-passes-independent14-4-83424098592912e-9/README.md) | 27 | adopted |
| 2026-09-07 | [Float32 pair-polynomial evaluation improves two-network adjusted score by 0.9786264148%](float32-pair-polynomial-evaluation-improves-two-network-adjusted-score/README.md) | 22 | adopted |
| 2026-09-08 | [Active congruence cuts 239634432 FLOPs but fails its synthetic timing prerequisite](active-congruence-cuts-239634432-flops-but-fails-synthetic-timing/README.md) | 4 | rejected |
| 2026-09-08 | [Cached rank512 Gram component improves independent14 adjusted score by .1195130%](cached-rank512-gram-component-improves-independent14-adjusted-score-1195130/README.md) | 41 | adopted |
| 2026-09-08 | [Restricted four-level Winograd views fail both-shape timing gate despite numerical consistency](restricted-four-level-winograd-views-fail-both-shape-timing/README.md) | 3 | rejected |
| 2026-09-09 | [Serial six-level Winograd cuts arithmetic but takes 3.1969129249046593x the four-level wall](serial-six-level-winograd-cuts-arithmetic-but-takes-3/README.md) | 5 | rejected |
| 2026-09-09 | [Six-level factor-propagation probe bills 7062260240 FLOPs versus four-level 8417538304](six-level-factor-propagation-probe-bills-7062260240-flops-versus/README.md) | 37 | inconclusive |
| 2026-09-10 | [Broader aligned six-level factor propagation improves native40 adjusted error: ratio .9751291882616261, 40/40 wins](broader-aligned-six-level-factor-propagation-improves-native40-adjusted/README.md) | 226 | adopted |
| 2026-09-10 | [Partial six-level products improve native40 adjusted score by 1.268286%, with 38/40 wins](partial-six-level-products-improve-native40-adjusted-score-1/README.md) | 206 | adopted |
| 2026-09-10 | [Partial six-level products on the refitted mean head improve adjusted score: ratio 0.9873303202063866](partial-six-level-products-refitted-mean-head-improve-adjusted/README.md) | 88 | adopted |
| 2026-09-12 | [Padded six-level factor propagation cuts cold cost to ratio 0.954146799147358 but takes 54.854506958s wall](padded-six-level-factor-propagation-cuts-cold-cost-ratio/README.md) | 154 | inconclusive |
| 2026-09-13 | [Block-major Winograd layout would save about 2.5k requests but add about 0.2% bill; rewrite rejected](../../records/research/2026-09-13-block-major-winograd-layout-would-save-about-2-5k/README.md) | result only | rejected |
| 2026-09-13 | [Compact scratch preserves product bytes and FLOPs while reducing warm component wall to ratio 0.6280433518051552](compact-scratch-preserves-product-bytes-flops-while-reducing-warm/README.md) | 124 | informational |
| 2026-09-13 | [Corrected scalar gain slicing restores native/client parity and independent14 adjusted score 3.805066974869664e-9](corrected-scalar-gain-slicing-restores-native-client-parity-independent14/README.md) | 302 | adopted |
| 2026-09-13 | [Exact-shape frames with restored six-level pushes preserve frozen output bytes at 23.4s native wall](exact-shape-frames-restored-six-level-pushes-preserve-frozen/README.md) | 5 | rejected |
| 2026-09-13 | [Five-level left-weight transform reuse saves 1133294176 FLOPs per full call but still exceeds 45-second wall](five-level-left-weight-transform-reuse-saves-1133294176-flops/README.md) | 141 | informational |
| 2026-09-13 | [Four-level factor products fail 45-second wall and 1e-5 fidelity gates; saved adjusted ratio 1.0179899302661999](four-level-factor-products-fail-45-second-wall-1e/README.md) | 88 | rejected |
| 2026-09-13 | [Four-level packed factor propagation reduces native wall from 38.4s to 28.2s at +4.1% cost](four-level-packed-factor-propagation-reduces-native-wall-38/README.md) | 3 | rejected |
| 2026-09-13 | [Four-level scheduling frontier reduces six-level synthetic product wall from 0.557365042s to 0.373902958s](four-level-scheduling-frontier-reduces-six-level-synthetic-product/README.md) | 102 | inconclusive |
| 2026-09-13 | [Pair-polynomial fusion saves 114211840 FLOPs per pair map but fails its separate remote timing gate](pair-polynomial-fusion-saves-114211840-flops-per-pair-map/README.md) | 189 | rejected |
| 2026-09-13 | [Size-based depth schedules trade local wall against bill while retaining final MSE near 1.84e-8](size-based-depth-schedules-trade-local-wall-against-bill/README.md) | 2 | rejected |
| 2026-09-13 | [Small Winograd products cost 10-20x more wall time than plain products for a 25-35% bill saving](small-winograd-products-cost-10-20x-more-wall-time/README.md) | 1 | rejected |
| 2026-09-13 | [Square-product depth cap 3 lowers native wall to 18.2s at +9.5% bill](square-product-depth-cap-3-lowers-native-wall-18/README.md) | 3 | rejected |
| 2026-09-13 | [Streaming scratch preserves output bytes but increases warm product wall to ratio 1.4943203934180924](streaming-scratch-preserves-output-bytes-but-increases-warm-product/README.md) | 129 | rejected |
| 2026-09-13 | [Three-root Gram batching preserves bytes and FLOPs but gains only 2.884806% warm wall against a 10% gate](three-root-gram-batching-preserves-bytes-flops-but-gains/README.md) | 120 | rejected |
| 2026-09-13 | [Three-way microbatching is 5.5056% faster but misses the 10% synthetic kernel timing gate](three-way-microbatching-5-5056-faster-but-misses-10/README.md) | 101 | rejected |
| 2026-09-19 | [Caching four five-level left-weight transforms saves 1077536896 FLOPs with identical 14-network outputs](caching-four-five-level-left-weight-transforms-saves-1077536896/README.md) | 15 | adopted |
| 2026-09-19 | [Exact-cost Winograd depth selection plus 1.5 MF per request improves adjusted error: ratio .957179078](exact-cost-winograd-depth-selection-plus-1-5-mf/README.md) | 5 | adopted |
| 2026-09-19 | [Exact-cost depth selector with a 750k instead of 1.5M FLOP request penalty: 14-net adjusted ratio .9906359025, held](exact-cost-depth-selector-750k-instead-1-5m-flop/README.md) | 7 | inconclusive |
| 2026-09-19 | [Five-level three-packet read extension passes 14-network panel at adjusted ratio 0.9873624985](five-level-three-packet-read-extension-passes-14-network/README.md) | 14 | adopted |
| 2026-09-19 | [Six-level four-packet read misses development gate: pooled adjusted ratio 0.9950934019424302](six-level-four-packet-read-misses-development-gate-pooled/README.md) | 6 | rejected |
| 2026-09-19 | [Vectorized pair-polynomial fusion saves 1819768776 FLOPs but misses 12/14 panel-wins gate](vectorized-pair-polynomial-fusion-saves-1819768776-flops-but-misses/README.md) | 14 | rejected |
| 2026-09-20 | [Finite multiplication catalog on the delayed-rotation estimator saves at most 15.19%/16.11% versus 21.39% needed](finite-multiplication-catalog-delayed-rotation-estimator-saves-most-15/README.md) | 2 | rejected |
| 2026-09-21 | [Midpoint/difference pair read fails wide numerical fixture: relative error 0.0035854277148522043](midpoint-difference-pair-read-fails-wide-numerical-fixture-relative/README.md) | 18 | rejected |
| 2026-09-22 | [Structural-zero fast multiplication adds no useful saving beyond the existing seven-program set](structural-zero-fast-multiplication-adds-no-useful-saving-beyond/README.md) | 4 | rejected |
| 2026-09-23 | [Grouped Winograd requests with copied operands: fewer requests but no metered saving](grouped-winograd-requests-copied-operands-fewer-requests-but-no/README.md) | 6 | rejected |
| 2026-09-23 | [Estimator-body request reductions: bit-exact predictions and 0.58% less metered work](estimator-body-request-reductions-bit-exact-predictions-0-58/README.md) | 48 | adopted |
| 2026-09-23 | [Selection rounded to 64 or 128 units: extra selected units outweigh deeper multiplication](selection-rounded-64-or-128-units-extra-selected-units/README.md) | 33 | rejected |
| 2026-09-23 | [Maximum-shape grouped kernels cut work 2.30-4.43% but raise server compute 27-73%](maximum-shape-grouped-kernels-cut-work-2-30-4/README.md) | 38 | rejected |
| 2026-09-23 | [Held aliases and last-level copy elimination preserve predictions and reduce work 1.18-1.23%](held-aliases-last-level-copy-elimination-preserve-predictions-reduce/README.md) | 37 | adopted |
| 2026-09-23 | [Depth-six squares and padded pushes save work but exceed the projected wall budget](depth-six-squares-padded-pushes-save-work-but-exceed/README.md) | 25 | rejected |
| 2026-09-23 | [Five-level layouts for previously shallower pushes reduce warm work about 1.9%](five-level-layouts-previously-shallower-pushes-reduce-warm-work/README.md) | 9 | adopted |
| 2026-09-23 | [Third-order plane sums converted to float32: 0.52% less work with rounding-level error](third-order-plane-sums-converted-float32-0-52-less/README.md) | 35 | adopted |
| 2026-09-23 | [Padded depth-six pushes save 0.426% work for 4.40% more server CPU](padded-depth-six-pushes-save-0-426-work-4/README.md) | 10 | inconclusive |
| 2026-09-23 | [Pair-map temporaries reuse the multiplication arena: bit-exact and about 1.2-1.6% less server CPU](pair-map-temporaries-reuse-multiplication-arena-bit-exact-about/README.md) | 8 | adopted |
| 2026-09-23 | [Batching covariance and residual-plane products cuts server CPU 2.12% and residual time 12.6 ms](batching-covariance-residual-plane-products-cuts-server-cpu-2/README.md) | 6 | adopted |
| 2026-09-23 | [Retained frame-binding and body aliases reduce hidden deallocation trips without changing work](retained-frame-binding-body-aliases-reduce-hidden-deallocation-trips/README.md) | 7 | adopted |
| 2026-09-23 | [Eliminating unused frame slots and unwind scratch cuts cold work 1.836% with identical predictions](eliminating-unused-frame-slots-unwind-scratch-cuts-cold-work/README.md) | 14 | adopted |
| 2026-09-23 | [Upper-block covariance multiplication: adjusted error improves 1.003% but adds requests and residual](upper-block-covariance-multiplication-adjusted-error-improves-1-003/README.md) | 3 | inconclusive |
| 2026-09-23 | [Layer-wide temporary retention fails: live-array peak increases 1,469 MB](layer-wide-temporary-retention-fails-live-array-peak-increases/README.md) | 3 | rejected |
| 2026-09-23 | [Smaller request and alias reductions preserve memory while removing about 938 warm round trips](smaller-request-alias-reductions-preserve-memory-while-removing-about/README.md) | 2 | adopted |
| 2026-09-23 | [Upper-block covariance and reduced requests improve adjusted error 1.094% on 40 networks](upper-block-covariance-reduced-requests-improve-adjusted-error-1/README.md) | 4 | adopted |
| 2026-09-24 | [Six-level read estimates omitted 64-padding: all-class work saving becomes a 0.39% increase](six-level-read-estimates-omitted-64-padding-all-class/README.md) | 6 | rejected |
| 2026-09-24 | [Shared left transforms and repeated-square reuse save 0.34% work with identical predictions](shared-left-transforms-repeated-square-reuse-save-0-34/README.md) | 6 | adopted |
| 2026-09-24 | [Row-capacity frame keying has negligible row-slice CPU cost and promises 0.66% less warm work](row-capacity-frame-keying-has-negligible-row-slice-cpu/README.md) | 3 | informational |
| 2026-09-24 | [Packetized six-level reads on eligible shapes save 0.363% work with rounding-level error](packetized-six-level-reads-eligible-shapes-save-0-363/README.md) | 6 | adopted |
| 2026-09-24 | [Extending padded depth-six pushes to 64-multiple row counts costs 49-93% more user CPU](extending-padded-depth-six-pushes-64-multiple-row-counts/README.md) | 1 | rejected |
| 2026-09-24 | [In-place tree Winograd removes level copies: 35.24% less user CPU and 1.750% less work](place-tree-winograd-removes-level-copies-35-24-less/README.md) | 8 | adopted |
| 2026-09-24 | [Alternative-basis Winograd reduces metered work 1.22% and user CPU 3.29%](alternative-basis-winograd-reduces-metered-work-1-22-user/README.md) | 12 | adopted |
| 2026-09-24 | [Tree six-level reads and 64-row padding: adjusted error improves 0.948% on 40 networks](tree-six-level-reads-64-row-padding-adjusted-error/README.md) | 2 | adopted |
| 2026-09-24 | [Independent alternative-basis kernel saves slightly more arithmetic but adds 531 warm operation calls](independent-alternative-basis-kernel-saves-slightly-more-arithmetic-but/README.md) | 4 | rejected |
| 2026-09-24 | [Six-level batched covariance and residual-plane products improve four-network adjusted error 1.557%](six-level-batched-covariance-residual-plane-products-improve-four/README.md) | 6 | adopted |
| 2026-09-25 | [Natural-order tree views remove 1,455-1,565 calls with identical outputs, work and peak live memory](natural-order-tree-views-remove-1-455-1-565/README.md) | 1 | adopted |
| 2026-09-25 | [Additional eligible depth-six products save 0.54% work for 3.95% more user CPU](additional-eligible-depth-six-products-save-0-54-work/README.md) | 2 | adopted |
| 2026-09-25 | [Float32 training needs float64 mean-path and cube-sum islands to meet the gradient gate](float32-training-needs-float64-mean-path-cube-sum-islands/README.md) | 37 | informational |
| 2026-09-25 | [Tree routing of thin residual-plane products saves only 0.25% within a 2,000-request allowance](tree-routing-thin-residual-plane-products-saves-only-0/README.md) | 5 | rejected |
| 2026-09-25 | [Strided level-operation fusion cuts requests 9.5% but increases user CPU 7.9%](strided-level-operation-fusion-cuts-requests-9-5-but/README.md) | 1 | adopted |
| 2026-09-25 | [Depth seven on fitting product classes saves only 0.55% work for 22% more local backend time](depth-seven-fitting-product-classes-saves-only-0-55/README.md) | 1 | rejected |
| 2026-09-25 | [Estimator-body value reuse and safe einsums price about 974 fewer requests per network](estimator-body-value-reuse-safe-einsums-price-about-974/README.md) | 3 | informational |
| 2026-09-25 | [Conservative forward-transform reuse saves 0.09% work and 376-483 requests with identical predictions](conservative-forward-transform-reuse-saves-0-09-work-376/README.md) | 2 | adopted |
| 2026-09-25 | [Depth-six contraction splitting saves 0.96% work; backend cost depends strongly on BLAS](depth-six-contraction-splitting-saves-0-96-work-backend/README.md) | 2 | informational |
| 2026-09-25 | [Summing split contractions in leaf space removes repeated unwind work](summing-split-contractions-leaf-space-removes-repeated-unwind-work/README.md) | 6 | adopted |
| 2026-09-25 | [Restricting request reductions to at most three einsum operands restores exactness](restricting-request-reductions-most-three-einsum-operands-restores-exactness/README.md) | 3 | adopted |
| 2026-09-25 | [Leaf-space contraction splitting improves adjusted error 0.807% on all 40 validation networks](leaf-space-contraction-splitting-improves-adjusted-error-0-807/README.md) | 7 | adopted |
| 2026-09-25 | [Setup-bound super-views fail: strided tree operations add 8-59% CPU for little binding saving](../../records/research/2026-09-25-setup-bound-super-views-fail-strided-tree-operations-add/README.md) | result only | rejected |
| 2026-09-25 | [Expanded contraction-split shape rule saves 1.147% work with rounding-level prediction changes](expanded-contraction-split-shape-rule-saves-1-147-work/README.md) | 1 | adopted |
| 2026-09-25 | [Corrected body request package passes exactness and client smoke on the final composition](corrected-body-request-package-passes-exactness-client-smoke-final/README.md) | 1 | adopted |
| 2026-09-25 | [Selective depth frontier identifies early shared-operand depth-seven products as a viable wall tradeoff](selective-depth-frontier-identifies-early-shared-operand-depth-seven/README.md) | 3 | informational |
| 2026-09-25 | [Depth-seven column packing needs true product dimensions to price padding](depth-seven-column-packing-needs-true-product-dimensions-price/README.md) | 4 | adopted |
| 2026-09-25 | [Depth-seven packed and square products combine to reduce arithmetic](depth-seven-packed-square-products-combine-reduce-arithmetic/README.md) | 2 | adopted |
| 2026-09-25 | [Inner-dimension split revision passes the paired server-CPU gate](inner-dimension-split-revision-passes-paired-server-cpu-gate/README.md) | 4 | adopted |
| 2026-09-26 | [Batched leaf multiplication: ordinary matrix multiplication is faster than tensor contraction or outer accumulation](batched-leaf-multiplication-ordinary-matrix-multiplication-faster-than-tensor/README.md) | 1 | rejected |
| 2026-09-26 | [Covariance output tiles narrowed to 256 or 128 rows: metered cost rises](covariance-output-tiles-narrowed-256-or-128-rows-metered/README.md) | 3 | rejected |
| 2026-09-26 | [Rank-15 rectangular multiplication: saving orientations cost 1.92–3.25 times CPU](rank-15-rectangular-multiplication-saving-orientations-cost-1-923/README.md) | 6 | rejected |
| 2026-09-26 | [Rank-23 three-by-three multiplication loses 4.66–19.90% compute and costs 2.5–3.7 times CPU](rank-23-three-three-multiplication-loses-4-6619-90/README.md) | 4 | rejected |
| 2026-09-26 | [Star diagonal reuses an existing transport: native bill falls another 0.3372% and 0.3393%](star-diagonal-reuses-existing-transport-native-bill-falls-another/README.md) | 1 | adopted |
| 2026-09-26 | [Thin star products routed to depth three: eight training scores improve by a pooled 0.573464%](thin-star-products-routed-depth-three-eight-training-scores/README.md) | 1 | adopted |
| 2026-09-26 | [Triangular covariance congruence: primitive FLOPs rise 12.4783%](triangular-covariance-congruence-primitive-flops-rise-12-4783/README.md) | 2 | rejected |
| 2026-09-26 | [Batched rectangular multiplication children: arithmetic saving fails CPU admission](batched-rectangular-multiplication-children-arithmetic-saving-fails-cpu/README.md) | 8 | rejected |
| 2026-09-27 | [Covariance upper strips of 128 rows: primitive bill rises 20.6134%](covariance-upper-strips-128-rows-primitive-bill-rises-20/README.md) | 4 | rejected |
| 2026-09-27 | [Covariance-only classical products stabilize layer 9 but final response still fails](covariance-only-classical-products-stabilize-layer-9-but-final/README.md) | 8 | rejected |
| 2026-09-27 | [Depth-seven long-inner reads: full-estimator adjusted gain is only 0.288486%](depth-seven-long-inner-reads-full-estimator-adjusted-gain/README.md) | 9 | rejected |
| 2026-09-27 | [Float64 covariance diagonal reductions: native adjusted gain is 0.251635%, below expansion gate](float64-covariance-diagonal-reductions-native-adjusted-gain-0-251635/README.md) | 9 | inconclusive |
| 2026-09-27 | [Mixed-radix multiplication pilot: cold runtime is 2.444 times the reference](mixed-radix-multiplication-pilot-cold-runtime-2-444-times/README.md) | 9 | rejected |
| 2026-09-28 | [Float64 covariance diagonal extension: 16-network adjusted error rises 0.117174%](float64-covariance-diagonal-extension-16-network-adjusted-error-rises/README.md) | 7 | rejected |
| 2026-09-28 | [Four-term Gaussian evaluation alone: incremental gain has confidence crossing zero](four-term-gaussian-evaluation-alone-incremental-gain-has-confidence/README.md) | 17 | rejected |
| 2026-09-28 | [Memory-blocked rank-15 multiplication: initial gain fails eight-network expansion](memory-blocked-rank-15-multiplication-initial-gain-fails-eight/README.md) | 12 | rejected |
| 2026-09-28 | [Precision edits with 256-row covariance symmetry: validation gain 0.7988805% stays below 1%](precision-edits-256-row-covariance-symmetry-validation-gain-0/README.md) | 16 | inconclusive |
| 2026-09-28 | [Rank-15 rectangular primitive: 4.267895711% fewer FLOPs but 3.169052 times CPU](rank-15-rectangular-primitive-4-267895711-fewer-flops-but/README.md) | 9 | rejected |
| 2026-09-28 | [Thin star depth four: primitive bill saving is only 0.0261%](thin-star-depth-four-primitive-bill-saving-only-0/README.md) | 2 | rejected |
| 2026-09-28 | [Three-term Gaussian evaluation and 128-row diagonal blocks: all 40 validation scores improve](three-term-gaussian-evaluation-128-row-diagonal-blocks-all/README.md) | 37 | adopted |
| 2026-09-29 | [Depth-three covariance diagonal products: adjusted gain 0.355945% misses the fixed 0.5% gate](depth-three-covariance-diagonal-products-adjusted-gain-0-355945/README.md) | 37 | rejected |
| 2026-09-30 | [Birth-direction linear projection and delta factorization: total adjusted gain reaches 0.7325507236314643%](birth-direction-linear-projection-delta-factorization-total-adjusted-gain/README.md) | 113 | adopted |
| 2026-09-30 | [Depth-seven read with inner padding: eight-network increment is only 0.0307362355%](depth-seven-read-inner-padding-eight-network-increment-only/README.md) | 104 | rejected |
| 2026-09-30 | [Dyadic balancing inside covariance-weight multiplication increases adjusted error 0.2767676444%](dyadic-balancing-inside-covariance-weight-multiplication-increases-adjusted/README.md) | 32 | rejected |
| 2026-09-30 | [Dyadic inner balancing of inherited young factors: adjusted error rises 0.4421872500%](dyadic-inner-balancing-inherited-young-factors-adjusted-error-rises/README.md) | 101 | rejected |
| 2026-09-30 | [Packed pair-block symmetry: exact outputs on all eight networks with 0.2085143273% adjusted gain](packed-pair-block-symmetry-exact-outputs-all-eight-networks/README.md) | 86 | adopted |
| 2026-09-30 | [Three-shape depth-seven read with eager bindings: eight-network adjusted gain is 0.3015740835%](three-shape-depth-seven-read-eager-bindings-eight-network/README.md) | 97 | adopted |
| 2026-09-30 | [Young negative trace and depth-five covariance products: validation improves adjusted score 1.0748874384119644%](young-negative-trace-depth-five-covariance-products-validation-improves/README.md) | 67 | adopted |
| 2026-10-01 | [Half-zero strip kernel for odd-32 dimensions lowers charged work by 1.30%](half-zero-strip-kernel-odd-32-dimensions-lowers-charged/README.md) | 4 | adopted |
| 2026-10-01 | [Response floors, larger selection counts and plain products fail to improve cost and error together](response-floors-larger-selection-counts-plain-products-fail-improve/README.md) | 8 | rejected |
| 2026-10-02 | [Seven-level odd-64 strips trade 0.70% lower charged work for approximately 12.8 seconds of wall time](seven-level-odd-64-strips-trade-0-70-lower/README.md) | 1 | rejected |
| 2026-10-02 | [Sharing strip-kernel leaf views per tree saves 146 / 122 index requests with identical predictions](sharing-strip-kernel-leaf-views-per-tree-saves-146/README.md) | 3 | adopted |
| 2026-10-10 | [Large-tree unfusing and eager bind-time fusion rejected after request and FLOP measurements](large-tree-unfusing-eager-bind-time-fusion-rejected-after/README.md) | 13 | rejected |
| 2026-10-11 | [Zero strip-push gain floor rejected: -0.051% FLOPs costs +7.5%/+2.9% kernel CPU](zero-strip-push-gain-floor-rejected-0-051-flops/README.md) | 7 | rejected |
| 2026-10-11 | [Seven-level strip-push variants rejected: gains -0.034% and -0.010% are too small](seven-level-strip-push-variants-rejected-gains-0-034/README.md) | 22 | rejected |
| 2026-10-12 | [Dead-cut tiles rejected on the current implementation: -0.0088% FLOPs costs 5,869 charged operations](dead-cut-tiles-rejected-current-implementation-0-0088-flops/README.md) | 4 | rejected |
| 2026-10-12 | [Merged newborn-common product parked: warm FLOPs -0.148% on one network and +0.083% on another](merged-newborn-common-product-parked-warm-flops-0-148/README.md) | 5 | rejected |
| 2026-10-12 | [Winograd head-message products parked: -0.10% FLOPs costs 420 operations at rounding-level error](winograd-head-message-products-parked-0-10-flops-costs/README.md) | 3 | rejected |
| 2026-10-13 | [Remainder-32 strided-read variant is byte-identical but does no work on the measured network](remainder-32-strided-read-variant-byte-identical-but-does/README.md) | 12 | inconclusive |
| 2026-10-14 | [Pair-local fusion: displacement RMS 0.0348 exceeds the 0.001 exactness cap](pair-local-fusion-displacement-rms-0-0348-exceeds-0/README.md) | 12 | rejected |
| 2026-10-14 | [Four learned projections via Winograd kernels: 405,837,584 FLOPs saved, displacement RMS 0.0326 exceeds the 0.001 cap](four-learned-projections-via-winograd-kernels-405-837-584/README.md) | 10 | rejected |
| 2026-10-14 | [Layer-7 plane rank 321 (one added signed tensor-trace direction): adjusted +6.18%, the odd rank disables aligned kernels](layer-7-plane-rank-321-one-added-signed-tensor/README.md) | 1 | rejected |
| 2026-10-14 | [Young plane read at contraction depth 7 with exact setup prebinding: two-network -0.48%, science-24 -0.098%, adopted](young-plane-read-contraction-depth-7-exact-setup-prebinding/README.md) | 6 | adopted |
| 2026-10-15 | [Rectangular rank-26 matrix product: representative current kernels cost +0.79% and +23.79%](../../records/research/2026-10-15-rectangular-rank-26-matrix-product-representative-current-kernels-cost/README.md) | result only | rejected |
| 2026-10-15 | [Grouped rank-40 matrix kernels: corrected all-24 adjusted -0.706639% versus the row-energy parent](grouped-rank-40-matrix-kernels-corrected-all-24-adjusted/README.md) | 58 | adopted |
| 2026-10-16 | [Kernel dispatch expanded to 167 shapes with layer-one read reallocation: all-24 adjusted -0.2096793%](kernel-dispatch-expanded-167-shapes-layer-one-read-reallocation/README.md) | 73 | adopted |
| 2026-10-16 | [Tree loads replaced by two unstacks and a cached half view: 465/506 fewer requests at identical FLOPs](tree-loads-replaced-two-unstacks-cached-half-view-465/README.md) | 8 | adopted |
| 2026-10-16 | [Seven-level kernel gain floor reduction: at most 0.095% warm bill saving with greater request risk](seven-level-kernel-gain-floor-reduction-most-0-095/README.md) | 16 | rejected |
| 2026-10-16 | [Lazy fusion of tree arithmetic: identical FLOPs but 908 more first and 96 more warm requests](lazy-fusion-tree-arithmetic-identical-flops-but-908-more/README.md) | 8 | rejected |
| 2026-10-16 | [K-split key refit and FIFO-to-LRU strip caches: identical predictions, 0.03% fewer FLOPs](k-split-key-refit-fifo-lru-strip-caches-identical/README.md) | 3 | adopted |
| 2026-10-16 | [Restoring dropped grouped-read keys saves only 0.004% bill for 217 additional operations](restoring-dropped-grouped-read-keys-saves-only-0-004/README.md) | 2 | rejected |
| 2026-10-16 | [Learned-operator products through tree kernels: 0.082% saving canceled by extra runtime-relief triggers](learned-operator-products-through-tree-kernels-0-082-saving/README.md) | 7 | rejected |
| 2026-10-16 | [Expanded read-kernel grid: warm bill -0.1666% but projected worst wall 116.7 seconds](expanded-read-kernel-grid-warm-bill-0-1666-but/README.md) | 9 | inconclusive |
| 2026-10-16 | [Unbound depth-six tree views served from maximal boxes: exact warm FLOPs -0.0601%](unbound-depth-six-tree-views-served-maximal-boxes-exact/README.md) | 6 | inconclusive |
| 2026-10-16 | [Boxing every depth-six tree key saves slightly more FLOPs but raises server instructions 13-15%](boxing-every-depth-six-tree-key-saves-slightly-more/README.md) | 6 | rejected |
| 2026-10-16 | [Direct kernel-slot writes plus sliced pair factors: exact warm FLOPs -0.0971% with lazy tree views](direct-kernel-slot-writes-plus-sliced-pair-factors-exact/README.md) | 16 | inconclusive |
| 2026-10-16 | [Wall-capped extra read plans: only 37 plans satisfy 0.1% bill per added server-second](wall-capped-extra-read-plans-only-37-plans-satisfy/README.md) | 9 | rejected |
| 2026-10-16 | [Grouped join and old-bank kernels: warm bill -0.347% but worst wall 121.23 seconds and risk 57.09%](grouped-join-old-bank-kernels-warm-bill-0-347/README.md) | 12 | rejected |
| 2026-10-16 | [Grouped young-bank push kernels: about 0.12% platform-mix bill saving for 1.9 seconds added mean wall](grouped-young-bank-push-kernels-about-0-12-platform/README.md) | 13 | inconclusive |
| 2026-10-16 | [Classical float32 and float64-accumulated matrix products do not improve final accuracy](classical-float32-float64-accumulated-matrix-products-do-not-improve/README.md) | 2 | rejected |
| 2026-10-16 | [Wall-capped read-grid swap with 12 zero-saving plans removed: fresh-pair warm bill -0.100%](wall-capped-read-grid-swap-12-zero-saving-plans/README.md) | 19 | inconclusive |
| 2026-10-16 | [Grouped strided push and small-read families add 490–4,050 cold requests and fail the static runtime limits](grouped-strided-push-small-read-families-add-4904-050/README.md) | 1 | rejected |
| 2026-10-16 | [Exact select rosters with dense remainders: bill -0.47% warm but MSE +2.39%; the padding units carry 2.3% of accuracy](exact-select-rosters-dense-remainders-bill-0-47-warm/README.md) | 7 | rejected |
| 2026-10-17 | [Approximate multiplication with signed-pair and common/difference layouts: native error rises by about 4.3%](approximate-multiplication-signed-pair-common-difference-layouts-native-error/README.md) | 5 | rejected |
| 2026-10-17 | [Rank-four exact cubic-message kernel saves 355,273,664 FLOPs per network; eight-network adjusted -0.204%](rank-four-exact-cubic-message-kernel-saves-355-273/README.md) | 7 | adopted |
| 2026-10-17 | [Hardware leaf einsum for the tree kernels: +8.8-35.4% kernel time and +2.4-11.1% leaf bill against matmul, closed](hardware-leaf-einsum-tree-kernels-8-8-35-4/README.md) | 9 | rejected |
| 2026-10-17 | [Twelve small strided-read shapes on shared 40-way batches: eight-network raw -0.199%; expansions lose](twelve-small-strided-read-shapes-shared-40-way-batches/README.md) | 17 | adopted |
| 2026-10-17 | [Exact depth peeling of the layer-6 read body: 0.73% bill uncapped, 0.52% under a 750-call budget, 0.42% as routed](exact-depth-peeling-layer-6-read-body-0-73/README.md) | 34 | adopted |
| 2026-10-17 | [All push-peeling routes and three reads reverted to parent executors: cold failures removed](all-push-peeling-routes-three-reads-reverted-parent-executors/README.md) | 10 | adopted |
| 2026-10-17 | [Read-peel removal and fusion prefix 192: small exact call reduction, no 1,500-call lever](read-peel-removal-fusion-prefix-192-small-exact-call/README.md) | 13 | rejected |
| 2026-10-17 | [Eager fused-operation tuples on the repair lineage increased cold requests and backend time](eager-fused-operation-tuples-repair-lineage-increased-cold-requests/README.md) | 1 | rejected |
| 2026-10-17 | [Stacking elementwise operations in the moment core saved no requests after stack and unstack costs](stacking-elementwise-operations-moment-core-saved-no-requests-after/README.md) | 2 | rejected |
| 2026-10-17 | [Pair-map and readout-feature vector batching removed 97–98 requests with bit-identical predictions](pair-map-readout-feature-vector-batching-removed-9798-requests/README.md) | 1 | informational |
| 2026-10-17 | [Tree aliases with union push products saved 0.333% bill at flat isolated backend and residual time](tree-aliases-union-push-products-saved-0-333-bill/README.md) | 1 | adopted |
| 2026-10-17 | [Removing the union-product shape guard reduced requests; the extra rank-48 kernel had no covered shapes](removing-union-product-shape-guard-reduced-requests-extra-rank/README.md) | 1 | adopted |
| 2026-10-17 | [Disabling grouped strided-read plans added 1.01% bill while reducing backend time by 11.9%](disabling-grouped-strided-read-plans-added-1-01-bill/README.md) | 18 | rejected |
| 2026-10-17 | [Six final read-cancellation operations in float64: small unconfirmed gain and an unsupported-shape exception](six-final-read-cancellation-operations-float64-small-unconfirmed-gain/README.md) | 1 | rejected |
| 2026-10-18 | [Exact signed-pair read rotation: improved a dormant microkernel but lost 0.854695% adjusted natively](exact-signed-pair-read-rotation-improved-dormant-microkernel-but/README.md) | 5 | rejected |
| 2026-10-18 | [Exact peeling with a 600-request cap did not transfer historical coverage to the current union products](exact-peeling-600-request-cap-did-not-transfer-historical/README.md) | 3 | rejected |
| 2026-10-18 | [Physics bookkeeping removed 338 requests but changed rounding and lost 0.050% adjusted](physics-bookkeeping-removed-338-requests-but-changed-rounding-lost/README.md) | 8 | rejected |
| 2026-10-18 | [Direct union products: -0.225537% adjusted on three networks, below the score target](direct-union-products-0-225537-adjusted-three-networks-below/README.md) | 10 | rejected |
| 2026-10-18 | [Covariance-update broadcast-view removal saved 133–134 requests for a 0.04588% bill cost](covariance-update-broadcast-view-removal-saved-133134-requests-0/README.md) | 4 | informational |
| 2026-10-18 | [Compact union-product portfolio: -0.392786% adjusted on seven current networks](compact-union-product-portfolio-0-392786-adjusted-seven-current/README.md) | 17 | informational |
| 2026-10-18 | [Five shared-product routes added to compact union products: -0.427150% adjusted on seven](five-shared-product-routes-added-compact-union-products-0/README.md) | 20 | informational |
| 2026-10-18 | [Expanded union and layer-5 products: 32-network adjusted -0.674299339%, pending runtime repair](expanded-union-layer-5-products-32-network-adjusted-0/README.md) | 24 | adopted |
| 2026-10-18 | [Product request cap reduced from 500 to 300: -0.4553932186% adjusted on 32 networks](product-request-cap-reduced-500-300-0-4553932186-adjusted/README.md) | 1 | adopted |
| 2026-10-18 | [Exact young-product fusion removed 270–334 requests but gave no confirmed score improvement](exact-young-product-fusion-removed-270334-requests-but-gave/README.md) | 21 | informational |
| 2026-10-18 | [Centered fourth-moment Hermite channel saved 0.0395428% bill in synthetic checks](centered-fourth-moment-hermite-channel-saved-0-0395428-bill/README.md) | 6 | informational |
| 2026-10-18 | [Young fusion, centered channel and core reuse stack: +0.004909% adjusted after 19 networks](young-fusion-centered-channel-core-reuse-stack-0-004909/README.md) | 1 | rejected |
| 2026-10-18 | [Retained-cumulant gate with centered channel, core reuse and padding: -0.183035% adjusted on seven](retained-cumulant-gate-centered-channel-core-reuse-padding-0/README.md) | 31 | inconclusive |
| 2026-10-18 | [Young fusion with early temporary release reduced memory but worsened the held symmetric-join stack](young-fusion-early-temporary-release-reduced-memory-but-worsened/README.md) | 2 | rejected |
| 2026-10-19 | [Kernel-tree box views: one operand route corrupts predictions; the other two save -0.011% FLOPs](kernel-tree-box-views-one-operand-route-corrupts-predictions/README.md) | 3 | adopted |
| 2026-10-19 | [More depth-seven kernel routes add +0.028% FLOPs, 325 requests per network and +30% residual time](more-depth-seven-kernel-routes-add-0-028-flops/README.md) | 1 | rejected |
| 2026-10-20 | [Exact copy removal cuts warm FLOPs 0.3332024% with byte-identical predictions on network 35](exact-copy-removal-cuts-warm-flops-0-3332024-byte/README.md) | 1 | adopted |
| 2026-10-20 | [Gram arithmetic plus float32 matrix-vector changes: -0.206095% adjusted on 32 networks](gram-arithmetic-plus-float32-matrix-vector-changes-0-206095/README.md) | 1 | adopted |
| 2026-10-20 | [Gram, covariance coefficient and exact-copy stack: -0.3920% adjusted, only 0.055 points beyond submitted stack](gram-covariance-coefficient-exact-copy-stack-0-3920-adjusted/README.md) | 1 | rejected |
| 2026-10-20 | [Float32 and exact-copy stack without added covariance coefficient: -0.183168835% versus #337393](float32-exact-copy-stack-without-added-covariance-coefficient-0/README.md) | 2 | rejected |
| 2026-10-20 | [Float32 arithmetic, exact copies and radical edge: -0.276832224% adjusted versus #337393](float32-arithmetic-exact-copies-radical-edge-0-276832224-adjusted/README.md) | 8 | adopted |
| 2026-10-21 | [Fixed inner product fallback adds .1258107% FLOPs and leaves insufficient score margin](../../records/research/2026-10-21-fixed-inner-product-fallback-adds-1258107-flops-leaves-insufficient/README.md) | result only | rejected |
| 2026-10-21 | [Carrier age singleton and square-depth cap 5 are byte-identical no-ops in the deployed estimator](carrier-age-singleton-square-depth-cap-5-byte-identical/README.md) | 2 | rejected |
