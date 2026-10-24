# Order-32 high-order closure: literal run exceeded the wall limit; diagnostic also exceeded residual time

2026-08-24 · experiment · failed · measured

The frozen order-32 high-order closure with a rank-2048 fourth-cumulant carrier failed the official 120-second wall gate before the runner received a response. A longer nonqualifying diagnostic completed at 2,180,584,652,856 FLOPs and 7,174,275,072 bytes, but required 214.343 seconds wall and 1.045 seconds residual.

Recorded measurement:

    FB-EC4-32 plus CP4R2048 V1 is byte-frozen and independently source-audited. The official literal run failed only the 120-second wall gate: the worker made no response before its 125-second runner deadline, while sampled solution RSS peaked at 6.150GB. A non-qualifying 240-second diagnostic completed finite at 2,180,584,652,856 FLOPs and 7,174,275,072 bytes, but took 214.343 seconds and 1.045 seconds residual. Profiling localizes 202.361 of 209.191 backend seconds to 339 Q32 einsums whose default path repeatedly materializes a 792MiB temporary. Additive V2 is replacing only that execution plan with an algebra-equivalent low-temporary path and batched four-moment evaluation. Frozen V2 reduced Q32 calls from 339 to 87, then completed finite at 2,179,715,874,872 FLOPs, 35.217 seconds, and 6,435,307,520 bytes. It passed the FLOP, wall, and memory gates but failed only residual control time at 1.086 seconds versus 0.4. Frozen and independently audited V3 then vectorized the scalar Hermite/Q38 work and reduced the projected/observed operation count from 16,480 to 9,271. Its literal common-driver meter exactly matched 2,179,652,189,740 FLOPs (99.1191%), returned from the in-source finite guard in 41.811 seconds, and used 6,309,822,464 bytes sampled solution RSS. It still failed only residual control time: 0.794457 seconds versus 0.4. Additive V4 now fuses the remaining scalar/plane assembly without removing any tensor state, Q32 term, CP4 carrier, or analytic channel. One aggressive 4,511-call fp32 star-push fusion was rejected before metering after it destabilized a fixed depth-16 orthogonal recurrence that frozen V3 completed; V4 now retains only association-preserving reuse and independently stable fp64 pointwise fusions. The coherent immutable c9 baseline stayed bitwise equal to V3 on four fixed depth-16 fixtures and returned finite predictions at 2,187,178,578,919 FLOPs (99.4614%), 37.919 seconds, and 5,441,667,072 bytes sampled RSS, but it still failed the sole residual gate at 0.813841 seconds. Exact trace decomposition showed that the 0.603150-second FlopScope wrapper overhead is exempt from the residual cap, while cProfile localized estimator-side allocation/lifetime work to _predict (0.218 seconds self) and the six-group Q32 four-pair consumer (0.142 seconds self). The next additive experiment therefore targets Q32 temporary lifetime without changing any channel or contraction order. The first immutable all-17-channel lifetime candidate preserved every fixed-fixture output and cut residual by 0.081563 seconds, but remained a residual-only failure at 0.732278 seconds. Its exact bill was unchanged at 2,187,178,578,919 FLOPs, wall was 37.645 seconds, and sampled solution RSS was 5,854,887,936 bytes. Its follow-up profile left _predict self time essentially unchanged, so the remaining 0.332278-second gap is now routed to full-reset and layer-state lifetime rather than wrapper-call count. A dedicated non-overlap timing replay attributes 0.407225 seconds, or 62.2% of its instrumented residual, to the 14 full ReLU resets. The next immutable native-out candidate reused the four-pair result and reduction planes while retaining all 17 channels and the six frozen group additions. It cut residual again to 0.585189 seconds at 2,187,296,019,431 FLOPs (99.4667%), 37.669 seconds, and 5,900,582,912 bytes sampled solution RSS. The remaining 0.185189-second gap is routed to persistent rho-power and quadratic-product workspaces that already pass poisoned-reuse and all four depth-16 bitwise gates. The combined rho/product/weight arena candidate then reduced residual to 0.486610 seconds while remaining finite at 2,184,175,457,255 FLOPs (99.3248%), 35.128 seconds, and 6,224,248,832 bytes sampled solution RSS. Its remaining 0.086610-second miss is routed to exact native-out connected-plane/full-reset scratch plus the already bitwise-qualified sparse quadratic map, not to any science deletion. A first combined reset-arena/CP3 ping-pong draft was rejected before preregistration or metering: its CAP-strided final-K3 scratch changed the GEMV association at n=32/r=513 by 3.0517578125e-05. The additive repair uses flat product storage reshaped to an actually compact (n,r) view; it now passes the fixed depth-16 suite plus expanded rank/cap and poisoned-reuse gates, with a corrected projected bill of 2,179,742,954,469 FLOPs and 19,280,301,083 FLOPs headroom. Its exact immutable meter then PASSED: finite predictions, zero failures, the exact projected bill (99.1232%), 35.341 seconds wall, 4,618,567,680 bytes sampled solution RSS, and 0.392632 seconds residual. This is the first fully legal literal meter for the rich closure, but the 7.368ms residual margin is explicitly fragile; freeze/moment/package gates proceed while an exact-science allocation successor seeks a safer board margin. The first exact-weight moment replay then passed finiteness, resource, r1, and variance gates but HOLDs scientific promotion: layers 10-16 contain only sparse late-tail order-three/four Hankel violations (absolute minima about 1e-30 to 2e-28) at near-dead units. A scale-normalized, worst-unit diagnostic must now distinguish cancellation dust from material moment-law inconsistency; no scalar clipping or closure feedback is allowed. The outward-rounded V2 diagnostic has now resolved that ambiguity for the stored binary64 tuples. Every sign interval was evaluated with nextafter-up bounds; none of the late negatives was rounding-compatible. It certified 9 negative A and 13 negative B determinants across layers 10-16 (22 of 32,768 tested signs), all at near-dead units with absolute margins from about 1e-30 down to 4.25e-39. Layers 1-9 remained wholly positive. The diagnostic itself passed exactly at a 5,902,272-FLOP increment and total 2,179,748,856,741, but scientific promotion remains HOLD because the represented moment tuples are genuinely infeasible even though their affected mass is tiny. A separate CP4 scratch reuse was also rejected before metering because cap-active advanced-indexed factors changed 39/64 retained-B entries by at most 2.22e-16. One frozen, independently audited board attempt is preregistered as an exploratory accuracy measurement with the moment HOLD disclosed; it is not a scientific promotion and permits no retry or target adaptation. That upload became submission 328199 and passed the 10-worker smoke test, but the full evaluation failed operationally: only MLPs 0 and 12 completed, 12 rows explicitly hit TIME_EXHAUSTED, and 86 further rows were closed without telemetry. The two completed public MLPs provide partial, not aggregate, e-8-class evidence: final-layer raw MSE 7.661e-8 and 7.071e-8 (completed-only mean 7.366e-8). Their board predict walls were 119.415 and 118.927 seconds with 116.966 and 116.450 seconds in backend kernels, while the explicit timeout rows ranged through 124.570 seconds. Thus that V4 form left P2R0E4 open on board wall feasibility at that point. The frozen A1 then removed the complete CP4 history while leaving H4+alpha to absorb the full corrected quartic reset. It passed source and literal resource gates at 1,943,806,562,276 FLOPs (88.3941% of budget), 4,287,905,792 bytes sampled descendant RSS, finite output, and 0.251630 seconds residual. But it did not establish wall safety: local backend time rose from 34.358040 to 35.072057 seconds and wall rose from 35.341105 to 35.795832 seconds. A later canonical-suite audit strengthened that rejection: A1 and its exact CP4-disabled reference fail the frozen pure-0.35Q fixture at the PSD eigensolve; its original verifier had substituted residual-identity forms. Full-V4 Q24 instead retains CP4/H4/L4/L3-squared and all 17 channels while shortening only the slow Hermite/rho term axis from q=0..32 to q=0..24. It passed the literal meter at 2,131,249,329,109 FLOPs (96.9180%), 28.354738 seconds backend, 29.222195 seconds wall, 0.294491 seconds residual, and 4,507,058,176 bytes sampled descendant RSS. Three fixed production-shape official iid-Gaussian-He seeds were finite and differed from V4 by only 2.78e-17, 9.45e-17, and 3.46e-17 final prediction MSE, so its one-attempt package gate passed. Off-contract small widths can still diverge or become nonfinite, forbidding any general Q24/Q32 science-equivalence claim. The Q24 archive then consumed its sole exploratory attempt as submission 328210. The independently audited terminal receipt records a complete 100/100 operational pass with zero failures and all board resource gates satisfied. Its exact final-layer raw MSE is 5.765125013112993e-7, so the 1e-8 accuracy target was not met; the public 50-row median is 8.5555e-8, and five rows above 5e-7 contribute 86.7525% of the displayed public raw sum. This terminal operational result closes the implementation/runtime gate but does not establish scientific promotion. A separately frozen manual left-associated rho-power spelling is bitwise parent-equivalent and passed its one-use target-free full-shape meter at 2,131,280,786,389 FLOPs, 25.132427 seconds wall, 0.377093 seconds residual, 4,767,105,024 bytes descendant RSS, and 7,208 calls. It is a solved runtime component only: no raw-MSE improvement, tail repair, package, or submission is claimed. Two target-free fallback forms are now closed. The Q32 fp32 pair-kernel successor is numerically unstable (official-He width 8 eigensolve failure, width 32 nonfinite, and width 64 final MSE 9.5719899e10 versus V4). The exact audited A2 third-only source is likewise rejected after its registered official-He width-32 fixture became nonfinite; neither fallback opened targets, ran production shape, packaged, or submitted.

Program source is in [`code/`](code/), preserving the relative layout below `.`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_manual_rho_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_manual_rho_v1/estimator.py)
- [code/estimator.py](code/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_a1_no_cp4_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_a1_no_cp4_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_a1_no_cp4_v1/verify_a1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_a1_no_cp4_v1/verify_a1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_a1_no_cp4_v1/verify_exact_static_delta.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_a1_no_cp4_v1/verify_exact_static_delta.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_manual_rho_v1/verify_candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_manual_rho_v1/verify_candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_v1/build_q24_from_v4.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_v1/build_q24_from_v4.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_v1/compare_target_free_v4_q24.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_v1/compare_target_free_v4_q24.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_v1/run_additional_fullwidth_seeds.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_v1/run_additional_fullwidth_seeds.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_v1/verify_resources.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_v1/verify_resources.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_v1_board_promotion_v1/verify_board_promotion_scaffold.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_v1_board_promotion_v1/verify_board_promotion_scaffold.py)

<details>
<summary>Additional programs and dependencies (26)</summary>

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v1/coherent_quartic.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v1/coherent_quartic.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v1/resource_ledger.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v1/resource_ledger.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v1/verify_algebra.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v1/verify_algebra.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v1/verify_moment_feasibility.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v1/verify_moment_feasibility.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v2/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v2/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v3/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v3/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v4/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v4/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v4_board_promotion_v1/verify_board_promotion.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v4_board_promotion_v1/verify_board_promotion.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v4_moment_diagnostic/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v4_moment_diagnostic/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v4_moment_diagnostic/run_target_free_diagnostic.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v4_moment_diagnostic/run_target_free_diagnostic.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v4_moment_diagnostic_v2/build_v2_from_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v4_moment_diagnostic_v2/build_v2_from_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v4_moment_diagnostic_v2/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v4_moment_diagnostic_v2/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v4_moment_diagnostic_v2/run_target_free_diagnostic_v2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v4_moment_diagnostic_v2/run_target_free_diagnostic_v2.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v4_moment_diagnostic_v2/verify_parity_v2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v4_moment_diagnostic_v2/verify_parity_v2.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v4_moment_diagnostic_v2/verify_static_v2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v4_moment_diagnostic_v2/verify_static_v2.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/meter_protocol/run_extended_wall_diagnostic.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/meter_protocol/run_extended_wall_diagnostic.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/meter_protocol/run_synthetic_meter.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/meter_protocol/run_synthetic_meter.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/meter_protocol/verify_meter_run.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/meter_protocol/verify_meter_run.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/v4_workbench_baselines/estimator_3ddddd87facd110d4f9a4bccaa865d667b59d919f6df8d7217534483e197847c.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/v4_workbench_baselines/estimator_3ddddd87facd110d4f9a4bccaa865d667b59d919f6df8d7217534483e197847c.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/v4_workbench_baselines/estimator_521cc2e917cb597e3b3ff4c4055f5e867b1c891d043ff4b7180abbcec63ad79c.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/v4_workbench_baselines/estimator_521cc2e917cb597e3b3ff4c4055f5e867b1c891d043ff4b7180abbcec63ad79c.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/v4_workbench_baselines/estimator_8883b2ccd8fce6f7993b32af615041ff9ec915c179f735a782ac5e7c86199734.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/v4_workbench_baselines/estimator_8883b2ccd8fce6f7993b32af615041ff9ec915c179f735a782ac5e7c86199734.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/v4_workbench_baselines/estimator_b9e88777b1b8838f8c56afce14c23fe39dbcb4ef1aca0af439d58a3efe7815a2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/v4_workbench_baselines/estimator_b9e88777b1b8838f8c56afce14c23fe39dbcb4ef1aca0af439d58a3efe7815a2.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/v4_workbench_baselines/estimator_c9fcb9ffd4c412baa1422020179641b50fe6687078f06cea94fc63a9b2968293.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/v4_workbench_baselines/estimator_c9fcb9ffd4c412baa1422020179641b50fe6687078f06cea94fc63a9b2968293.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/v4_workbench_baselines/estimator_e76351b4f6565fce084d3024ad3c82d350bcd23f85dfb2d531a0b7231bd92ad2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/v4_workbench_baselines/estimator_e76351b4f6565fce084d3024ad3c82d350bcd23f85dfb2d531a0b7231bd92ad2.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/v4_workbench_baselines/section_timing_888.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/v4_workbench_baselines/section_timing_888.py)

</details>

## Evidence

- [FULL_BUDGET_CLOSURE_DIRECTIVE.md#L1](../../../records/research/2026-08-23-width-equivariant-residual-observer-32-channels-over-four-weight/FULL_BUDGET_CLOSURE_DIRECTIVE.md#L1)
- [FULL_BUDGET_COST_REOPEN_AUDIT.md#L1](FULL_BUDGET_COST_REOPEN_AUDIT.md#L1)
- [FULL_BUDGET_CP4_CAP2048_RESIDUAL_CARRIER_AUDIT.md#L1](FULL_BUDGET_CP4_CAP2048_RESIDUAL_CARRIER_AUDIT.md#L1)
- [SOURCE_MANIFEST.json#L1](SOURCE_MANIFEST.json#L1)
- [INDEPENDENT_SOURCE_AUDIT.json#L1](INDEPENDENT_SOURCE_AUDIT.json#L1)
- [raw_whest_stdout.json#L1](raw_whest_stdout.json#L1)
- [raw_process_tree_rss.jsonl#L6889](raw_process_tree_rss.jsonl#L6889)
- [receipt.json#L586](receipt.json#L586)
- [SOURCE_MANIFEST-cc6cdee4.json#L1](SOURCE_MANIFEST-cc6cdee4.json#L1)
- [FREEZE_RECEIPT_V2.json#L1](FREEZE_RECEIPT_V2.json#L1)
- [INDEPENDENT_SOURCE_AUDIT_V2.json#L1](INDEPENDENT_SOURCE_AUDIT_V2.json#L1)
- [receipt-05a8ac5b.json#L567](receipt-05a8ac5b.json#L567)
- [SOURCE_MANIFEST-d0d19048.json#L1](SOURCE_MANIFEST-d0d19048.json#L1)
- [FREEZE_RECEIPT_V3.json#L1](FREEZE_RECEIPT_V3.json#L1)
- [INDEPENDENT_SOURCE_AUDIT_V3.json#L1](INDEPENDENT_SOURCE_AUDIT_V3.json#L1)
- [receipt-55c37d61.json#L611](receipt-55c37d61.json#L611)
- [REJECTED_AGGRESSIVE_PUSH.json#L33](REJECTED_AGGRESSIVE_PUSH.json#L33)
- [receipt-ddb29af8.json#L1](receipt-ddb29af8.json#L1)
- [CPROFILE_C9_RESIDUAL_FINDING.json#L26](CPROFILE_C9_RESIDUAL_FINDING.json#L26)
- [receipt-ac439366.json#L1](receipt-ac439366.json#L1)
- [CPROFILE_ALL17_888_RESIDUAL_FINDING.json#L42](CPROFILE_ALL17_888_RESIDUAL_FINDING.json#L42)
- [SECTION_TIMING_888_FINDING.json#L45](SECTION_TIMING_888_FINDING.json#L45)
- [receipt-b240a698.json#L598](receipt-b240a698.json#L598)
- [receipt-236bc177.json#L622](receipt-236bc177.json#L622)
- [REJECTED_E763_CAP_STRIDED_CP_SCRATCH.json#L1](REJECTED_E763_CAP_STRIDED_CP_SCRATCH.json#L1)
- [receipt-14d6eeb5.json#L623](receipt-14d6eeb5.json#L623)
- [SOURCE_MANIFEST-0cad2cc4.json#L69](SOURCE_MANIFEST-0cad2cc4.json#L69)
- [FREEZE_RECEIPT_V4.json#L56](FREEZE_RECEIPT_V4.json#L56)
- [INDEPENDENT_SOURCE_AUDIT_V4.json#L172](INDEPENDENT_SOURCE_AUDIT_V4.json#L172)
- [SOURCE_MANIFEST-ab774cb8.json#L1](SOURCE_MANIFEST-ab774cb8.json#L1)
- [FREEZE_RECEIPT.json#L1](FREEZE_RECEIPT.json#L1)
- [receipt-93124980.json#L1](receipt-93124980.json#L1)
- [SOURCE_MANIFEST_V2.json#L1](SOURCE_MANIFEST_V2.json#L1)
- [FREEZE_RECEIPT_V2-faa670bb.json#L1](FREEZE_RECEIPT_V2-faa670bb.json#L1)
- [INDEPENDENT_SOURCE_AUDIT_V2-fc02c4b6.json#L1](INDEPENDENT_SOURCE_AUDIT_V2-fc02c4b6.json#L1)
- [receipt-4a792924.json#L1136](receipt-4a792924.json#L1136)
- [POST_REPLAY_FINDING_V2.json#L1](POST_REPLAY_FINDING_V2.json#L1)
- [INDEPENDENT_POST_REPLAY_AUDIT.json#L1](INDEPENDENT_POST_REPLAY_AUDIT.json#L1)
- [REJECTED_8D7E_CP4_DROPPED_LAYOUT.json#L1](REJECTED_8D7E_CP4_DROPPED_LAYOUT.json#L1)
- [BOARD_EXPLORATORY_PREREG_V1.json#L28](BOARD_EXPLORATORY_PREREG_V1.json#L28)
- [ROOT_PREREG_INDEPENDENT_AUDIT.json#L1](ROOT_PREREG_INDEPENDENT_AUDIT.json#L1)
- [PROMOTION_MANIFEST.json#L69](PROMOTION_MANIFEST.json#L69)
- [BOARD_ATTEMPT_328199_RESULT.json#L90](BOARD_ATTEMPT_328199_RESULT.json#L90)
- [INDEPENDENT_BOARD_RESULT_AUDIT.json#L38](INDEPENDENT_BOARD_RESULT_AUDIT.json#L38)
- [PREREGISTER.md#L21](PREREGISTER.md#L21)
- [SOURCE_HOLD.json#L1](SOURCE_HOLD.json#L1)
- [EXACT_STATIC_DELTA.json#L1](../../flop-accounting/known-arithmetic-reductions-fourth-cumulant-rank-cuts-rejected-measured/EXACT_STATIC_DELTA.json#L1)
- [INDEPENDENT_SOURCE_AND_METER_AUDIT_A1.json#L135](INDEPENDENT_SOURCE_AND_METER_AUDIT_A1.json#L135)
- [receipt-45b546c3.json#L597](receipt-45b546c3.json#L597)
- [SOURCE_MANIFEST-dffd9eba.json#L117](SOURCE_MANIFEST-dffd9eba.json#L117)
- [FREEZE_RECEIPT_A1.json#L82](FREEZE_RECEIPT_A1.json#L82)
- [INDEPENDENT_CANONICAL_FIXTURE_ERRATUM_A1.json#L1](INDEPENDENT_CANONICAL_FIXTURE_ERRATUM_A1.json#L1)
- [PREREGISTER-a7b54086.md#L16](PREREGISTER-a7b54086.md#L16)
- [SOURCE_MANIFEST-1627632a.json#L101](SOURCE_MANIFEST-1627632a.json#L101)
- [FREEZE_RECEIPT_Q24.json#L85](FREEZE_RECEIPT_Q24.json#L85)
- [INDEPENDENT_SOURCE_AND_METER_AUDIT_Q24.json#L149](../../flop-accounting/known-arithmetic-reductions-fourth-cumulant-rank-cuts-rejected-measured/INDEPENDENT_SOURCE_AND_METER_AUDIT_Q24.json#L149)
- [receipt-d69fd338.json#L603](receipt-d69fd338.json#L603)
- [receipt-70c8bbc1.json#L202](receipt-70c8bbc1.json#L202)
- [receipt-777dfdf5.json#L1](receipt-777dfdf5.json#L1)
- [INDEPENDENT_FULLWIDTH_COMPARATOR_AUDIT_Q24.json#L163](INDEPENDENT_FULLWIDTH_COMPARATOR_AUDIT_Q24.json#L163)
- [PACKAGE_MANIFEST.json#L1](PACKAGE_MANIFEST.json#L1)
- [PHASE2_WEIGHT_GENERATION_LAW_AUDIT.json#L1](PHASE2_WEIGHT_GENERATION_LAW_AUDIT.json#L1)
- [PROMOTION_MANIFEST-850be6d2.json#L66](PROMOTION_MANIFEST-850be6d2.json#L66)
- [INDEPENDENT_FINAL_READY_AUDIT.json#L149](INDEPENDENT_FINAL_READY_AUDIT.json#L149)
- [BOARD_ATTEMPT_328210_IN_FLIGHT.json#L1](BOARD_ATTEMPT_328210_IN_FLIGHT.json#L1)
- [BOARD_ATTEMPT_328210_PROGRESS_65.json#L1](BOARD_ATTEMPT_328210_PROGRESS_65.json#L1)
- [BOARD_ATTEMPT_328210_RESULT.json#L1](BOARD_ATTEMPT_328210_RESULT.json#L1)
- [INDEPENDENT_BOARD_RESULT_AUDIT_328210.json#L1](INDEPENDENT_BOARD_RESULT_AUDIT_328210.json#L1)
- [PREREGISTER.md#L22](../order-24-correlation-power-recurrence-identical-outputs-residual-time/PREREGISTER.md#L22)
- [SOURCE_MANIFEST.json#L1](../order-24-correlation-power-recurrence-identical-outputs-residual-time/SOURCE_MANIFEST.json#L1)
- [FREEZE_RECEIPT_MANUAL_RHO.json#L1](../order-24-correlation-power-recurrence-identical-outputs-residual-time/FREEZE_RECEIPT_MANUAL_RHO.json#L1)
- [INDEPENDENT_STRICT_AUDIT_MANUAL_RHO.json#L1](../order-24-correlation-power-recurrence-identical-outputs-residual-time/INDEPENDENT_STRICT_AUDIT_MANUAL_RHO.json#L1)
- [ROOT_LITERAL_METER_RELEASE_MANUAL_RHO.json#L1](../order-24-correlation-power-recurrence-identical-outputs-residual-time/ROOT_LITERAL_METER_RELEASE_MANUAL_RHO.json#L1)
- [receipt.json#L1](../order-24-correlation-power-recurrence-identical-outputs-residual-time/receipt.json#L1)
- [ROOT_LITERAL_METER_RESULT_MANUAL_RHO.json#L1](../order-24-correlation-power-recurrence-identical-outputs-residual-time/ROOT_LITERAL_METER_RESULT_MANUAL_RHO.json#L1)
- [REJECTION_RECEIPT.json#L1](REJECTION_RECEIPT.json#L1)
- [REJECTION_RECEIPT-468120ba.json#L1](REJECTION_RECEIPT-468120ba.json#L1)
- [INDEPENDENT_NAMESPACE_RECHECK_V4.json#L1](INDEPENDENT_NAMESPACE_RECHECK_V4.json#L1)
- [OFFICIAL_PHASE2_LAUNCH_UPDATE.md#L1](../../../records/research/2026-08-23-phase-2-evaluation-contract-width-1024-depth-16-2/OFFICIAL_PHASE2_LAUNCH_UPDATE.md#L1)
- [INDEPENDENT_LAYER3_ATTRIBUTION_DESIGN_AUDIT_V5.json#L1](INDEPENDENT_LAYER3_ATTRIBUTION_DESIGN_AUDIT_V5.json#L1)

<details>
<summary>All result files and records (80)</summary>

- [FULL_BUDGET_CLOSURE_DIRECTIVE.md](../../../records/research/2026-08-23-width-equivariant-residual-observer-32-channels-over-four-weight/FULL_BUDGET_CLOSURE_DIRECTIVE.md) · record
- [FULL_BUDGET_COST_REOPEN_AUDIT.md](FULL_BUDGET_COST_REOPEN_AUDIT.md) · record
- [FULL_BUDGET_CP4_CAP2048_RESIDUAL_CARRIER_AUDIT.md](FULL_BUDGET_CP4_CAP2048_RESIDUAL_CARRIER_AUDIT.md) · record
- [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) · result
- [INDEPENDENT_SOURCE_AUDIT.json](INDEPENDENT_SOURCE_AUDIT.json) · result
- [raw_whest_stdout.json](raw_whest_stdout.json) · result
- [raw_process_tree_rss.jsonl](raw_process_tree_rss.jsonl) · log
- [receipt.json](receipt.json) · result
- [SOURCE_MANIFEST-cc6cdee4.json](SOURCE_MANIFEST-cc6cdee4.json) · result
- [FREEZE_RECEIPT_V2.json](FREEZE_RECEIPT_V2.json) · result
- [INDEPENDENT_SOURCE_AUDIT_V2.json](INDEPENDENT_SOURCE_AUDIT_V2.json) · result
- [receipt-05a8ac5b.json](receipt-05a8ac5b.json) · result
- [SOURCE_MANIFEST-d0d19048.json](SOURCE_MANIFEST-d0d19048.json) · result
- [FREEZE_RECEIPT_V3.json](FREEZE_RECEIPT_V3.json) · result
- [INDEPENDENT_SOURCE_AUDIT_V3.json](INDEPENDENT_SOURCE_AUDIT_V3.json) · result
- [receipt-55c37d61.json](receipt-55c37d61.json) · result
- [REJECTED_AGGRESSIVE_PUSH.json](REJECTED_AGGRESSIVE_PUSH.json) · result
- [receipt-ddb29af8.json](receipt-ddb29af8.json) · result
- [CPROFILE_C9_RESIDUAL_FINDING.json](CPROFILE_C9_RESIDUAL_FINDING.json) · result
- [receipt-ac439366.json](receipt-ac439366.json) · result
- [CPROFILE_ALL17_888_RESIDUAL_FINDING.json](CPROFILE_ALL17_888_RESIDUAL_FINDING.json) · result
- [SECTION_TIMING_888_FINDING.json](SECTION_TIMING_888_FINDING.json) · result
- [receipt-b240a698.json](receipt-b240a698.json) · result
- [receipt-236bc177.json](receipt-236bc177.json) · result
- [REJECTED_E763_CAP_STRIDED_CP_SCRATCH.json](REJECTED_E763_CAP_STRIDED_CP_SCRATCH.json) · result
- [receipt-14d6eeb5.json](receipt-14d6eeb5.json) · result
- [SOURCE_MANIFEST-0cad2cc4.json](SOURCE_MANIFEST-0cad2cc4.json) · result
- [FREEZE_RECEIPT_V4.json](FREEZE_RECEIPT_V4.json) · result
- [INDEPENDENT_SOURCE_AUDIT_V4.json](INDEPENDENT_SOURCE_AUDIT_V4.json) · result
- [SOURCE_MANIFEST-ab774cb8.json](SOURCE_MANIFEST-ab774cb8.json) · result
- [FREEZE_RECEIPT.json](FREEZE_RECEIPT.json) · result
- [receipt-93124980.json](receipt-93124980.json) · result
- [SOURCE_MANIFEST_V2.json](SOURCE_MANIFEST_V2.json) · result
- [FREEZE_RECEIPT_V2-faa670bb.json](FREEZE_RECEIPT_V2-faa670bb.json) · result
- [INDEPENDENT_SOURCE_AUDIT_V2-fc02c4b6.json](INDEPENDENT_SOURCE_AUDIT_V2-fc02c4b6.json) · result
- [receipt-4a792924.json](receipt-4a792924.json) · result
- [POST_REPLAY_FINDING_V2.json](POST_REPLAY_FINDING_V2.json) · result
- [INDEPENDENT_POST_REPLAY_AUDIT.json](INDEPENDENT_POST_REPLAY_AUDIT.json) · result
- [REJECTED_8D7E_CP4_DROPPED_LAYOUT.json](REJECTED_8D7E_CP4_DROPPED_LAYOUT.json) · result
- [BOARD_EXPLORATORY_PREREG_V1.json](BOARD_EXPLORATORY_PREREG_V1.json) · result
- [ROOT_PREREG_INDEPENDENT_AUDIT.json](ROOT_PREREG_INDEPENDENT_AUDIT.json) · result
- [PROMOTION_MANIFEST.json](PROMOTION_MANIFEST.json) · result
- [BOARD_ATTEMPT_328199_RESULT.json](BOARD_ATTEMPT_328199_RESULT.json) · result
- [INDEPENDENT_BOARD_RESULT_AUDIT.json](INDEPENDENT_BOARD_RESULT_AUDIT.json) · result
- [PREREGISTER.md](PREREGISTER.md) · record
- [SOURCE_HOLD.json](SOURCE_HOLD.json) · result
- [EXACT_STATIC_DELTA.json](../../flop-accounting/known-arithmetic-reductions-fourth-cumulant-rank-cuts-rejected-measured/EXACT_STATIC_DELTA.json) · result
- [INDEPENDENT_SOURCE_AND_METER_AUDIT_A1.json](INDEPENDENT_SOURCE_AND_METER_AUDIT_A1.json) · result
- [receipt-45b546c3.json](receipt-45b546c3.json) · result
- [SOURCE_MANIFEST-dffd9eba.json](SOURCE_MANIFEST-dffd9eba.json) · result
- [FREEZE_RECEIPT_A1.json](FREEZE_RECEIPT_A1.json) · result
- [INDEPENDENT_CANONICAL_FIXTURE_ERRATUM_A1.json](INDEPENDENT_CANONICAL_FIXTURE_ERRATUM_A1.json) · result
- [PREREGISTER-a7b54086.md](PREREGISTER-a7b54086.md) · record
- [SOURCE_MANIFEST-1627632a.json](SOURCE_MANIFEST-1627632a.json) · result
- [FREEZE_RECEIPT_Q24.json](FREEZE_RECEIPT_Q24.json) · result
- [INDEPENDENT_SOURCE_AND_METER_AUDIT_Q24.json](../../flop-accounting/known-arithmetic-reductions-fourth-cumulant-rank-cuts-rejected-measured/INDEPENDENT_SOURCE_AND_METER_AUDIT_Q24.json) · result
- [receipt-d69fd338.json](receipt-d69fd338.json) · result
- [receipt-70c8bbc1.json](receipt-70c8bbc1.json) · result
- [receipt-777dfdf5.json](receipt-777dfdf5.json) · result
- [INDEPENDENT_FULLWIDTH_COMPARATOR_AUDIT_Q24.json](INDEPENDENT_FULLWIDTH_COMPARATOR_AUDIT_Q24.json) · result
- [PACKAGE_MANIFEST.json](PACKAGE_MANIFEST.json) · result
- [PHASE2_WEIGHT_GENERATION_LAW_AUDIT.json](PHASE2_WEIGHT_GENERATION_LAW_AUDIT.json) · result
- [PROMOTION_MANIFEST-850be6d2.json](PROMOTION_MANIFEST-850be6d2.json) · result
- [INDEPENDENT_FINAL_READY_AUDIT.json](INDEPENDENT_FINAL_READY_AUDIT.json) · result
- [BOARD_ATTEMPT_328210_IN_FLIGHT.json](BOARD_ATTEMPT_328210_IN_FLIGHT.json) · result
- [BOARD_ATTEMPT_328210_PROGRESS_65.json](BOARD_ATTEMPT_328210_PROGRESS_65.json) · result
- [BOARD_ATTEMPT_328210_RESULT.json](BOARD_ATTEMPT_328210_RESULT.json) · result
- [INDEPENDENT_BOARD_RESULT_AUDIT_328210.json](INDEPENDENT_BOARD_RESULT_AUDIT_328210.json) · result
- [PREREGISTER.md](../order-24-correlation-power-recurrence-identical-outputs-residual-time/PREREGISTER.md) · record
- [SOURCE_MANIFEST.json](../order-24-correlation-power-recurrence-identical-outputs-residual-time/SOURCE_MANIFEST.json) · result
- [FREEZE_RECEIPT_MANUAL_RHO.json](../order-24-correlation-power-recurrence-identical-outputs-residual-time/FREEZE_RECEIPT_MANUAL_RHO.json) · result
- [INDEPENDENT_STRICT_AUDIT_MANUAL_RHO.json](../order-24-correlation-power-recurrence-identical-outputs-residual-time/INDEPENDENT_STRICT_AUDIT_MANUAL_RHO.json) · result
- [ROOT_LITERAL_METER_RELEASE_MANUAL_RHO.json](../order-24-correlation-power-recurrence-identical-outputs-residual-time/ROOT_LITERAL_METER_RELEASE_MANUAL_RHO.json) · result
- [receipt.json](../order-24-correlation-power-recurrence-identical-outputs-residual-time/receipt.json) · result
- [ROOT_LITERAL_METER_RESULT_MANUAL_RHO.json](../order-24-correlation-power-recurrence-identical-outputs-residual-time/ROOT_LITERAL_METER_RESULT_MANUAL_RHO.json) · result
- [REJECTION_RECEIPT.json](REJECTION_RECEIPT.json) · result
- [REJECTION_RECEIPT-468120ba.json](REJECTION_RECEIPT-468120ba.json) · result
- [INDEPENDENT_NAMESPACE_RECHECK_V4.json](INDEPENDENT_NAMESPACE_RECHECK_V4.json) · result
- [OFFICIAL_PHASE2_LAUNCH_UPDATE.md](../../../records/research/2026-08-23-phase-2-evaluation-contract-width-1024-depth-16-2/OFFICIAL_PHASE2_LAUNCH_UPDATE.md) · record
- [INDEPENDENT_LAYER3_ATTRIBUTION_DESIGN_AUDIT_V5.json](INDEPENDENT_LAYER3_ATTRIBUTION_DESIGN_AUDIT_V5.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-24-order-32-high-order-closure-literal-run-exceeded-wall) · [Copy provenance](../../../PROVENANCE.json)

