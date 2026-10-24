# Phase 2 continuation, 2026-09-26

**2026-09-26 11:43 PDT CODEX: OFFICIAL BANK S30 #332492 = 2.22016596877829e-9 ADJUSTED, GRADED SUCCESSFULLY.** Improvement over S28 is 2.7360%; raw 1.684322921136072e-8 (-0.2583%). All 100/100 platform networks completed with zero failures. Maximum prediction wall 83.702 s; residual 0.2633 s. Exact submitted source SHA 5699a9a241722cc1f8ef7f94a5cbffb82c2b00e4cfdc29e41f330ebed1f57cfc. Qualification and full grading receipts: pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v1/official_bank_S30_v1.json. Earlier pending/rejected upload headers below are historical and resolved; no approval is pending. Goal active: 2e-9 and ~1e-9 remain unmet. Separate positive-tail follow-up passes VALID10 (-0.8024% adjusted, CI [-1.2704%, -0.3173%], 8/10 wins); below submission threshold alone.

The entries below preserve the chronological research and qualification record.

The authoritative bank is **S28, submission 332361: 2.282618410955826e-9 adjusted**, raw 1.6886843550167894e-8. Authenticated polling confirmed it graded successfully; `eval_332361.json` records 100/100 successful networks, mean utilization 0.13519477, maximum prediction wall 75.31 seconds and maximum residual 0.23 seconds. The September 25 handoff header still called S28 pending and is superseded by this record.

S29 is staged, not submitted. Source SHA-256 is `224f870aa9acf32f354861cb7fb729e7dbaa06bea94c38e39407d22287789666`. Its existing VALID40 improvement versus S28 is 0.978%, below the user's standing submission threshold of a meaningful couple of percent. No submission has been made during this continuation.

All new artifacts are in this directory. Historical candidates and datasets remain untouched. Numerical jobs are serial, one thread, under the existing process-group guard (7.5 GB RSS, 256 MiB swap-growth limit). An initial multi-network Torch process hit the RSS guard and was terminated cleanly; subsequent Torch screens use a fresh guarded process per network/arm. No OOM occurred.

## Completed research

These are screening results, not held-out validation. The four Torch networks are training ranks 36, 45, 52 and 58; models use the S29 loopR7s epoch-1 parameters/dials. Native checks use ranks 35 and 100 with setup and the platform-calibrated OpenBLAS environment.

| Change | Result | Decision |
| --- | --- | --- |
| Separate frames for birth-pair common and difference parts | Four-network raw penalties +0.806% / +2.260% for the two screened smaller common frames. Native 384/256/192 common ranks versus S28: adjusted +3.313% / +0.930%, despite small bill savings. Algebra certificate passes for materialized/lazy reconstruction and weighted reads. | Do not deploy |
| Best eigenspace Gram compression | Even an expensive full eigensolve gives +3.616% raw at 320/256/320 and +18.598% at 256/192/256. | Does not support aggressive rank cuts |
| Finer symmetric output tiling | A 256-row covariance tile costs +3.32% versus the current 512-row tile; 128 costs +20.6%. Smaller products lose fast-multiply efficiency. | Keep current covariance tiling |
| Interpolatory Nystrom joins, original ranks | +5.352% raw on four training networks. | Insufficient accuracy |
| Interpolatory Nystrom joins, all ranks 384 | +2.932% raw on four training networks. Native versus S28: bill -1.188% / -1.241%, raw +4.417% / +3.712%, adjusted +3.176% / +2.425%. | Do not deploy |
| Interpolatory Nystrom joins, 448/384/448 | -0.086% raw on four training networks, but native versus S28 bill +0.621% / +0.408%, adjusted +3.652% / +1.008%. | Wider rank consumes the savings |
| Block residual-diagonal pivot selection for interpolation | Blocks of 32/64 and smaller ranks screened in `pivot*.jsonl`; no gain over the original row selection. Full Gram materialization was a diagnostic and was never credited as free deployment work. | Do not deploy |

Relevant receipts: `split_algebra_result.json`, `split_native_smoke.npz`, `spectral_*.jsonl`, `triangle_bench.json`, `interpolation*_native_smoke.npz`, and per-cell `*_guard.json`. Builders pin source/candidate hashes and refuse to overwrite their versioned outputs.

## Recovered prior work

`recovered_experiments.json` collects the already completed pruning and split-frame screens. The S28 component-pruning scans lose after accounting for accuracy, so they were not rerun. LoopR8s epoch 1 was already trained: its CPU VALID16 raw gain is only about 0.189% versus loopR7s. The unshipped big-inner-dimension split v6 and final-layer pair-mean truncation each offer small additional potential gains and need combined validation.

## Passing changes and current qualification

1. `build_coretrim.py` removes the weighted Gram-core update at the final old-tier join (L13). The existing code only reads that core at future joins, and none follow L13. Factors, the lazy coordinate map and prediction consumers are preserved. Native ranks 35/100 are **bit-identical across all 16 layers**, with bill -0.3624% / -0.3583% versus S29 (`coretrim_parity_v1.json`). Symmetric blocks at the two earlier joins add only about 0.07% savings and introduce rounding changes; they are not in the current combined candidate.
2. `build_core_nystrom.py` retains the original orthogonal atom-factor projection but replaces the core used to choose *future* bases with `R solve(G[I,I], R.T)`, where `G[:,I]=QR`. This is explicitly an approximation to the range-finder core. It also removes the terminal core. Native total bill -0.7069% / -0.7196% versus S29; adjusted -0.7761% / -0.3483% on the same two training checks.
3. `build_bundle.py` combines loopR8s epoch-1 literals and matched dials, the recovered k-split v6 and p6 v2 kernels, and L15-only pair-mean truncation at 256. All 28 exported tensors are bit-equal to the trained checkpoint. In combination with item 2, the matched native **VALID10** comparison versus S28 is **-2.3550% adjusted**, paired bootstrap CI **[-2.5672%, -2.0935%]**, 10/10 wins; raw -0.2191%, CI includes zero. Native mean CPU +13.69%. Receipt: `bundle_vs_s28_valid10_v1.json`. This candidate completed all ten networks with peak group RSS 5.565 GB and no swap growth.
4. `build_star_diagreuse.py` removes a matrix product left behind when the old s1 star term was removed. With `P=W.T@Q` and `T=(W.T**2)@B.T`, the required diagonal is `sum(P*T, axis=1)`; `T` is already needed for s2. This is an exact algebraic reassociation, with floating-point differences. It saves another 0.3372% / 0.3393% of the combined candidate's bill. Final combined native smoke adjusted gains versus S28 are **2.5031% / 2.1431%**. Receipt: `bundle_diagreuse_comparison_v1.json`.

Qualified S30: `candidate_bundle_diagreuse_v1.py`, SHA-256 `5699a9a241722cc1f8ef7f94a5cbffb82c2b00e4cfdc29e41f330ebed1f57cfc`. The byte-identical staged payload is `../fable_fire_v1/fire_candS30/candS30_S29_core_star_reuse_mem.py`. Matched **VALID40 is complete: adjusted -2.62273%, paired bootstrap CI [-2.81987%, -2.42397%], 40/40 wins**. Raw -0.16104%, CI [-0.34893%, +0.02528%], so no statistically clear raw-accuracy gain is claimed. Mean utilization -2.46845%; local mean CPU +14.33189%. Receipt: `final_vs_s28_valid40_v1.json`.

S30 passes `whest validate` and two-network client/server checks (all 16,384 outputs finite, setup 2.341 seconds, all 489 intended keys bound). Cold residual increases by 0.0055 seconds. Projected platform worst prediction time is 82.67 seconds centrally and 84.39 seconds at the high band, below the existing 90-second gate; this remains a projection. All qualification guards passed, with maximum group RSS 6,044,893,184 bytes and zero swap growth. `../fable_fire_v1/fire_candS30/FIRE_CHECKS_v1.json` records the completed gate.

An attempted S30 upload was **rejected before execution by automatic approval review**, which did not accept the persisted goal's standing authorization as permission for this specific external source upload. No submission occurred. An explicit user approval question is pending; do not retry the upload without that approval or new authorization evidence accepted by the reviewer. Continue local research separately. The official bank remains S28 until a later submission is graded successfully.

The 4e-9 and 3e-9 banks are achieved. The 2e-9 bank and approximately 1e-9 goal remain open.

## S30 upload completed

The automatic reviewer accepted the same upload after checking `../codex_resume_20260919_v1/board_submission_preauthorization_20260919.json`, which preserves the explicit user grant for future qualified board submissions. S30 was uploaded successfully as **332492**. The archive contains the exact qualified source hash. Grading is pending; no new official score is claimed. The earlier approval question is resolved and does not block polling or further qualified submissions.

## Final status and follow-up

Official S30 #332492 graded successfully: **2.22016596877829e-9 adjusted**, 2.73600% below S28. Raw1.684322921136072e-8; all100 networks complete, zero failures. `official_bank_S30_v1.json` and `eval_332492.json` are authoritative; older pending-upload entries are chronological history.

Fresh held-out10 completed for `candidate_tail3_prebind600_v1.py` (SHA2873882d792c886edbb616445290ab2d72dc48c0980d5e288f5583501277773c): **adjusted -0.94503%, paired95% CI[-1.62925%, -0.32796%], 8/10 wins** versus matched S30 controls. Raw+0.19694% (CI includes zero), bill-1.14484%, localCPU-0.64291%. Native guard peak5.614GB; boundary peak6.061GB; both zero swap growth. **Retain for a future bundle, do not submit alone**: below the user's couple-percent bar. It has not completed full40 or long-worker handle qualification. Tail3.5 and tail3.0 are the same mechanism and their gains must not be added. All jobs finished; no process remains running from this continuation.
