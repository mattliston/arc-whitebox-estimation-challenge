# Astra → Opus 5.5 FINAL handoff

Published **2026-10-11T03:32:13Z** from `date -u` (Oct 10, Pacific evening). The requested wind-down boundary is complete. **Nc was NOT ACCEPTED; no new candidate was exported, judged, built, released or submitted.** Our best remains S46 twin335261 at1.581165431246144e-9. The next scientific action is genuine EDBAND zero/nonzero/native qualification from standing S46, then the prepared24-gradient one-point experiment on fresh A30 selection data.

**Running at the final03:30:22UTC snapshot:** A30 reference producerPID17133 and public watcherPID66351. A30 had16/16 sizing,2/16 acceptance,0/32 future-final receipts; it continues producing. The completed Nc worker26407, guard26404, driver26556 and both launch parents are all retired. No agent teammate remains active. The user explicitly requested pausing the automated goal after this handoff; the final assistant message reports the actual pause result. Background OS jobs are intentionally left running.

**Read next:** `O/trainer/OPUS_HANDOFF_20261011T0328Z_v1.md` (SHA45125db9a437970d000e663649a9b6793bcf3ccd8e9d115cede301648de32223), `O/trainer/cert/cut_refit_Nc_launch_v1/FINAL_RECEIPT_v1.json` (SHA9614db536f412832c15be81aa30256b1e1f53756584d5fcde3c2a7dcb84251b6), and the newest~150 TEAM_LOG lines. Revalidate live processes before compute; the snapshot is not future resource authorization.

**Independent root verification:** `O/lead/opus_handoff_20261011T031005Z/` contains `source_pin_verification_v1.json` (24critical source hashes PASS), `nc_terminal_verification_v1.json` (19terminal/source/checkpoint pointers PASS; rejection, exit0 and whole-group retirement verified), `background_snapshot_v1.json`, and `winddown_request_v1.json`. All source/selection results, rejected trial and original draft are preserved.

## User's current instructions and handoff boundary

- Win Phase 2, 1024×16 MLPs, by 2026-10-17 23:59 UTC. Stack and submit verified 1–2% improvements over the previous best; do not require a 10% single-candidate gain. A moving leaderboard means a cumulative ~15% aspiration is sensible, but it is not a per-candidate gate.
- The user explicitly says not to spend time on designation. Do not let older master/goal wording override that.
- Maximum six active subagents including descendants. At publication all five lanes are completed/standby; trainer has delivered its final Opus note and ended its turn. Their background OS processes are separate from agent turns.
- The user requested a graceful switch to Opus, can wait for the current refit to finish, and explicitly said: **“pause when ready. give me the prompt to send to opus as well as making the hadnoff note.”**
- Completed boundary: Nc's ONE prescribed point was rejected, so no export/INPUT_MANIFEST was created. Trainer recorded terminal evidence and ended its turn. No final judge, native qualification, release or next EDBAND experiment is to be launched by Astra after that boundary. The user will resume with Opus.
- Leave the authorized A30 reference generator and public watchers running through the model switch. The automated goal is to be paused immediately after final publication and verification, as explicitly requested. Pausing the goal does not kill OS jobs.

## How to read this handoff

1. Read this FINAL note and `O/lead/opus_handoff_20261011T031005Z/nc_terminal_verification_v1.json`.
2. Read the newest ~150 lines of `O/TEAM_LOG.md` and `O/trainer/OPUS_HANDOFF_20261011T0328Z_v1.md`.
3. Read the exact receipts/sources for whichever next branch actually applies. Do not restart source-preparation campaigns already closed below.
4. `ASTRA_TAKEOVER_20261010.md` is useful historical context; the newer source pins, receipts and outcomes below supersede its live-state descriptions. Original lane `TAKEOVER_ASTRA.md` files are older still.

Path abbreviations throughout:

```
R = /Users/matt/arc-white-box-estimation
O = R/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
T = O/trainer
PyTorch Python = /Users/matt/anaconda3/bin/python
repository Python = R/.venv/bin/python
```

## Rules that must survive the switch

- NEVER print, read, copy or expose `key.txt`. Only `O/fire/fire_once.py` submits and reads the key.
- Submitted inference must be flopscope-only; no underpriced-operation tricks. Inspect every new einsum signature and its actual operands before shipping.
- Never delete or overwrite important artifacts. Use new versions, exclusive creation and retained archives. The worktree is very dirty; do not reset/revert other work.
- Every TEAM_LOG timestamp comes from `date -u`, not inferred time or a model clock.
- At most 10 submission attempts per UTC day. Oct 11 was audited at zero attempts around 01:05 UTC and Astra has submitted none since; re-audit actual fire receipts immediately before firing. Oct 10 had six confirmed attempts.
- Fire gains only after untouched generated-network judging, native parity/measured cost, and release gates; safe + twin pair. Selection/check networks are not final judging evidence.
- 48 GB RAM, 16 physical cores (12P+4E), one GPU. Whole-pool memory gates, no OOM, no auto-restart or auto-scaling. Preserve release's quiet C/slow timing window.
- Use HTTP/CLI paths already working for leaderboard/submissions; no Safari dependency. Do not send external messages to people.
- Current approved cutoff/cut-simulator profile has all four flags true. Falling back to default `law_restore_right=false` is wrong.

## Latest leaderboard and best submitted pair

Authoritative exact board scan is `O/analyst/a11_status_20261011T0159Z.json`, `a5_leaders_20261011T0159Z.md`, and `scan_335271_plus_20261011T0159Z.jsonl`; it completed around 02:01:58 UTC, through ID335296. Rounded public page snapshots can lag.

| Rank | Team | Adjusted score | Submission |
|---|---|---|---|
| 1 | J2W | 1.4424709351216109e-9 | 335075 |
| 2 | marius_binner | 1.4628862759782258e-9 | 335281 |
| 3 | Luna | 1.5612079989513157e-9 | 335221 |
| 4 | mliston (us) | 1.581165431246144e-9 | 335261 |
| 5 | oliver_chen | 1.5978213806820937e-9 | 335274 |

Reduction from our current best needed at that snapshot: podium 1.26220%, second 7.48050%, first 8.77166%. Marius improved 3.11055% since the prior scan, almost entirely through bill reduction. Do not infer their algorithm from telemetry totals.

Our graded pair, both 100/100 clean:

- Safe **335259**, score1.5811980561457008e-9: `O/release/runs/a_s46_63f89e8d_safe_pz/submission.tar.gz`; archive SHA `42bf4a4e9c700cf128a5a7459f49a0986535a403193d5a3513f1722648b86964`; source `O/release/variants/s46_63f89e8d_safe_pz/candidate.py`, SHA `70f7857fdabbbca6398d44bb11dbf608268a7199c1f22e67d2e9ca31a514b540`.
- Twin **335261**, score1.581165431246144e-9: `O/release/runs/a_s46e_f89b8951_primary/submission.tar.gz`; archive SHA `50a8c1a3d631ee739f67b86c9705d7b1466dd17dfbd360a54fbda45be0c3f734`; source `O/release/variants/s46e_f89b8951_primary/candidate.py`, SHA `f89b89511e62eb5474348bcbfbd7cfedce1a4bbd03de8efa5a9a4fa230266fe5`.
- Graded receipt: `O/lead/takeover_20261010T232330Z/graded_pair.json`. `CHAMPION.json` is stale. Extra stage-D work on the already graded S46 pair was deferred; `release/s46_worker_pair_v1.py` was not launched.

## Nc: completed and NOT ACCEPTED

**Final decision03:27:57UTC: NOT ACCEPTED.** All352 unique requests completed:96 public gradients +256 evaluations, including all192 arm evaluations and exactly one fixed point. No errors, recycle or retry.

| Evidence | Pooled MSE change | SE of network percentage changes | Networks improved |
|---|---:|---:|---:|
| Sizing136–151 | −0.9117023509953137% | 0.3455137577220325% | 10/16 |
| Independent check152–167 | −0.2354527961318409% | 0.3403561212510006% | 10/16 |

The fixed rule required check change < −SE; −0.235453 is not < −0.340356. The forecast−0.988007%/LOO−0.395937% did not override the observed check. This is not a statistically accepted or release-qualified gain. S46 remains standing. **Do not export/promote the rejected trial, change the rule, resize it from these check outcomes, or reuse these selection networks as fresh evidence.** The failure of this one point is not a proof that every future cut-aware refit must fail; any new experiment needs a genuinely new prospective protocol/data allocation.

- Decision `T/cert/refit/jmNc_first_point.json`, SHA `9460962f9a797aa5a3c6dadc0301c1447bf16106520757db6c009b12d707e549`.
- Rejected trial `T/cert/refit/jmNc_p1.pt`, SHA `42510c2780e507a7e353d9b121008508314a0dc13f6eead691286c10aaed215b`; head `jmNc_p1_hu.json`, SHA `12b046f948473bb8ca0e15604d06475d82803591f7ac3b8895fac3f693ca48d8`.
- Starting/standing checkpoint/head hashes below are unchanged. No export manifest exists for rejected Nc. A27 final168–199 remains sealed and unjudged.
- Driver exit0 at03:27:57; guard/worker exit0 and whole_group_retired at03:27:58. Root independently confirmed all worker/driver/guard/parent PIDs absent03:28 and03:30. Guard peakRSS8,838,987,776bytes, footprint1,864,440,504bytes, swap growth0; elapsed10405.706s.
- `T/run_M/STOP` is preserved, containing03:27:57 first-point driver terminal exit0. Do not restart old supervisors/workers or duplicate Nc. Any later authorized worker must version/archive STOP and use its own qualified runtime/resource plan.
- Authoritative lane evidence: `T/OPUS_HANDOFF_20261011T0328Z_v1.md`, `T/cert/cut_refit_Nc_launch_v1/FINAL_RECEIPT_v1.json`, `worker_terminal.json`, `driver_terminal.json`, `worker_guard.json`, immutable worker/driver logs. The old tool sessions72882/19567 are closed (trainer observed exit0; root later got unknown-session because already closed), not hung jobs.

Historical handles and scientific inputs from the completed run (do not restart):

- Worker26407, guard26404, driver26556; worker exec session72882, driver session19567. Started00:34:37 UTC. One8-thread worker, four-hour guard, no auto restart/scaling.
- `T/cert/cut_refit_Nc_launch_v1/{worker_handle.json,driver_handle.json,worker.log,driver.log,prelaunch.json}`.
- Launcher `T/cert/launch_cut_firstpoint_v2.py`; driver `T/cert/refit/jmix24.sh`; protocolv6; solver `jmix_generated_v4.py`.
- All96 public gradients completed02:09:13 UTC. All16 sizing and16 acceptance baselines completed. Actual frozen arms are12, not the older assumed9: `arrp, arre, h6, hb3, h20, h17, kce, kcl, dX1, dbscale, dc0, dk31`.
- Arms: `T/cert/refit/jmNc_d1/arms.json`, SHA `5ba9add451a7d2f30f3869c1efbfa1795c088822e0a2076e3d4d412b238e1ce2`.
- The completed protocol after192 direction evaluations was: solve ONE point, evaluate16 finite sizing +16 independent acceptance, record decision, STOP. Acceptance is `pooled_change_pct < -se_pct`, where pooled change uses the ratio of mean MSEs and SE uses per-network percentage changes divided by sqrt16. This is selection evidence, not final promotion confidence.
- Source fixed96: `T/cert/refit/astra_cut_first96_v1.txt`, SHA `666884abbc91276d50d9253b4311eeefbf2abbea862d67d81b4b12b2ed58bb02`; selection sidecar SHA `250753620c02cb6fad2fed02185e274eede0d0e71b587c5774cdc93e33076de4`. Selected from326 eligible public networks, all holdouts excluded, no gradient/result-driven subset selection.
- Starting checkpoint `jmI_p2_noAS.pt` SHA `f36e034c6196f7693b4e01c453f7e2044aca9b69010bd2f4ab5c2340022411e3`; HU `jmI_p2_hu.json` SHA `d5b16582cfec6f84ae3f50d75e3d854c91585da5284a83f3742a45af7e40fa7c`.
- Full request `T/cert/refit/jmN_request_common_v1.json`: certifiedv11 all flags true, ncar/f4c/B6/ss0. Never reconstruct an incomplete request from memory.
- Export helper `T/cert/export_firstpoint_v1.py` accepts an accepted terminal receipt and a NEW output directory. It creates immutable params/head/request/native_base plus `INPUT_MANIFEST.json`; exported does not mean judged, native-qualified or release-ready.
- Native DLY conversion must use `O/engineer/a52_pld/dly_from_trainer_v2.py`, all16 absolute values including lateiso/k4la1. Exporter `T/export_dials_s46_v2.py` preserves full arrays.

## Running reference production and public watcher

### A27 is COMPLETE and retired

- Old GPU PID21955 completed final seed199 and exited02:47:42 UTC. Do not restart it.
- `O/analyst/a27_fast_truth_20261011T0023Z` (`A27`). Plan SHA `71f455d1d7f5f7239503468cacd5710f4db9d595c7f1be389c8d937e4f64a979`; producer SHA `01185175b0f6c1948a88bf7816dad69dfbe7ea0c191c46a86d385b07423100ce`.
-136–151 sizing16 at64M;152–167 independent selection16 (152/15364M,154–16732M); **168–199 sealed final32 at N15,990,784**, all completion receipts ready. Final labels/predictions have not been inspected or judged by Astra.
- Metadata-only completion receipt `O/analyst/A35_A27_TO_A30_METADATA_HANDOFF_v1_20261011T024856Z.json`, SHA `f17237d01b05e996f59f4be3e2d1250b12ca503c5f9c751bea6fe56ca40296ad`.

### A30 is RUNNING — continue it through the switch

- PID **17133**, started02:49:01 UTC, nice10, no auto restart. `O/analyst/a30_edband_selection_and_final_20261011T0057Z` (`A30`).
- Exact `produce_v2.py` SHA `9b3737c3e9537ba43450d90658a1ab74499a37aaa71c52fd572f4ae383400698`; `plan_v2.json` SHA `b35e8a1c7c83d4018ea44b8b01464f96069cab02f1583f2e0bfbfbc9ee758f4a`.
- Fixed order:232–247 fresh sizing16,248–263 independent acceptance16, then200–231 sealed and still unassigned final32. All N15,990,784 (244 balanced half-batches), batch32768. Keep this one64-entry plan; old A29 final-only plan is superseded and unlaunched.
- Root launch evidence: `O/lead/a30_preflight_20261011T024821Z/` contains `resource_snapshot_v1.json`, `root_authorization_v1.json`, `launch_handle_v1.json`, `producer_v1.log`, `first_network_verification_v1.json`.
- Authorization SHA `acdc9e521c55c309117a1a87a7f185cf70dc8d7a61207466a673b332387f68c3`. Fresh free+inactive20.063629GiB passed17GiB conservative whole-pool gate (primary guard10+GPU3+OS4); shared gpu_request/STOP absent, no quiet or current refinement need.
- First seed232 completed02:51:09; metadata/source/auth verified without reading truth arrays. Startup receipt `O/analyst/A35_A30_STARTUP_RECEIPT_v1_20261011T024945Z.json`.
- Poll actual PID, `A30/state.jsonl`, `prod.jsonl` and receipt metadata. A file or old state alone does not prove a job is live. No restart merely because an observation timed out.
- `A30/STOP` is checked before each network. For planned release quiet, **create STOP exclusively at least five minutes before actual C/slow start; finish current network/publication, then verify PID terminal before opening QUIET WINDOW**. Five minutes is a buffer, not a guarantee. Do not SIGTERM an ordinary planned stop.
- `O/gpu_request` yields between MC batches but retains process/tensors. It is not memory retirement and can prevent current-network completion/STOP until its owner clears the request. Never clear somebody else's marker.
- For a needed Nc64M refinement, A30 yields at the current-network boundary immediately; it must not finish an entire selection block ahead of a higher-priority current candidate.
- Resume only after actual quiet closes and a fresh resource/priority/root authorization. Archive STOP with no overwrite (`mv -n`, then verify archive exists and STOP absent); use same pinned producer/plan with NEW authorization/log. Existing truth receipts retain original authorization. Nonblocking publisher lock prevents overlap.
- Exact scheduling/launch/stop/resume contract: `O/analyst/A34_A30_STOP_RESUME_SCHEDULE_v1_20261011T0219Z.json`. Timing depends on actual release and compute; no automated future STOP has been scheduled.

### Public watchers

- Waiter PID **66351**, `O/analyst/a11_waiter_v4.sh`, `a11_watch_v4.py`, `a11_rescan_v4.py`, output `a11_waiter_v4.out`. Corrected to date-u stamps; old waiter6212/sleep63768 retired. Evidence `a32_date_u_watcher_20261011T0106Z/VERIFICATION_RECEIPT_v1.json`.
- Board scans02/06/10/14/18/22 UTC; rule/package/release checks hourly. Latest03:00 UTC: no errors/no change, flopscope/client/server0.12.1, whestbench0.16.1, GitHubv0.12.1, forum topics30. Next board scan06:00UTC.
- Ancient idle `m_round` PID29740 was deliberately left alone. Do not indiscriminately kill Python processes. Legacy44506/44524 retired and their old STOP/archive preserved.

## Prepared Nc judge/release pipeline — NOT applicable to rejected Nc

**Nc failed its acceptance gate; skip this section for the current rejected point.** These ready sources and release timings are retained for continuity, not as authorization to judge or submit Nc. Future compatible candidates require their own prospective protocol/frozen manifest; EDBAND uses T88/A33 below.

1. For a future authorized accepted compatible candidate, verify its actual accepted terminal receipt, immutable exported `INPUT_MANIFEST.json` and all pinned artifacts, and confirm trainer worker/guard/driver retirement. No repeated public50 screen is required before the final judge.
2. Assign engineer/release and physicist teammates as needed, staying under six active agents. Build safe+twin from the frozen export with exact all16 DLY/HU/NP/KC values. Read new einsums and actual operands. Do not change the candidate after viewing final judging.
3. Use **`O/physicist/t85_pinned_judge_v7.py`** SHA `3d792804140f05879a72efd5d67e9e89f72cb4de52df9c23ddea171f7e9ab2d0`, exact worker4/source/profile/full-request pins, fixed Nc vs S46, sealed168–199. All32 truth/completion metadata must be ready before ANY pair STARTED/results; all are now available. No actual final manifest was created for Nc; it was rejected before final judging.
4. Score via **`O/analyst/a28_promotion_scorer_20261011T0033Z/score_v2.py`**, SHA `fa65d1e3d4ccb0873bb0de2164ca3cc5640a138e0c0563c1ff78c4595f83fa81`. Protocol `A27/promotion_two_look_v1.json`, SHA `a80bbcece92976858eaf5d49534437905b3c1263cb02c6426116182133687eca`.
5. Frozen protocol: pooled R=sum(candidate MSE)/sum(base MSE);100k paired-network bootstrap, PCG64seed2026101101, linear quantiles. Early16M upper.995<1 (alpha.005). If complete valid nonpass, abandon or ONE same-candidate/same-seed64M refinement with saved predictions and upper.98<1 (alpha.020). No partial aggregation, candidate tuning, new seed choice or prediction rerun. Total union-bound alpha.025; bootstrap coverage on32 networks is approximate. MC diagnostic q=-mean((c-b)*(A-B));MCSE=sqrt(sum(q²))/32; do not double-count it in bootstrap.
6. Native651/378 parity can be first two of the native10; reuse them. Measure actual native bill, do not assume same architecture implies identical cost. Safe native10 historically~15m51, twin~13m45; build~1–2min. Paired final32 judge~28min at3threads can overlap native10 only after a fresh whole-pool check.
7. Release uses `O/release/release_gate_v2.py`, `O/release/S47_RELEASE_v2.md`; required A/C/slow, exact judged hashes/mode and FIRELINE. C+slow historically safe29m09+twin21m42; reserve55min quiet. Stop A30 cleanly first, verify all heavy jobs retired, then date-u TEAM_LOG `[lead] ... QUIET WINDOW OPEN/CLOSED`. Never call GPU pause memory retirement.
8. Root alone audits daily attempts and uses `O/fire/fire_once.py` for the verified safe+twin pair. Recheck archive/source hashes and gates. Expect roughly85–95min from accepted freeze to submission if all gates pass; **03:30UTC/8:30p.m.Pacific was acceptance, not submission**. Approximate submission~05:00UTC/10p.m.Pacific.

Judge/scorer integration receipt: `O/physicist/T85_V7_SCORER_V2_INTEGRATION_v1.json` (source + synthetic actual readers PASS; no final data read).

### Optional Nc64M source is ready, NOT launched

- `O/analyst/a31_nc_full_refinement_20261011T0059Z`: `produce_v2.py` SHA `17d9d9289c776dc4def9e785820c1afe4ed9406ffe6f2c68fa8929cb2aee7b4b`, `spec_v2.json` SHA `1b93beaf1e1a3896da306ab14c186dd84409ac5811a8fe161c19a6988c836d3f`.
- `SOURCE_RECEIPT_v2`, physicist `T89_NC64M_SOURCE_REVIEW_v2`: source/11stdlib fixtures/default no-data verification PASS. Actual activation requires complete valid nonpassing EARLY_COMPLETE, frozen candidate, root resources/quiet, exclusive GPU, other producer retired, FULL_LOOK_CLAIM single lineage. Exact168–199,978 balanced half-batches,N64,094,208, saved64predictions/32COMPLETE preflight. Separate no-overwrite paths under A31/run; no candidate, final manifest or actual refinement exists yet.

## ACTUAL NEXT BRANCH: qualify and run EDBAND from standing S46

Do not recycle final168–199 casually: if Nc never judged, they remain untouched/reserved; explicit reviewed block assignment is still required. Use the latest accepted/graded best, not the rejected point, for future qualification/training. EDBAND source is extensively prepared, but **genuine runtime zero/nonzero/native qualification has NOT been performed**. Never substitute synthetic fixtures, historical baselines or booleans as that evidence.

EDBAND idea: split EDGE bias/output bands8–9,10–12,13–14; late band fixed reference, shared bias/output plus early/middle offsets form six16-vector arms/64 extra offset DOF. Native no extra per-layer operations; six setup arrays384bytes. No full-network gain has yet been measured.

Ready source/evidence index:

- `O/engineer/a56_edge_band/{fragment_edband_v1.py,edband_build_v3.py,IMPLEMENTATION_v2.md}`; `O/engineer/stack/build_edband_v3.py`.
- `T/otr_edband_v1.py`, `worker_edband_v2.py` SHA `f1dae5b23057a08871e9691fbb42e0cbe7b0565e1d970c6eea273ed66618366f`, `edband_requests_v1.py`; exporter `export_edband_s46_v2.py` SHA `6cb291bc3e690178386e68f9b3cd4fa22937e1b77a226fd965e8be0337ed8978`.
-28-array envelope, preserve full base/reset/provenance, no request-only PL/_EDBAND override. Production exporter requires a genuine worker result.
- Raw-parent adapter `O/engineer/a56_edge_band/edband_s46_adapter_v3.py`, SHA `eb9b4519e67b7e9c5b80d2adfc4aaf884d22c324707e8b39deb382437a20ac54`; `RAW_PARENT_ADAPTER_RECEIPT_v1`. Exact raw parent→single S46/HU/full16DLY/NP/KC/EDBAND stack;20stdlibfixtures+safe/twin conventions pass. Do not invent transformed-parent exceptions. NativeBSK.34 vs trainer.3402257055 is a known certified bridge, not exact identity.
- `T/edband_six_arms_v3.py` SHA `272a1a070fedec8930494196d49336e957c73106075730979162304a312961f0`; `edband_six_fit_v3.py` SHA `86d632fad3b2d4b238c19847156fa478f610bad20d1e1317ce611dcfb32c17ec`. Ridge.02diagGram finite prediction deltas,16sizing only, LOO diagnostic, unconstrained weights, independent finite check mandatory. Synthetic numeric PASS8.9s/242MB, retired: `T/cert/edband_six_synthetic_v1/RECEIPT.json`.
- First24 public ranks are fixed from the original96 order, not gradient-selected:109,6,140,631,463,625,301,554,258,178,536,45,525,176,61,280,158,560,184,649,622,602,458,87.
- One-point driver **`T/edband_six_driver_v6.py`** SHA `8e3dc70a9ff66de70f59a3687fbd2da65b908ea20bf077f67d3967357c93caf9`; verifier **`T/edband_qualification_v6.py`** SHA `91b06e8cdc7701bfb0b587eb305f60fd16c2370b74144e30d2ae73e262829ba0`.
- Driver source freeze `T/cert/edband_driver_source_freeze_v1/RECEIPT.json`; independent source review `O/physicist/T90_EDBAND_DRIVER_SCORER_REVIEW_v1.json` SHA `ef8368705a35f70d498f05606b5d9bfcabcf0f296eff3b5df599b125a240c076`.12stdlib checks+final dependency/compile/JSON-key regression pass. These prove source preparation, not runtime qualification.
- Flow: fixed public24→≤6arms→A30size232–247→ONE fit→immutable POINT_FROZEN_BEFORE_CHECK→base/point check248–263→same pooled-change<-SE decision→STOPfinally. No export/judge/fire/automatic retry/resume. One guarded worker with exact qualified3/4/8thread configuration; helpers only when neural queue empty/all issued results complete, then fully retired.
- Qualification binds actual latest-best checkpoint/full request, actual plain/enabled ALL16 zero results, source/runtime/request identity, real export/adapter/native arrays, and fixed-nonzero probe on651/378. Exact JSON double bytes, no float32 narrowing. Plainbase↔envelope fingerprint validated by an artifact-only Torch child. JSON int-key comparison fixed using canonical JSON values; raw native AST literals stay exact.
- Native contract **`O/engineer/a56_edge_band/NATIVE_PARITY_CONTRACT_v1.json`**, SHA `e56ecad2820ced16a2db419ff175fd81270130ba251a8a0995d0ab08931debbe`: reuse quick.py JSON+little-endianf32NPY and parity_report_s46_v1.py, full source/export/adapter/guard/request/result pins, recompute all16RMS. FinalRMS≤7e-6 on BOTH651/378; no threshold slack. Genuine production COMPLETE→adapter→native counted replay, guardexit0/group retired. Zero identity and nonzero routing are separate required evidence.
- Future final judge **`O/physicist/t88_edband_judge_v3.py`** SHA `0b45e2de422d4467fd2b0c97f2744137ef9edcd94b83140738277effd129b9af`;29stdlibfixtures/source reviews PASS. Exact envelope/export/worker/profile/HU/full request, all32truths first, explicit ROOT_REVIEWED block/plan/protocol/scorer binding; reserved overlap requires reviewed release. No actual manifest or launch.
- Future scorer **`O/analyst/a33_edband_promotion_20261011T0128Z/score_v1.py`** SHA `edf939d61516746fb735da0fab0caf5bb3812b9bd50b658e405e8f6076dc3795`; protocol `promotion_two_look_v1.json` SHA `ba609596bfc4390d688376d0e34005dbb062dc241940ae3166a9ce04377548ff`. PinsT88v3/A30v2; six math/readiness helpers AST-identical to Nc. Synthetic early/full readers/provenance PASS. **200–231 remains reserved/unassigned**, actual reviewed block release and fixed accepted candidate required. Optional EDBAND64M producer is not prepared.

### Parked helper/resource decisions, do not reopen by default

- Optional second Nc eval-only worker is PARKED. Existing primary guard's actual RSS backstop10GB, footprint7.5GB, swap-growth256MiB;9GB was an expected peak, not a cap. No safe live cap reload; do not restart/replace a healthy worker to manufacture room.
- With GPU, whole-pool extra-worker requirement23GiB (10primary+6helper+3GPU+4OS). A02:10 sample19.812GiB failed. Without GPU gate20GiB; after A27 retired root deliberately selected A30 next-data throughput even though20.0636GiB then passed20. Default rationale: at most~23min earlier first verdict versus~75min delay of the next selection across quiet. No sampler campaign or speculative second worker.
- Prepared strict eval-only wrapper `T/worker_eval_only_v1.py`, SHA `6399f883b5747ddead3c8309f3d50d569ae94b2d346cd9a91e110a4760ceac97`, receipt `T/cert/cut_refit_Nc_launch_v1/eval_only_wrapper_source_receipt_v2.json`. v1's19GB proposal required replacing primary and is superseded.
- Earlier genuine EDBAND used651 zero pair is also PARKED.18natural samples01:05:53–01:08:43 max20.882GiB<22GiB required; sampler exited0, no model/helper/guard launched. `T/cert/edband_used651_parked_v1.json`. Planned3threads/180s/5GiB exact two evals, historical request13 vs zero envelope. Future qualification must be against actual latest best; do not turn a stale historical pair into a certificate.

## Cut-simulator work completed by Astra — do not repeat

- `O/engineer/a22_cutsim/S46_v11_PINNED_PROFILE.json`, SHA `06fe3a1b601a7893f61725a8ddee17a590039ab21aaaf7477385b466890a2f7b`: enabled/law_response_fit/law_beta/law_restore_right ALLtrue.
- Certificate `T/cert/cutsim_s46_20261010T232518Z/CERTIFICATE_V11_SCOPED_v1.json`, frozen_v11_sources. Oldv2 pointed at staleS37 and is preserved, superseded. Native base SHA `63f89e8da3e599d550a332972229a27c8600d1e8b97bfbb61ba02f8995e11428`.
- OFF651/378 exact old Torch ALL16; ON nativeRMS3.73856/3.87289e-6, ~95% residual-power removal. This is simulator fidelity, not leaderboard gain.111 ordered forward/replay rosters exact, finite gradients.
- Whole-card FD has a known surrogate mismatch and is not a strict PASS. Do not reopen a broad FD campaign. T86 localVJP L7/L13: five endpoints exact,32paths finite, float64FD4.83e-8, mixed.254%.
- T85 used104 bridge: all16prediction/MSE exact,3threads52.2s,RSS4.112GB, retired.
-8thread benchmark65143.48s vs4thread52.21s, peak8.149GB;111rosters/replay exact, small gradient rounding≤.031%EDGE/.20%dial; eval-vs-grad8 all16 exact. Supports Nc8threads. Worker lifetime gradient peak8.84GB is not the present eval's RSS.

## Further source-ready research and closed results

T48 repair-head standby (no runtime experiment): separate old/young HU31/32, L7–14; HC5/HCF15 unchanged, separate2column projection. Full1024-feature moments, vcfloor1e-10, roster-only center/clip±8, outside0; capture old/young separately before cleanup, never combined-minus-old. Native zero/disabled omits wrapper and yields exact parent bytes; forward cache VJP. See `O/physicist/T87_PREPARATION_COMPLETE_v1.json` and `O/engineer/a59_repair_heads/{SOURCE_AND_VECTOR_GATES_v1.json,SOURCE_BUILD_RECEIPT_v2.json,NATIVE_INTEGRATION_CONTRACT_v1.json}`. Native fragmentv2 SHA prefix06422496, builderv2 prefix7c03be83. Root read `kn,kc->nc` with actual2×1024 and2×128 ordinary dense operands. Projection526336FLOPs/3requests; capture30654/43 including stub is not full bill. Torch243MB/native50MB fixtures retired. Still needs genuine export/nativezero+nonzero651/378/fullgradient-cache/bill/fit/freshjudge/release. Queue after EDBAND.

Closed or parked absent NEW evidence:

- jmL no accepted step: check+.052%SE.162; edge arm+.45%SE.22.
- Layer15 unmask/ceiling closed; U15aOFF.
- Gamma weak: .15 gave−.21%SE.33 on old11, .20 unmeasured, .125 failed; parked.
- ru192H2 TRAIN8 raw+1.2877%, bill−.83335%, adjusted+.4476%,3/8wins, CIcross0. Needs.442%MSE recovery just to break even; no proven5–10% bill path.
- Old rank288/320 refits and tail2/1.5 negative; young-carrier compression negative.
- Head128→256 only−.075%VALID16 against+.88%bill; auxiliary loss closed.
- EDGE16→32 source feasible+2064params,+.4558%bill, unequal.4/.6 split, unimplemented; after cheaper features.
- New Marius board change received a bounded source/metadata review: `O/engineer/BOARD_COST_REVIEW_20261011T0200Z_v2.json` (v1 preserved). FinalMSE−.02588%, billshare−3.08425%,100telemetry meanbill−3.10168%, wallmean+14.507%. Scanner dropped mlp_index, so position differences are conditional, not authenticated network pairing. Our historical S45 census91.64%products; no new algorithm is identifiable from totals.
- Seven-level strip wall objection is now stale: S46 safe worst90.1s gives9.9s under100s vs historical+4.3s estimate. **Request cap objection remains:** safe43,398first/42,543warm leaves402/157 under43,800/42,700; estimated+1000 exceeds598/843. Primary uses risk gate, but worstresidual.3588s leaves1.2ms under local.36s; hard platform.4s is distinct. Closed due to actual constraints, not because gain<1%.

## Cold-start operational checklist for Opus

1. Read FINAL actual Nc decision/export/terminal snapshot before doing anything dependent on it. Revalidate PIDs with `ps`; the ones above describe the handoff, not perpetual liveness.
2. Verify A30 still live or actually terminal and inspect only appropriate metadata. If it completed its64entry plan, leave it terminal; no extension/restart is authorized. Keep its final200–231 unassigned.
3. Check the newest public watcher status. Keep maximum six active agent teammates. Root alone submits.
4. Start with the genuine EDBAND qualification branch above; current Nc was rejected. Preserve the prepared Nc judge/release pipeline for appropriately authorized future compatible candidates. Resume the same scientific objective. Do not quietly lower acceptance, confidence, parity, bill, memory or release gates.
5. Before heavy concurrency, measure current memory and account for every process plus GPU/OS reserve. Before C/slow, clean-stop producer(s) and verify exit. Version outputs and record dated conclusions.
6. Continue stacking every verified gain and firing safe+twin pairs within the10/day cap. There is no verified new gain until real final/native/release evidence says so.

All normal work and conclusions are recorded in `O/TEAM_LOG.md` and the named versioned receipts. The source-ready pipelines above were repeatedly checked; use those artifacts rather than reconstructing them from prose.
