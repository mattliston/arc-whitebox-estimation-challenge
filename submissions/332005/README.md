# Submission #332005: joint closure refit, batched products, and memory-layout changes

2026-09-23T11:43:24Z · submission · adopted · measured

Added a joint closure refit, batched fast matrix products, and exact changes to young-cohort and arena memory handling. The official public score is 3.175776422887701e-09 with 100 networks and 0 failures.

Recorded measurement:

    {"submission_id": 332005, "date": "2026-09-23T11:43:24Z", "status": "GRADED", "score_public": 3.175776422887701e-09, "n": 100, "failed_networks": 0, "utilization": 0.17650622829811255, "wall_mean_s": 91.957, "wall_max_s": 103.02, "resid_mean_s": 0.255, "resid_max_s": 0.335, "source_sha256": "a514c21b23ea209a7a3fd25b778a0e4fa2ed9850fc1a97a8c6bfce2bdc2cafc6"}
    - 2026-09-23 04:47 PDT FIRE #332005 (Fable): fable_fire_v1/fire_candS6b/candS6b_mem.py (sha256 a514c21b23ea209a7a3fd25b778a0e4fa2ed9850fc1a97a8c6bfce2bdc2cafc6) = Fable S6b: S5e (#331992) + full closure refit (analyst refit_s2_full_lr2e4_v2 params_epoch2: 3 epochs on 160 TRAIN nets, learned literals only) + engineer's --batch (bit-exact batched FLMS products) + engineer2 body stages yright/arena (bit-exact). VALID40 (cold) adj 0.89591 vs 331824; vs S5e raw -0.50% (33/40); panel8/virgin6 0.9963/0.9877 vs S5e. Gate vs S5e: server CPU -2.1%, residual -12.6 ms, handles 74.7k (S4 77.1k). Expected ~3.155e-9. Poll: .venv/bin/python autoresearch/poll_sub.py 332005
    - 2026-09-23 05:23 PDT GRADED #332005 (S6b): adjusted 3.175776422887701e-9, raw 1.799167293370374e-8, util 0.17651. Ties #331992 (3.1763e-9; -0.02%). Raw -0.59% vs S5e (40/50 public nets better): the closure refit transfers. But util rose +0.51%, because the platform set up every worker TWICE (10 setup executions vs 5 for S5e), doubling the cold first-MLPs. Each pays ~+4-4.5% flops for frame binding (MLPs 17-21 at +3.4..+4.5%); the median per-MLP flops change is +0.01%. Wall mean 92.0 s, worst 103.0 s; residual max 0.335 s; 0 failures. Current best on the board: #332005 by a hair.
    - 2026-09-23 07:50 PDT FIRE #332026 (Fable): fable_fire_v1/fire_candS7/candS7_full_cb3_mem.py (sha256 9011916a405b8bb5d17e63bd7c4c5200d31121ba92b0089b9334d0731e6dc158) = Fable S7: S6b (#332005) + engineer's --coldbind3 (cheaper cold-start frame binding: region markers, read5 frames once per shape, used-slot last level, scratchless unwind; bit-exact) + ORIGINAL push5p6 depth rule (padded six-level products) + --caps --caps2 --bindhold + hold2 (bit-exact). Built by the porter (build_stack.sh), bit-equal to candS7_mem.py at all 16 layers (OpenBLAS env, 4 nets). Gates: VALID40 vs S6b rows (.venv, all cold) raw -0.016% (CI -0.068..+0.038%), util -2.15%, adjusted -2.16% (40/40); cb3 alone -1.84% cold. engineer2 paired 20-net server gate vs S6b: CPU +2.72%, residual -0.5 ms (cold ~0.32 s projected), handles 70.7k, peak +31 MB, util -0.78%; wall +1.5 s mean (worst expected ~96 s; analyst timeout model risk ~0.1%). Expected ~3.15e-9 (-0.8%). Poll: fable_fire_v1/fire_candS7/poll.log.
    - 2026-09-23 08:43 PDT GRADED #332026 (S7): adjusted 3.1262222290703757e-9, raw 1.7992122227639128e-8, util 0.17386. NEW PHASE-2 BANK, -1.56% vs #332005 (3.1758e-9). Raw unchanged (+0.002%); the whole gain is bill (-1.50%). Platform: 0 failed, wall mean 95.7 s / max 103.0 s, residual max 0.32 s. The server kernel per flop was +8% vs every earlier run (push5p6 or host; S9 will discriminate). The 3e-9 bank needs -4.0% more.

Public score: 3.175776422887701e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [SUBMISSION_LEDGER.md#L715](../../records/SUBMISSION_LEDGER.md#L715)
- [SUBMISSION_LEDGER.md#L716](../../records/SUBMISSION_LEDGER.md#L716)
- [SUBMISSION_LEDGER.md#L717](../../records/SUBMISSION_LEDGER.md#L717)
- [SUBMISSION_LEDGER.md#L718](../../records/SUBMISSION_LEDGER.md#L718)
- [TEAM_LOG_2026-09-22_to_09-26.md#L197](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L197)
- [TEAM_LOG_2026-09-22_to_09-26.md#L203](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L203)

<details>
<summary>All result files and records (5)</summary>

- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [TEAM_LOG_2026-09-22_to_09-26.md](../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [submit_output.txt](submit_output.txt) · record
- [poll.log](poll.log) · log
- [eval_332005.json](eval_332005.json) · result

</details>

## Related work

- ← shipped in: [Batching covariance and residual-plane products cuts server CPU 2.12% and residual time 12.6 ms](../../experiments/arithmetic-kernels/batching-covariance-residual-plane-products-cuts-server-cpu-2/README.md)
- ← shipped in: [Full closure refit after mathematical changes improves native raw error 0.50%](../../experiments/learned-correction/full-closure-refit-after-mathematical-changes-improves-native-raw/README.md)
- ← shipped in: [Pair-map temporaries reuse the multiplication arena: bit-exact and about 1.2-1.6% less server CPU](../../experiments/arithmetic-kernels/pair-map-temporaries-reuse-multiplication-arena-bit-exact-about/README.md)
- ← shipped in: [Worker restarts erased a 0.60% raw-error gain by rebinding cold frames](../../records/research/2026-09-23-worker-restarts-erased-0-60-raw-error-gain-rebinding/README.md)
- supersedes: [Submission #331992: final third-moment row selection, five-level products, and memory reductions](../331992/README.md)
- ← supersedes: [Submission #332026: cheaper first-network frame binding and padded six-level products](../332026/README.md)

[Research graph](../../graph/index.html#node=submission-332005) · [Copy provenance](../../PROVENANCE.json)

