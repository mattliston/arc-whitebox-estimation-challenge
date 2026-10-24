# Submission #331992: final third-moment row selection, five-level products, and memory reductions

2026-09-23T10:13:59Z · submission · adopted · measured

Added final-layer third-moment row selection, five-level products where six levels are unavailable, float32 plane sums, frame eviction, and exact memory reductions. The official public score is 3.1762783588318823e-09 with 100 networks and 0 failures.

Recorded measurement:

    {"submission_id": 331992, "date": "2026-09-23T10:13:59Z", "status": "GRADED", "score_public": 3.1762783588318823e-09, "n": 100, "failed_networks": 0, "utilization": 0.17562302814264513, "wall_mean_s": 92.367, "wall_max_s": 107.976, "resid_mean_s": 0.273, "resid_max_s": 0.343, "source_sha256": "c34bdd791c21bbda3c9a5264a8be6166e1668c1688434c9f68ca75af23f3baf9"}
    - 2026-09-23 03:20 PDT FIRE #331992 (Fable): fable_fire_v1/fire_candS5e/candS5e_mem.py (sha256 c34bdd791c21bbda3c9a5264a8be6166e1668c1688434c9f68ca75af23f3baf9) = Fable S5e: S4 (#331961) + physicist's L15 k3 row cut (|z|<3.5) + engineer's push5 depth rule (five Winograd levels where six are impossible) + f32 third-order plane sums + --evict5 frame eviction + engineer2's bit-exact memory diet. VALID40 (cold) adj 0.90083 vs 331824, 0.9776 vs S4 (40/40); warm nets add ~1 pt of bill cut. Panel 0.8998 / 0.9075. Handles over 20 nets 74.1k (S4 77.1k); live peak 2.40 GB; server CPU ~= S4. Expected ~3.16-3.19e-9. Poll: .venv/bin/python autoresearch/poll_sub.py 331992
    - 2026-09-23 03:59 PDT GRADED #331992 (S5e): adjusted 3.1762783588318823e-9, raw 1.809785754147697e-8, util 0.17562. NEW PHASE-2 BANK, -2.64% vs #331961 (3.2623e-9); adjusted better on 50/50 public nets. Platform: wall mean 92.4 s, worst 108.0 s (MLP 4, a cold first-of-worker net; S4 worst 101.1 s), kernel 83.2 s, overhead 8.9 s, residual max 0.343 s (p95 0.306; limit 0.4). 0 failed MLPs.
    - 2026-09-23 04:47 PDT FIRE #332005 (Fable): fable_fire_v1/fire_candS6b/candS6b_mem.py (sha256 a514c21b23ea209a7a3fd25b778a0e4fa2ed9850fc1a97a8c6bfce2bdc2cafc6) = Fable S6b: S5e (#331992) + full closure refit (analyst refit_s2_full_lr2e4_v2 params_epoch2: 3 epochs on 160 TRAIN nets, learned literals only) + engineer's --batch (bit-exact batched FLMS products) + engineer2 body stages yright/arena (bit-exact). VALID40 (cold) adj 0.89591 vs 331824; vs S5e raw -0.50% (33/40); panel8/virgin6 0.9963/0.9877 vs S5e. Gate vs S5e: server CPU -2.1%, residual -12.6 ms, handles 74.7k (S4 77.1k). Expected ~3.155e-9. Poll: .venv/bin/python autoresearch/poll_sub.py 332005
    - 2026-09-23 05:23 PDT GRADED #332005 (S6b): adjusted 3.175776422887701e-9, raw 1.799167293370374e-8, util 0.17651. Ties #331992 (3.1763e-9; -0.02%). Raw -0.59% vs S5e (40/50 public nets better): the closure refit transfers. But util rose +0.51%, because the platform set up every worker TWICE (10 setup executions vs 5 for S5e), doubling the cold first-MLPs. Each pays ~+4-4.5% flops for frame binding (MLPs 17-21 at +3.4..+4.5%); the median per-MLP flops change is +0.01%. Wall mean 92.0 s, worst 103.0 s; residual max 0.335 s; 0 failures. Current best on the board: #332005 by a hair.

Public score: 3.1762783588318823e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [SUBMISSION_LEDGER.md#L713](../../records/SUBMISSION_LEDGER.md#L713)
- [SUBMISSION_LEDGER.md#L714](../../records/SUBMISSION_LEDGER.md#L714)
- [SUBMISSION_LEDGER.md#L715](../../records/SUBMISSION_LEDGER.md#L715)
- [SUBMISSION_LEDGER.md#L716](../../records/SUBMISSION_LEDGER.md#L716)
- [TEAM_LOG_2026-09-22_to_09-26.md#L170](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L170)
- [TEAM_LOG_2026-09-22_to_09-26.md#L183](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L183)

<details>
<summary>All result files and records (5)</summary>

- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [TEAM_LOG_2026-09-22_to_09-26.md](../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [submit_output.txt](submit_output.txt) · record
- [poll.log](poll.log) · log
- [eval_331992.json](eval_331992.json) · result

</details>

## Related work

- ← shipped in: [Five-level layouts for previously shallower pushes reduce warm work about 1.9%](../../experiments/arithmetic-kernels/five-level-layouts-previously-shallower-pushes-reduce-warm-work/README.md)
- ← shipped in: [Frame-handle census finds single-use views dominate cache growth](../../instruments/frame-handle-census-finds-single-use-views-dominate-cache/README.md)
- ← shipped in: [Layer-15 third-cumulant response cut \|z\| < 3.5: native adjusted error improves 0.66% on 40 networks](../../experiments/late-layer-exactness/layer-15-third-cumulant-response-cut-z-3-5/README.md)
- ← shipped in: [Live-memory peak locator and temporary lifetime reductions save 351 MB without changing predictions](../../instruments/live-memory-peak-locator-temporary-lifetime-reductions-save-351/README.md)
- ← shipped in: [Third-order plane sums converted to float32: 0.52% less work with rounding-level error](../../experiments/arithmetic-kernels/third-order-plane-sums-converted-float32-0-52-less/README.md)
- supersedes: [Submission #331961: held views, copy elimination, large arenas, and fitted covariance scales](../331961/README.md)
- ← supersedes: [Submission #332005: joint closure refit, batched products, and memory-layout changes](../332005/README.md)

[Research graph](../../graph/index.html#node=submission-331992) · [Copy provenance](../../PROVENANCE.json)

