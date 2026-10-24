# Submission #332026: cheaper first-network frame binding and padded six-level products

2026-09-23T14:50:27Z · submission · adopted · measured

Reduced first-network frame binding and added padded six-level products, capped batching, and retained frames. The official public score is 3.1262222290703757e-09 with 100 networks and 0 failures.

Recorded measurement:

    {"submission_id": 332026, "date": "2026-09-23T14:50:27Z", "status": "GRADED", "score_public": 3.1262222290703757e-09, "n": 100, "failed_networks": 0, "utilization": 0.17386048069379284, "wall_mean_s": 95.66, "wall_max_s": 102.998, "resid_mean_s": 0.256, "resid_max_s": 0.316, "source_sha256": "9011916a405b8bb5d17e63bd7c4c5200d31121ba92b0089b9334d0731e6dc158"}
    - 2026-09-23 07:50 PDT FIRE #332026 (Fable): fable_fire_v1/fire_candS7/candS7_full_cb3_mem.py (sha256 9011916a405b8bb5d17e63bd7c4c5200d31121ba92b0089b9334d0731e6dc158) = Fable S7: S6b (#332005) + engineer's --coldbind3 (cheaper cold-start frame binding: region markers, read5 frames once per shape, used-slot last level, scratchless unwind; bit-exact) + ORIGINAL push5p6 depth rule (padded six-level products) + --caps --caps2 --bindhold + hold2 (bit-exact). Built by the porter (build_stack.sh), bit-equal to candS7_mem.py at all 16 layers (OpenBLAS env, 4 nets). Gates: VALID40 vs S6b rows (.venv, all cold) raw -0.016% (CI -0.068..+0.038%), util -2.15%, adjusted -2.16% (40/40); cb3 alone -1.84% cold. engineer2 paired 20-net server gate vs S6b: CPU +2.72%, residual -0.5 ms (cold ~0.32 s projected), handles 70.7k, peak +31 MB, util -0.78%; wall +1.5 s mean (worst expected ~96 s; analyst timeout model risk ~0.1%). Expected ~3.15e-9 (-0.8%). Poll: fable_fire_v1/fire_candS7/poll.log.
    - 2026-09-23 08:43 PDT GRADED #332026 (S7): adjusted 3.1262222290703757e-9, raw 1.7992122227639128e-8, util 0.17386. NEW PHASE-2 BANK, -1.56% vs #332005 (3.1758e-9). Raw unchanged (+0.002%); the whole gain is bill (-1.50%). Platform: 0 failed, wall mean 95.7 s / max 103.0 s, residual max 0.32 s. The server kernel per flop was +8% vs every earlier run (push5p6 or host; S9 will discriminate). The 3e-9 bank needs -4.0% more.
    - 2026-09-23 09:26 PDT FIRE #332037 (Fable): fable_fire_v1/fire_candS9/candS9_S7_B2refit_mem.py (sha256 1534b1ddfac6f8fc69ed928e56e48e1f62790fdea88d4ebf80fd48d59c8f1ad1) = Fable S9: S7 #332026 + physicist's B2 closure refit, epoch 0 (full groups + GNN trunk msg/enc0/2/4 weights and biases; params fable_physicist_v1/refit6b/s6b_B2_fullnet_lr1e4_trunk5e5/params_epoch0.pt, sha 46f27bdd). Code token-identical to S7 (5 literal statements; einsum gate PASS). Gates: torch VALID40 -1.09% (33/40); native VALID40 vs S7 rows raw -1.00% (CI -1.35..-0.65), 30/40, util identical on all 40 nets; native vs torch per-net corr .985. Expected ~3.095e-9. Its platform kernel ns/kflop also tests whether S7's +8% came from the host (~215) or from push5p6 (~232). 6th fire of UTC 09-23.
    - 2026-09-23 10:05 PDT GRADED #332037 (S9): adjusted 3.083369326590302e-9, raw 1.774489632566656e-8, util 0.17387. NEW PHASE-2 BANK, -1.37% vs #332026 (3.1262e-9). The raw gain (-1.37%) is the B2 trunk refit, larger than VALID40's -1.00%; util unchanged.

Public score: 3.1262222290703757e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [SUBMISSION_LEDGER.md#L717](../../records/SUBMISSION_LEDGER.md#L717)
- [SUBMISSION_LEDGER.md#L718](../../records/SUBMISSION_LEDGER.md#L718)
- [SUBMISSION_LEDGER.md#L719](../../records/SUBMISSION_LEDGER.md#L719)
- [SUBMISSION_LEDGER.md#L720](../../records/SUBMISSION_LEDGER.md#L720)
- [TEAM_LOG_2026-09-22_to_09-26.md#L258](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L258)
- [TEAM_LOG_2026-09-22_to_09-26.md#L277](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L277)

<details>
<summary>All result files and records (6)</summary>

- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [TEAM_LOG_2026-09-22_to_09-26.md](../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [submit_output.txt](submit_output.txt) · record
- [poll.log](poll.log) · log
- [eval_332026.json](eval_332026.json) · result
- [eval_332026-ddd93602.json](eval_332026-ddd93602.json) · result

</details>

## Related work

- ← shipped in: [Cache residual creep traced to eviction finalizers; larger entry caps avoid steady-state evictions](../../experiments/wall-and-residual/cache-residual-creep-traced-eviction-finalizers-larger-entry-caps/README.md)
- ← shipped in: [Eliminating unused frame slots and unwind scratch cuts cold work 1.836% with identical predictions](../../experiments/arithmetic-kernels/eliminating-unused-frame-slots-unwind-scratch-cuts-cold-work/README.md)
- ← shipped in: [Retained frame-binding and body aliases reduce hidden deallocation trips without changing work](../../experiments/arithmetic-kernels/retained-frame-binding-body-aliases-reduce-hidden-deallocation-trips/README.md)
- supersedes: [Submission #332005: joint closure refit, batched products, and memory-layout changes](../332005/README.md)
- ← supersedes: [Submission #332037: joint closure refit including graph-neural-network trunk weights](../332037/README.md)

[Research graph](../../graph/index.html#node=submission-332026) · [Copy provenance](../../PROVENANCE.json)

