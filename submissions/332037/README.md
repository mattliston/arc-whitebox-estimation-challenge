# Submission #332037: joint closure refit including graph-neural-network trunk weights

2026-09-23T16:26:18Z · submission · adopted · measured

Replaced learned literals with an epoch-0 joint refit including graph-neural-network trunk weights, leaving the arithmetic structure unchanged. The official public score is 3.083369326590302e-09 with 100 networks and 0 failures.

Recorded measurement:

    {"submission_id": 332037, "date": "2026-09-23T16:26:18Z", "status": "GRADED", "score_public": 3.083369326590302e-09, "n": 100, "failed_networks": 0, "utilization": 0.17387094990735932, "wall_mean_s": 93.863, "wall_max_s": 110.425, "resid_mean_s": 0.252, "resid_max_s": 0.3, "source_sha256": "1534b1ddfac6f8fc69ed928e56e48e1f62790fdea88d4ebf80fd48d59c8f1ad1"}
    - 2026-09-23 09:26 PDT FIRE #332037 (Fable): fable_fire_v1/fire_candS9/candS9_S7_B2refit_mem.py (sha256 1534b1ddfac6f8fc69ed928e56e48e1f62790fdea88d4ebf80fd48d59c8f1ad1) = Fable S9: S7 #332026 + physicist's B2 closure refit, epoch 0 (full groups + GNN trunk msg/enc0/2/4 weights and biases; params fable_physicist_v1/refit6b/s6b_B2_fullnet_lr1e4_trunk5e5/params_epoch0.pt, sha 46f27bdd). Code token-identical to S7 (5 literal statements; einsum gate PASS). Gates: torch VALID40 -1.09% (33/40); native VALID40 vs S7 rows raw -1.00% (CI -1.35..-0.65), 30/40, util identical on all 40 nets; native vs torch per-net corr .985. Expected ~3.095e-9. Its platform kernel ns/kflop also tests whether S7's +8% came from the host (~215) or from push5p6 (~232). 6th fire of UTC 09-23.
    - 2026-09-23 10:05 PDT GRADED #332037 (S9): adjusted 3.083369326590302e-9, raw 1.774489632566656e-8, util 0.17387. NEW PHASE-2 BANK, -1.37% vs #332026 (3.1262e-9). The raw gain (-1.37%) is the B2 trunk refit, larger than VALID40's -1.00%; util unchanged.
    - 2026-09-24 10:24 PDT FIRE #332182 (Fable): fable_fire_v1/fire_candS11/candS11_S10L_B2e1_mem.py (sha256 acf67b9d59c8b68f6eb7ab0f1b9ce8209f11f5b40b0ac792d223ecce5e89a24a) = Fable S11: S9 #332037 + engineer's --symout v3 (covariance second product as upper-block plain products, G21 mirrored; no symmetric einsum) + --lean v3 (fnp.split quadrants + function-scoped holds; bit-exact) + B2 closure refit EPOCH 1 (params_epoch1.pt). push5p6 kept.
    - 2026-09-24 10:56 PDT GRADED #332182 (S11): adjusted 3.046329368760037e-9, raw 1.7707661381649587e-8, util 0.17215. NEW PHASE-2 BANK, -1.20% vs #332037 (3.0834e-9). Expected ~3.044e-9 (VALID40 -1.29%), so the local prediction held to 0.09 pt.

Public score: 3.083369326590302e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [SUBMISSION_LEDGER.md#L719](../../records/SUBMISSION_LEDGER.md#L719)
- [SUBMISSION_LEDGER.md#L720](../../records/SUBMISSION_LEDGER.md#L720)
- [SUBMISSION_LEDGER.md#L724](../../records/SUBMISSION_LEDGER.md#L724)
- [SUBMISSION_LEDGER.md#L728](../../records/SUBMISSION_LEDGER.md#L728)
- [TEAM_LOG_2026-09-22_to_09-26.md#L287](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L287)
- [TEAM_LOG_2026-09-22_to_09-26.md#L302](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L302)

<details>
<summary>All result files and records (6)</summary>

- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [TEAM_LOG_2026-09-22_to_09-26.md](../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [submit_output.txt](submit_output.txt) · record
- [poll.log](poll.log) · log
- [eval_332037.json](eval_332037.json) · result
- [eval_332037-b4f11a62.json](eval_332037-b4f11a62.json) · result

</details>

## Related work

- ← shipped in: [Training message and encoder weights improves native raw error 1.001% with identical work](../../experiments/learned-correction/training-message-encoder-weights-improves-native-raw-error-1/README.md)
- supersedes: [Submission #332026: cheaper first-network frame binding and padded six-level products](../332026/README.md)
- ← supersedes: [Submission #332182: upper-block covariance products, lean views, and an epoch-1 refit](../332182/README.md)

[Research graph](../../graph/index.html#node=submission-332037) · [Copy provenance](../../PROVENANCE.json)

