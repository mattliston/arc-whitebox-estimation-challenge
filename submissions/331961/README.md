# Submission #331961: held views, copy elimination, large arenas, and fitted covariance scales

2026-09-23T07:59:26Z · submission · adopted · measured

Added held views, last-level copy elimination, larger arithmetic arenas, and zero-cost covariance scales fitted with Gauss–Newton. The official public score is 3.262269644043314e-09 with 100 networks and 0 failures.

Recorded measurement:

    {"submission_id": 331961, "date": "2026-09-23T07:59:26Z", "status": "GRADED", "score_public": 3.262269644043314e-09, "n": 100, "failed_networks": 0, "utilization": 0.18018416176846586, "wall_mean_s": 92.537, "wall_max_s": 101.087, "resid_mean_s": 0.255, "resid_max_s": 0.302, "source_sha256": "082f2b501047d4470dc83ee6517c18b87c9fbce84801655709e84b31afba34d1"}
    - 2026-09-23 00:59 PDT FIRE #331961 (Fable): fable_fire_v1/fire_candS4/candS4_S3HEB_gn.py (sha256 082f2b501047d4470dc83ee6517c18b87c9fbce84801655709e84b31afba34d1) = Fable S4: fired S2 lineage + bit-exact HEB kernel (hold + last-level copy elimination + big arena; engineer) + zero-bill GN covariance knobs v3 (covhunter; ported by porter). VALID40 adj 0.9245 vs 331824 (40/40), 0.9777 vs S2 (37/40); panel8 3.572e-9, virgin6 2.925e-9 (0.976 / 0.974 of S2's panel); HEB gates: compute neutral vs HE, live 3.05 GB. Expected ~3.25e-9. Poll: .venv/bin/python autoresearch/poll_sub.py 331961
    - 2026-09-23 01:35 PDT GRADED #331961: adjusted 3.262269644043314e-9, raw 1.8113294792954095e-8, util 0.18018. NEW PHASE-2 BANK, -2.30% vs #331947 (3.339e-9); local VALID40 predicted -2.23%. Platform wall mean 92.6 s, worst 101.1 s (S2: 95.7 / 105.6), kernel 83.7 s, overhead 8.6 s; 0 failed MLPs.
    - 2026-09-23 03:20 PDT FIRE #331992 (Fable): fable_fire_v1/fire_candS5e/candS5e_mem.py (sha256 c34bdd791c21bbda3c9a5264a8be6166e1668c1688434c9f68ca75af23f3baf9) = Fable S5e: S4 (#331961) + physicist's L15 k3 row cut (|z|<3.5) + engineer's push5 depth rule (five Winograd levels where six are impossible) + f32 third-order plane sums + --evict5 frame eviction + engineer2's bit-exact memory diet. VALID40 (cold) adj 0.90083 vs 331824, 0.9776 vs S4 (40/40); warm nets add ~1 pt of bill cut. Panel 0.8998 / 0.9075. Handles over 20 nets 74.1k (S4 77.1k); live peak 2.40 GB; server CPU ~= S4. Expected ~3.16-3.19e-9. Poll: .venv/bin/python autoresearch/poll_sub.py 331992
    - 2026-09-23 03:59 PDT GRADED #331992 (S5e): adjusted 3.1762783588318823e-9, raw 1.809785754147697e-8, util 0.17562. NEW PHASE-2 BANK, -2.64% vs #331961 (3.2623e-9); adjusted better on 50/50 public nets. Platform: wall mean 92.4 s, worst 108.0 s (MLP 4, a cold first-of-worker net; S4 worst 101.1 s), kernel 83.2 s, overhead 8.9 s, residual max 0.343 s (p95 0.306; limit 0.4). 0 failed MLPs.

Public score: 3.262269644043314e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [SUBMISSION_LEDGER.md#L711](../../records/SUBMISSION_LEDGER.md#L711)
- [SUBMISSION_LEDGER.md#L712](../../records/SUBMISSION_LEDGER.md#L712)
- [SUBMISSION_LEDGER.md#L713](../../records/SUBMISSION_LEDGER.md#L713)
- [SUBMISSION_LEDGER.md#L714](../../records/SUBMISSION_LEDGER.md#L714)
- [TEAM_LOG_2026-09-22_to_09-26.md#L128](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L128)
- [TEAM_LOG_2026-09-22_to_09-26.md#L140](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L140)

<details>
<summary>All result files and records (6)</summary>

- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [TEAM_LOG_2026-09-22_to_09-26.md](../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [submit_output.txt](submit_output.txt) · record
- [FIRE_PLAN.md](FIRE_PLAN.md) · record
- [poll.log](poll.log) · log
- [eval_331961.json](eval_331961.json) · result

</details>

## Related work

- ← shipped in: [Covariance error value flows through the pair map; zero-cost Gauss-Newton knobs improve held-out error](../../experiments/dials/covariance-error-value-flows-through-pair-map-zero-cost/README.md)
- ← shipped in: [Held aliases and last-level copy elimination preserve predictions and reduce work 1.18-1.23%](../../experiments/arithmetic-kernels/held-aliases-last-level-copy-elimination-preserve-predictions-reduce/README.md)
- supersedes: [Submission #331947: rowwise (3,1) correction, gain correction, and joins every third layer](../331947/README.md)
- ← supersedes: [Submission #331992: final third-moment row selection, five-level products, and memory reductions](../331992/README.md)

[Research graph](../../graph/index.html#node=submission-331961) · [Copy provenance](../../PROVENANCE.json)

