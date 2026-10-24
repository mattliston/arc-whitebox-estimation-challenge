# Submission #331947: rowwise (3,1) correction, gain correction, and joins every third layer

2026-09-23T05:46:45Z · submission · adopted · measured

Added rowwise (3,1) cumulant scaling, a gain-mixture correction, joins at layers 7, 10, and 13, and an exact reduction in body requests. The official public score is 3.3389637144886302e-09 with 100 networks and 0 failures.

Recorded measurement:

    {"submission_id": 331947, "date": "2026-09-23T05:46:45Z", "status": "GRADED", "score_public": 3.3389637144886302e-09, "n": 100, "failed_networks": 0, "utilization": 0.1820967188259965, "wall_mean_s": 95.67, "wall_max_s": 105.643, "resid_mean_s": 0.258, "resid_max_s": 0.299, "source_sha256": "28583ca7bce60611babf51ef4c76ca342cc71e26de85352802ec575c766e8279"}
    - 2026-09-23T05:46:36Z FIRE #331947 (Fable): candS2_e2v7_k31r_bfix01_every3.py = Fable S2: bank 331824 + bit-exact body request diet + per-row K31 factor + gain b-fix g0.1 + old-tier join schedule L7/10/13. VALID40 adj -5.4% (40/40); panel8 0.948, virgin6 0.955 vs bank (sha256 28583ca7bce60611). Poll: .venv/bin/python autoresearch/poll_sub.py 331947
    - 2026-09-22 23:22 PDT GRADED #331947: adjusted 3.3389637144886302e-9, raw 1.8344364551836634e-8 (implied util 0.18202). NEW PHASE-2 BANK, -4.8% vs #331824 3.507e-9 (local VALID40 predicted -5.4%, panel8 -5.2%). File fable_fire_v1/candS2_e2v7_k31r_bfix01_every3.py (sha256 28583ca7bce60611...). Not under the 3e-9 milestone; next stack = S3 base (fable_engineer_v1/candS3_S2_HE.py, bit-identical, util -0.9%) + GN knobs + sq6 kernel.
    - 2026-09-23 01:35 PDT GRADED #331961: adjusted 3.262269644043314e-9, raw 1.8113294792954095e-8, util 0.18018. NEW PHASE-2 BANK, -2.30% vs #331947 (3.339e-9); local VALID40 predicted -2.23%. Platform wall mean 92.6 s, worst 101.1 s (S2: 95.7 / 105.6), kernel 83.7 s, overhead 8.6 s; 0 failed MLPs.

Public score: 3.3389637144886302e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [SUBMISSION_LEDGER.md#L709](../../records/SUBMISSION_LEDGER.md#L709)
- [SUBMISSION_LEDGER.md#L710](../../records/SUBMISSION_LEDGER.md#L710)
- [SUBMISSION_LEDGER.md#L712](../../records/SUBMISSION_LEDGER.md#L712)
- [TEAM_LOG_2026-09-22_to_09-26.md#L74](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L74)
- [TEAM_LOG_2026-09-22_to_09-26.md#L90](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L90)

<details>
<summary>All result files and records (7)</summary>

- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [TEAM_LOG_2026-09-22_to_09-26.md](../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [submit_output.txt](submit_output.txt) · record
- [FIRE_PLAN.md](FIRE_PLAN.md) · record
- [poll.log](poll.log) · log
- [eval_331947.json](eval_331947.json) · result
- [eval_331947-b9853319.json](eval_331947-b9853319.json) · result

</details>

## Related work

- ← shipped in: [Estimator-body request reductions: bit-exact predictions and 0.58% less metered work](../../experiments/arithmetic-kernels/estimator-body-request-reductions-bit-exact-predictions-0-58/README.md)
- ← shipped in: [Gain-mixture vector correction at strength 0.1 improves raw error by 1.10% on four networks](../../experiments/third-order-planes/gain-mixture-vector-correction-strength-0-1-improves-raw/README.md)
- ← shipped in: [Old-cohort joins every third layer improve raw error 2.2% on twelve training networks at neutral work](../../experiments/third-order-planes/old-cohort-joins-every-third-layer-improve-raw-error/README.md)
- ← shipped in: [Old-plane joins every third layer outperform more accurate frame construction](../../experiments/third-order-planes/old-plane-joins-every-third-layer-outperform-more-accurate/README.md)
- ← shipped in: [Per-row (3,1)-plane scale 0.85 + 0.5 μ κ₃/v²: native adjusted ratio 0.98606](../../experiments/fourth-order-terms/per-row-3-1-plane-scale-0-85-0/README.md)
- supersedes: [Submission #331824: 1,536-value message-to-covariance context fit improves adjusted error 0.018%](../331824/README.md)
- ← supersedes: [Submission #331961: held views, copy elimination, large arenas, and fitted covariance scales](../331961/README.md)

[Research graph](../../graph/index.html#node=submission-331947) · [Copy provenance](../../PROVENANCE.json)

