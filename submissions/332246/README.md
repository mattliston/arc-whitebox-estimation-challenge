# Submission #332246: cross-term removal and Nyström old-cohort joins

2026-09-25T06:22:22Z · submission · adopted · measured

Removed the residual-plane cross term and replaced old-cohort joins with a Nyström sketch from 384 rows selected by Gram diagonal. The official public score is 2.395096202270728e-09 with 100 networks and 0 failures.

Recorded measurement:

    {"submission_id": 332246, "date": "2026-09-25T06:22:22Z", "status": "GRADED", "score_public": 2.395096202270728e-09, "n": 100, "failed_networks": 0, "utilization": 0.14165804373268656, "wall_mean_s": 63.464, "wall_max_s": 75.678, "resid_mean_s": 0.23, "resid_max_s": 0.26, "source_sha256": "9247a212c5a742120a1aac9e25f483e3091eb1e96d1e2cf5c7760bb64b508f6b"}
    - 2026-09-24 23:22 PDT FIRE #332246 (Fable): fable_fire_v1/fire_candS23/candS23_S21s1_loopR1_s1z_jn_mem.py (sha256 9247a212...) = Fable S23 = S21s1 + loopR1 refit literals + round-1 dials + s1 = 0 (the star's s1 term dropped; engineer2) + Nystrom old-tier join (384 top-Gram-diagonal rows at L7/10/13; porter). Every partial byte-rebuilds its tested file.
    - 2026-09-24 23:45 PDT GRADED #332246 (S23): adjusted 2.395096202270728e-9, raw 1.6912490359288768e-8, util 0.14166. NEW PHASE-2 BANK, -4.40% vs #332241 (2.5054e-9), -5.45% vs S21s1; expected ~2.384e-9 (VALID40 vs S21s1 -5.90%).
    - 2026-09-25 02:05 PDT GRADED #332255 (S24): adjusted 2.372624767712433e-9, raw 1.6833305132024634e-8, util 0.14098. NEW PHASE-2 BANK, -0.94% vs #332246 (2.3951e-9); expected ~2.375e-9 (VALID40 -0.854%).

Public score: 2.395096202270728e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [SUBMISSION_LEDGER.md#L864](../../records/SUBMISSION_LEDGER.md#L864)
- [SUBMISSION_LEDGER.md#L871](../../records/SUBMISSION_LEDGER.md#L871)
- [SUBMISSION_LEDGER.md#L881](../../records/SUBMISSION_LEDGER.md#L881)
- [TEAM_LOG_2026-09-22_to_09-26.md#L755](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L755)
- [TEAM_LOG_2026-09-22_to_09-26.md#L770](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L770)

<details>
<summary>All result files and records (6)</summary>

- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [TEAM_LOG_2026-09-22_to_09-26.md](../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [submit_output.txt](submit_output.txt) · record
- [FIRE_CHECKS.txt](FIRE_CHECKS.txt) · record
- [poll.log](poll.log) · log
- [eval_332246.json](eval_332246.json) · result

</details>

## Related work

- ← shipped in: [Nyström join from largest Gram-diagonal rows improves native adjusted error 1.816%](../../experiments/third-order-planes/nystrom-join-largest-gram-diagonal-rows-improves-native-adjusted/README.md)
- ← shipped in: [Removing the residual-plane cross term improves raw error 0.438% and adjusted error 2.911%](../../experiments/third-order-planes/removing-residual-plane-cross-term-improves-raw-error-0/README.md)
- supersedes: [Submission #332241: one-epoch learned refit at the fitted physics scales](../332241/README.md)
- ← supersedes: [Submission #332255: joint refit, arithmetic-level fusion, and a Nyström residual-plane range finder](../332255/README.md)

[Research graph](../../graph/index.html#node=submission-332246) · [Copy provenance](../../PROVENANCE.json)

