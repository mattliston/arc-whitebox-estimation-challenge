# Submission #333435: fourth-cumulant column shifts and per-unit kurtosis for pairs up to age two

2026-10-02T03:17:16Z · submission · adopted · measured

S37 adds two physics terms to the S36 file without any refit: the fourth-cumulant star carried on the bank's young column pairs and fed into the next births by an in-place column shift (strength 0.6, columns up to 2 layers old, cheap anchor) and the non-isotropic per-unit fourth cumulant from the exact hub coefficients (age 2, c 1). Qualified by stages A, C and D in a quiet window with a native 10-rank session as stage B. Graded 1.951e-9, the first bank below 2e-9 and 4th place; the final-layer MSE fell 4.26% against the offline -5.2 to -6.0%.

Recorded measurement:

    adjusted 1.9506441350217897e-9 (S36 2.031e-9: -3.94%; S35: -5.62%; expected about 1.92e-9); final-layer MSE 1.639e-8 (-4.26% vs S36; offline 24 held-out -5.2%, native 10 -6.0%), all-layer 6.458e-9 (-4.7%), utilisation 0.1190149549529906 (+0.34%); 100/100, wall 78.0 / 87.6 s (projected 74.3 / 85.5), residual 0.303 / 0.329 s (projected 0.302 / 0.336); stage B native 10 fresh vs RRPI: MSE -6.00% [-7.51,-4.35] 10 of 10, bill -1.32%; stage C bill 0.98452x of S35, requests 0.9914x (43,461 / 42,268 on networks 35 / 100), F4 + K4L cost 2,000 requests; stage D handles 56,441, group RSS 6.20 GB, bytes identical to native; 4th (J2W 1.6233e-9, marius_binner 1.6517e-9, Luna 1.8437e-9)

Public score: 1.9506441350217897e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)
- [lead_s37_chain.sh](lead_s37_chain.sh)

## Evidence

- [SUBMISSION_LEDGER.md#L904](../../records/SUBMISSION_LEDGER.md#L904)
- [SUBMISSION_LEDGER.md#L907](../../records/SUBMISSION_LEDGER.md#L907)
- [TEAM_LOG_2026-10-01_to_10-22.md#L78](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L78)
- [TEAM_LOG_2026-10-01_to_10-22.md#L85](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L85)
- [TEAM_LOG_2026-10-01_to_10-22.md#L86](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L86)
- [TEAM_LOG_2026-10-01_to_10-22.md#L96](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L96)
- [TEAM_LOG_2026-10-01_to_10-22.md#L98](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L98)
- [TEAM_LOG_2026-10-01_to_10-22.md#L104](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L104)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L365](../../records/SESSION_NOTES_2026-10-01_to_10-03.md#L365)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L372](../../records/SESSION_NOTES_2026-10-01_to_10-03.md#L372)

<details>
<summary>All result files and records (10)</summary>

- [attempt.json](attempt.json) · record
- [receipt.json](receipt.json) · record
- [eval_summary.txt](eval_summary.txt) · result
- [eval_333435.json](eval_333435.json) · result
- [lead_s37_chain.log](lead_s37_chain.log) · log
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [eval_333435-00c37f8c.json](eval_333435-00c37f8c.json) · result
- [eval_333435-cdff4089.json](eval_333435-cdff4089.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [SESSION_NOTES_2026-10-01_to_10-03.md](../../records/SESSION_NOTES_2026-10-01_to_10-03.md) · record

</details>

## Related work

- ← shipped in: [Young-column fourth-cumulant shifts reduce held-out error by 3.54% on all 24 networks](../../experiments/fourth-order-terms/young-column-fourth-cumulant-shifts-reduce-held-out-error/README.md)
- ← shipped in: [Fourth-cumulant shift strength 0.6 and per-unit kurtosis through age two win held-out selection](../../records/research/2026-10-02-fourth-cumulant-shift-strength-0-6-per-unit-kurtosis/README.md)
- ← shipped in: [Qualification requires quiet compute windows during instruction-count and worker stages](../../records/research/2026-10-02-qualification-requires-quiet-compute-windows-during-instruction-count-worker/README.md)
- ← shipped in: [Submission designation requires successful grading and measured wall and residual margins](../../records/research/2026-10-02-submission-designation-requires-successful-grading-measured-wall-residual/README.md)
- ← shipped in: [Young-pair hub coefficients give a non-isotropic kurtosis correction and 1.41% further error reduction](../../experiments/fourth-order-terms/young-pair-hub-coefficients-give-non-isotropic-kurtosis-correction/README.md)
- supersedes: [Submission #333429: float32 arithmetic, half-zero strips, refreshed setup keys and separate tree levels](../333429/README.md)
- ← supersedes: [Submission #333481: fitted dials, masked covariance correction and three-request operand loads](../333481/README.md)

[Research graph](../../graph/index.html#node=submission-333435) · [Copy provenance](../../PROVENANCE.json)

