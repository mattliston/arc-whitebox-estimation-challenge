# Submission #333587: young-pair fourth-chaos plane, norm-coupling dial solve and fewer kernel requests

2026-10-02T16:51:05Z · submission · adopted · measured

candidate generation 41 (S41) is candidate generation 40 (S40) plus the fourth-chaos part of the young pairs' local quartic form folded into the same read (ages 1-2, strength 1), the first joint solve of the law's dials (fold strength 1.83 / 1.40 / 1.18 by layer band, factor 1.272 on the removed multiple of C, source strengths 1.142 / 0.697 / 0.791, c0 0.6222, k31 knob -0.1659), the kurtosis birth by tables, the pair law's faster kernels and the release lane's request diet (leaf views, 64M un-fuse, 128 first-load views at bind). Qualified without a quiet window (stage C on load-independent counts, stage D with its swap line unjudged on the shared box); graded 1.7110e-9, third, after waiting 59 minutes in the platform's queue; its run came out 3-4% slow in everything.

Recorded measurement:

    adjusted 1.7114087400000878e-9 (S40 1.7590e-9, -2.73%; expected about 1.71e-9); final-layer MSE 1.4620e-8 (-2.99%; native ten fresh -3.26% [-4.97,-1.66], 9 of 10), all-layer 5.785e-9 (-2.4%), utilisation 0.11703521822889343 (+0.27%); 100/100, wall 83.63 / 90.22 s (projected 80.9 / 92.4; mean 2.7 s above, kernel 74.28 s), residual 0.318 / 0.354 s (projected 0.305-0.309 / 0.361); stage C bill 1.00288 x S40, requests 42,764 a network (first 43,466 / warm 42,062); stage D handles 56,691, footprint 3.57 GB, byte-identical parity; torch: K31L4 -1.87% (se 0.39), dials -1.48% (se 0.28) on 25 ranks; J2W 1.5995e-9 (error 1.4567e-8 = ours, share 0.1091), marius_binner 1.6407e-9 ahead by 6.5% and 4.1%

Public score: 1.7114087400000878e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)
- [../../experiments/wall-and-residual/shared-strip-views-selective-tree-fusion-binding-time-operand/code/release/variants/s41hv_33bac9db_lvfs64_l3b128/candidate.py](../../experiments/wall-and-residual/shared-strip-views-selective-tree-fusion-binding-time-operand/code/release/variants/s41hv_33bac9db_lvfs64_l3b128/candidate.py)
- [fragment_np_v8.py](fragment_np_v8.py)
- [candidate.py](candidate.py)

## Evidence

- [SUBMISSION_LEDGER.md#L917](../../records/SUBMISSION_LEDGER.md#L917)
- [SUBMISSION_LEDGER.md#L918](../../records/SUBMISSION_LEDGER.md#L918)
- [TEAM_LOG_2026-10-01_to_10-22.md#L366](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L366)
- [TEAM_LOG_2026-10-01_to_10-22.md#L369](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L369)
- [TEAM_LOG_2026-10-01_to_10-22.md#L376](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L376)
- [TEAM_LOG_2026-10-01_to_10-22.md#L382](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L382)
- [TEAM_LOG_2026-10-01_to_10-22.md#L383](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L383)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L595](../../records/SESSION_NOTES_2026-10-01_to_10-03.md#L595)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L617](../../records/SESSION_NOTES_2026-10-01_to_10-03.md#L617)

<details>
<summary>All result files and records (9)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [attempt.json](attempt.json) · record
- [receipt.json](receipt.json) · record
- [eval_summary2.txt](eval_summary2.txt) · result
- [eval_333587.json](eval_333587.json) · result
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [eval_333587-6a348b1f.json](eval_333587-6a348b1f.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [SESSION_NOTES_2026-10-01_to_10-03.md](../../records/SESSION_NOTES_2026-10-01_to_10-03.md) · record

</details>

## Related work

- ← shipped in: [Designation caps set to 43,800 first-network requests, 42,700 warm requests and 82 s mean wall](../../records/research/2026-10-02-designation-caps-set-43-800-first-network-requests-42/README.md)
- ← shipped in: [Joint norm-coupling dial solve lowers MSE 1.48% on 25 ranks and enters submission #333587](../../experiments/dials/joint-norm-coupling-dial-solve-lowers-mse-1-48/README.md)
- ← shipped in: [Shared strip views, selective tree fusion and binding-time operand views reduce requests without changing predictions](../../experiments/wall-and-residual/shared-strip-views-selective-tree-fusion-binding-time-operand/README.md)
- ← shipped in: [Young-pair fourth-chaos (3,1) plane lowers MSE 1.87% on 25 ranks and enters submission #333587](../../experiments/fourth-order-terms/young-pair-fourth-chaos-3-1-plane-lowers-mse/README.md)
- supersedes: [Submission #333559: norm-coupling law on column pairs folded into the third-order plane read](../333559/README.md)

[Research graph](../../graph/index.html#node=submission-333587) · [Copy provenance](../../PROVENANCE.json)

