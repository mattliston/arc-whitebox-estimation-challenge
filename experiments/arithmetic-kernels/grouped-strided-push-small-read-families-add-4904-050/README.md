# Grouped strided push and small-read families add 490–4,050 cold requests and fail the static runtime limits

2026-10-16T16:40:44Z · experiment · rejected · measured

The two Smirnov (grouped strided-read kernel) families that had failed on wall were re-priced on the clean S3 build against its request margins: the small join/old-bank family adds about 170 requests per routed call (+4,050 first-network requests on the estimator submitted as #336412 (S2tr)) and is out before any wall test; the push family v2 at the 400 and 100 MF/Gcyc rules binds its lazy shapes on the cold network (+490 / +1,171 first requests) and even with a static cold-first skip lands +202 requests on the second network for +0.12% bill, leaving the slow warm margin under the 300 rule. Moving the push bind into setup would push the 1.3x slow setup to about 4.75 s, over the 4.60 s gate.

Recorded measurement:

    pushSM r400 on S3: first +490 requests (slow 42,880 FAIL), warm -62, bill first +0.06% / warm -0.113%, worst wall 95.3 s (+3.7); r100: first +1,171 / warm +607, median .3209, wall 100.9 s; r400c (cold skip): second net +202 requests, +0.12% bill (slow warm 40,974, margin 226 < 300), public mix ~-0.08%; r100c second +889. Bind cost ~550 requests on S3 (179 on GOV v4g). Setup pace 17,660 requests / ~4.25 s = 0.24 ms each -> ~0.13 s nominal, ~0.17 s at 1.3x: slow setup 4.585 -> ~4.75 s.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_passrel_v1/smx_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cand_s3_psm_r400_v1.py](code/cand_s3_psm_r400_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4334](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4334)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4340](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4340)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4351](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4351)

<details>
<summary>All result files and records (4)</summary>

- [fragment_smx_coldskip_v1.py.txt](fragment_smx_coldskip_v1.py.txt) · code
- [RULE_pushsm_r400_v1.json](RULE_pushsm_r400_v1.json) · config
- [RULE_pushsm_r100_v1.json](RULE_pushsm_r100_v1.json) · config
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-grouped-strided-push-small-read-families-add-4904-050) · [Copy provenance](../../../PROVENANCE.json)

