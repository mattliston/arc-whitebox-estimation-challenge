# Phase 2 checkpoint submission 329843 — accepted, grading pending

Matt explicitly requested submitting the fully validated local 4.514e-9
candidate as a bank. This authorizes this exact source upload; the earlier
upload-permission blocker is resolved for this submission. One submission
was created; no duplicate upload, promotion, or commit.

AIcrowd accepted at **2026-09-05T04:46:47.155Z** = September 4, 2026,
21:46:47 PDT. Challenge ID 1174, current round ID 1429. Initial status
`submitted`, message `Successfully enqueued 1 Job`, scores null.

Candidate: `candidate_fullpair_scalar_final_cpcubes_v1.py`.
Source SHA256:
`c08b6f3b688cc6ff0484db5d493b10caa91da74bcfd8d3b04828ad3754ceb973`.
Archive: `fullpair_scalar_final_cpcubes_v1_whest016.tar.gz`.
Archive SHA256:
`bb63190e9ea596dc0c57f4a93ae2efe4ba4bcef8cabc0e50d3386747ab697a9f`.
Exact archive was passed positionally to Whest .16, not repackaged. Immediately
before upload, source hash, archive hash, manifest source hash, and archive
members (only estimator.py + manifest.json) were checked against the frozen
`verified_local_fullpair_cpcubes_v1.json` receipt. All passed.

Independent 4M noise-corrected local evaluation:

| Panel | Raw final MSE | Mean utilization | Adjusted |
| --- | ---: | ---: | ---: |
| Public 8 | 2.45202979646e-8 | .1998125 | 4.90596127732e-9 |
| Virgin 6 | 2.00466680601e-8 | .198383333 | 3.99134667266e-9 |
| All 14 | 2.26030280056e-8 | .1992 | 4.51398358961e-9 |

Native repeated parity and actual NumPy-free remote execution/resource gates
passed for this exact source. These LOCAL results are not an official grade.
Before this submission, official best was #329790 adjusted 1.43699341254e-8.
The approximately 4e-9 goal remains unproven and active pending official
grading and any further improvements needed.

Whest upload/watch process session 71061 returned terminal exit 0 with the
accepted ID. Poll this ID; do not repeat the upload to recover a status.
Later grades must be recorded in a new versioned receipt, leaving this
acceptance snapshot intact.
