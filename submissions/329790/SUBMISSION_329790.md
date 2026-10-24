# Phase 2 checkpoint submission 329790

Accepted at 2026-09-04T22:01:46.007Z (server timestamp).
Candidate: candidate_recovered_cp3_8k_v1.py.
Archive: recovered_cp3_8k_v1.tar.gz, containing only estimator.py and manifest.json.

Independent-4M grader-style panel: public8 adjusted 1.605e-8, raw 3.486e-8,
util 0.4597; virgin6 adjusted 1.302e-8, raw 2.873e-8, util 0.4546.
Adjusted gains vs submitted cap256 baseline: 10.4% and 9.3%, respectively.
This is an incremental checkpoint, not achievement of the ~4e-9 goal.

Standard whest validation passed. On development ranks35 and100 all 15
live learned-head feature arrays and correction arrays match the instrument
bit-for-bit. The raw-output parity gate in recovered_parity_v2.json says
false because the instrument can emit small negative means while the
candidate clamps them to zero; the per-layer differences equal those
negative minima (checked against the instrument's saved post_mean arrays).
The first parity attempt, v1, omitted RETRO_DUMP_FEAT=1, which also supplies
some live mechanistic inputs; its failed comparison is retained as a test
configuration error, not evidence of a candidate mismatch.

Official grade (polled 2026-09-04): adjusted 1.436993412541317e-8,
raw 3.1560041051648114e-8. This is a 10.84% adjusted improvement
relative to submission 329754 (1.611649718906911e-8).
The ~4e-9 target is not achieved. No promotion or commit performed.
