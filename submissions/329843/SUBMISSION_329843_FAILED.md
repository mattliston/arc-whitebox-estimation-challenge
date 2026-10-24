# Phase 2 checkpoint 329843 — official scoring failed

The API reports terminal status `failed`, scores null, and generic message
`Error : Error while scoring your submission`. Matt supplied the detailed
official grader diagnostic on September 4 PDT:

- `REPEATED_MLP_FAILURE`, after 4 participant subprocess respawns, limit 3.
- All 4 causes were `TIME_EXHAUSTED` at the 120-second per-MLP wall limit.
- No MLP completed, no score, and nothing counted against the quota.
- This is estimator runtime failure, not evidence of a grader fault.

The local exact-source NumPy-free test was 85.10 seconds on this Mac. That
did not provide sufficient official-worker headroom. Its local adjusted
4.51398358961e-9 remains a LOCAL accuracy/cost result, not a banked score.
Official best remains #329790, adjusted 1.43699341254e-8.

Do not re-upload the failed archive unchanged. Pause new accuracy refits
until a runtime-compliant deployment path is established. Preserve the
source, package, acceptance snapshot, and all successful local receipts;
they accurately describe their narrower test environment.

New objective threshold clarified by Matt: [instruction removed].
The active goal remains unmet. No promotion or commit.

Completed diagnostics immediately before the failure report:
`current_prefix_gain_r35_v1.json` / r100: .9 prefix gain improves only
.42% / .18%; 1.1 worsens both, no gain calibration/refit justified.
`rectangular_schedule_value_v1.json`: fixed-arena D3..D7 schedule alternatives
project only .137% of total old-census FLOPs saved; no port warranted.
`compact_activeonly_r35_v1.json` / r100: .05 active cutoff with .01 births
makes unlearned raw 5.64x / 6.43x worse than .01/.01. No threshold sweep/refit.
Correlation-graph refit and Gaussian triangle-source ideas were discussed
but NO new rows, trainer, or candidate were launched for either.
