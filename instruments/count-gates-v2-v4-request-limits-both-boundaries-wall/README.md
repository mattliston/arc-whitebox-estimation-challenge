# Count gates v2-v4: request limits on both boundaries, wall and setup caps, instruction-weighted residual median

2026-10-16T15:15:48Z · tool · adopted · measured

count_gates_v2 checks a release or boundary directory against the static fire gate (first <= 42,700 and warm <= 41,200 requests on both the normal and 1.3x-slow boundary, projected worst wall <= 100 s, setup <= 4.60 s) and models P(any/public fail) for the residual and the wall with a per-network log model. v3 took the lead re-fit (worst wall <= 106 s, the estimator submitted as #336412 (S2tr)-family and all-Oct-16 noise). v4 added a hard line: the instruction-weighted projected median residual (the #336412 median times the warm request ratio times the Python-instructions-per-request ratio) must be <= 0.320 s, after LR3 showed that request counts alone miss client Python cost.

Recorded measurement:

    Calibration vs the lead model (post regime): .355 -> 87.8/68.9% (lead 89/70), .32 -> 19.7/12.7 (28/17), .31 -> 9.7/5.8, wall 96.6 s -> 12.7/9.4. v4 on LR2: PASS .3171 (instr/req x1.000 / x0.997); LR3 FAIL .3276 (x1.010 / x1.026). Repeated boundary runs of identical files differ by 0.3-1.4% in residual instructions per request.

## Programs

- [code/count_gates_v2.py](code/count_gates_v2.py)
- [code/count_gates_v3.py](code/count_gates_v3.py)
- [code/count_gates_v4.py](code/count_gates_v4.py)
- [code/lowres_probe_v1/gate_probe_v1.sh](code/lowres_probe_v1/gate_probe_v1.sh)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4254](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4254)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4263](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4263)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4281](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4281)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4293](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4293)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-10-16-count-gates-v2-v4-request-limits-both-boundaries-wall) · [Copy provenance](../../PROVENANCE.json)

