# Native final-layer head ridge refit: held-out 49-network adjusted +0.016%, control-24 +0.504%

2026-10-16T09:15:50Z · experiment · rejected · measured

A 146-network capture supported a grouped cross-validated ridge fit of the 65 final head parameters. The fitted 38% parameter displacement overfit training networks; low-dimensional scale/bias alternatives were also negligible.

Recorded measurement:

    97trainCV-.123%,lambda1e-3;49held+.016%,SE.26,aggregate+.040%;controls+.504%,SE.34,aggregate+.729%;headnormchange.38;scale.964held-.01%/control-.00%;scale+biasheld-.06%/control+.04%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/opus_headfit_v1/build_headfit_v1.py](code/lead/runs/opus_headfit_v1/build_headfit_v1.py)
- [code/lead/runs/opus_headfit_v1/cand_s2tr_headfit_v1.py](code/lead/runs/opus_headfit_v1/cand_s2tr_headfit_v1.py)
- [code/lead/runs/opus_headfit_v1/check_headfit_v1.py](code/lead/runs/opus_headfit_v1/check_headfit_v1.py)
- [code/lead/runs/opus_headfit_v1/fit_head_lowdim_v1.py](code/lead/runs/opus_headfit_v1/fit_head_lowdim_v1.py)
- [code/lead/runs/opus_headfit_v1/fit_head_v1.py](code/lead/runs/opus_headfit_v1/fit_head_v1.py)
- [code/lead/runs/opus_headfit_v1/capture_head_v1.py](code/lead/runs/opus_headfit_v1/capture_head_v1.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)
- [code/lead/runs/opus_s2tr_v1/candidate_S2tr_v1.py](code/lead/runs/opus_s2tr_v1/candidate_S2tr_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4068](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4068)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4066](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4066)

<details>
<summary>All result files and records (4)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [fragment_headfit_v1.py.txt](fragment_headfit_v1.py.txt) · result
- [FIT_OUTPUT_hunter_waiter_v1.txt](FIT_OUTPUT_hunter_waiter_v1.txt) · result
- [SEEDS_headfit_v1.json](SEEDS_headfit_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-native-final-layer-head-ridge-refit-held-out-49) · [Copy provenance](../../../PROVENANCE.json)

