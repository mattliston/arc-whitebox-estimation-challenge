# Transporting omitted units' own third cumulants worsens raw error +0.13% and +3.1%

2026-10-20T05:07:36Z · experiment · rejected · measured

Transporting non-roster units own third cumulants through cubed weights failed on networks 35 / 100; including all units produced +26%. The missing third-cumulant content was attributed to cross terms rather than omitted diagonal transport.

Recorded measurement:

    - [fable-lead] 2026-10-20T05:07:36Z OWN-CUMULANT DIAGONAL TRANSPORT DEAD (native oracle, nets 35/100): non-roster units' own k3 pushed through W^3 adds 1.5% rms to k3 and reads +0.13% / +3.1% (worse); k4 version -0.3 / +0.1; all-units version +26% (the bank already carries the roster hubs' own skewness). The per-unit k3 deficit (card = 0.85 x truth) is in the cross terms (pair-born (2,1) content), not in dropped units.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_rel_nosync_sci_v1.py](code/candidate_rel_nosync_sci_v1.py)
- [code/native_cov_oracle_v2.py](code/native_cov_oracle_v2.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6135](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6135)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-transporting-omitted-units-own-third-cumulants-worsens-raw-error) · [Copy provenance](../../../PROVENANCE.json)

