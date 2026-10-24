# Old-tail diagonal trace without radial repair: expanded adjusted gain falls to 0.2283%

2026-09-26 · experiment · rejected · measured

Restoring only the old diagonal trace while retaining the masked-tensor radial formula passed two training cases but lost most of the gain after expansion.

Recorded measurement:

    TRAIN2 adjusted -1.2791%; TRAIN6 -0.2283%, CI [-1.7976%, +1.0774%]; raw +1.1040%, bill -1.2971%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_old_tail_trace_z25_v1.py](code/candidate_old_tail_trace_z25_v1.py)

## Evidence

- [RESULTS.md#L429](../all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md#L429)

<details>
<summary>All result files and records (1)</summary>

- [RESULTS.md](../all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-26-old-tail-diagonal-trace-without-radial-repair-expanded-adjusted) · [Copy provenance](../../../PROVENANCE.json)

