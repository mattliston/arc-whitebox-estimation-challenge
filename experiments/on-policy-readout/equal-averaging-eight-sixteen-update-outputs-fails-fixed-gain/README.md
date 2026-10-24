# Equal averaging of eight- and sixteen-update outputs fails the fixed gain and win gates

2026-09-30 · experiment · rejected · measured

A fixed half-average of saved outputs was evaluated as a diagnostic only. The assumed16-update bill was not a measured ensemble cost, and output averaging did not establish parameter-average behavior.

Recorded measurement:

    Raw-.4250793%,hypothetical adjusted-.5455300%,5wins,maxreg+.07296148%;both descriptive CIs negative;fixed-.65%/6win gate failed.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/hill_climb_rrpi_detached_star_avg8_16_v1/export_average.py](code/hill_climb_rrpi_detached_star_avg8_16_v1/export_average.py)
- [code/hill_climb_rrpi_detached_star_avg8_16_v1/forecast.py](code/hill_climb_rrpi_detached_star_avg8_16_v1/forecast.py)
- [code/hill_climb_rrpi_detached_star_avg8_16_v1/native_parameters.py](code/hill_climb_rrpi_detached_star_avg8_16_v1/native_parameters.py)
- [code/hill_climb_pair_rr_inverse_v1/candidate.py](code/hill_climb_pair_rr_inverse_v1/candidate.py)

## Evidence

- [HILL_CLIMB_CURRENT.md#L285](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L285)

<details>
<summary>All result files and records (1)</summary>

- [HILL_CLIMB_CURRENT.md](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-30-equal-averaging-eight-sixteen-update-outputs-fails-fixed-gain) · [Copy provenance](../../../PROVENANCE.json)

