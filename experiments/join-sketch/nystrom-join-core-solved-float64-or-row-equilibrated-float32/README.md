# Nystrom join core solved in float64 or row-equilibrated float32: adjusted +0.41% and +0.01%; core condition 4.5-5.9

2026-10-19T05:20:12Z · experiment · rejected · derived

Solving the Nystrom join core (QR of Y and the core solve) in float64 raised the adjusted score because float64 linear algebra is billed at twice the rate, and a row-equilibrated float32 solve was noise. A numerics agent found the inverted core matrix Qn[rows] has condition 4.5-5.9 at every join (cond(Y) up to 9e5 is irrelevant because Y is never inverted), float64 or equilibrated solves move the core by 3e-8 relative, and the only above-epsilon deviation is production kernel rounding in forming Y, which the landmark symmetrization already averages: join numerics are not a lever.

Recorded measurement:

    joinf64 23117001 (7 ranks vs 95c): raw -0.130% (se .13, 4/7 lower), bill +0.536%, adjusted +0.41%; eqsolve db20b09e: adjusted -0.236% (se .11), raw -0.096%, bill -0.140% = +0.01% vs #337028; solve64 judge retired by the 7.5 GB guard at rank 200 (RSS 6.94 GB, footprint 7.68 GB); cond(Qn[rows]) 4.5-5.9; core moves 3e-8 relative; shipped f32 + symmetrization within 4.5e-8 of the best symmetric f64 core; largest deviation 1.5e-5 at rank 200 L7

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_join_f64_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_jsym_joinf64_v1.py](code/candidate_jsym_joinf64_v1.py)
- [code/candidate_jsym_eqsolve_sci_v1.py](code/candidate_jsym_eqsolve_sci_v1.py)
- [code/candidate_jsym_solve64_sci_v1.py](code/candidate_jsym_solve64_sci_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5900](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5900)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5906](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5906)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5910](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5910)

<details>
<summary>All result files and records (3)</summary>

- [science7_v1.log](science7_v1.log) · log
- [science7_eqsolve_sci.log](science7_eqsolve_sci.log) · log
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-19-nystrom-join-core-solved-float64-or-row-equilibrated-float32) · [Copy provenance](../../../PROVENANCE.json)

