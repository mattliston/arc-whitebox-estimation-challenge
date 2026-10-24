# Later approximate wall triggers recover a modeled 0.116% public-only gain from the expanded read grid

2026-10-16T11:08:42Z · experiment · inconclusive · measured

An exact-relief-aware replay allowed the larger read table to retain most saving while delaying expensive approximation. The configuration was marginal under late detection and was held as a public-only candidate.

Recorded measurement:

    LEAN118to122s,SHED123to125s;net-.116%vsswap;L6Ppublic4.1-8.0%,L8 11.8%;forcedexactwarmFLOPs+1.70%,instructions-30%,localwall-40%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_wallx_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cand_pubmax_A_v4h_release.py](code/cand_pubmax_A_v4h_release.py)
- [code/designate_wallx_v1.py](code/designate_wallx_v1.py)
- [code/test_A_fx3.py](code/test_A_fx3.py)
- [code/test_A_nofx.py](code/test_A_nofx.py)
- [code/test_swap_fx3.py](code/test_swap_fx3.py)
- [code/test_swap_nofx.py](code/test_swap_nofx.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4154](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4154)

<details>
<summary>All result files and records (3)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [WALLX_RESULT_v1.json](WALLX_RESULT_v1.json) · result
- [DESIGNATE_wallx_v1.json](DESIGNATE_wallx_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-later-approximate-wall-triggers-recover-modeled-0-116-public) · [Copy provenance](../../../PROVENANCE.json)

