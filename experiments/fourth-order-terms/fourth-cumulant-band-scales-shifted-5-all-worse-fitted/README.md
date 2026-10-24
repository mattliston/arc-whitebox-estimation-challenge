# Fourth-cumulant band scales shifted by 5%: all worse, with fitted optima within 1%

2026-10-19T18:32:00Z · experiment · adopted · measured

The three fourth-cumulant band scales were moved 5% either way on 28 store ranks at zero bill: every move is worse, so each band sits at its network-average optimum; the parabola through each pair locates optima within 1% of the native value (x0.991 / x1.007 / x0.990) worth at most 0.04% each, and those optima entered the micro-stack. A seven-rank read of band 2 (x1.05 at -0.78%) was source-set noise (+0.95% on the held-out 21): dials are read on all ranks.

Recorded measurement:

    band 3 (L12-14): x0.95 raw +0.56%, adjusted +0.64% (se .25), held-out +0.66%; x1.05 raw +1.6%, adjusted +0.74% (worse 7/7; lane killed at 16 ranks); optimum x0.990 for -0.04%; band 2 (L8-11): x0.95 raw +1.07% (adj +1.14%), x1.05 raw +0.58% (held-out +0.88%); optimum x1.007 for -0.02%; band 1 (L1-7): x0.95 +0.24%, x1.05 +0.54%; optimum x0.991 for -0.01%; curvatures 168/342/361 %/unit^2; per-rank band-2 slopes sd 33 %/unit

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_k4band_b1m_sci_v1.py](code/candidate_k4band_b1m_sci_v1.py)
- [code/candidate_k4band_b1p_sci_v1.py](code/candidate_k4band_b1p_sci_v1.py)
- [code/candidate_k4band_b2m_sci_v1.py](code/candidate_k4band_b2m_sci_v1.py)
- [code/candidate_k4band_b2p_sci_v1.py](code/candidate_k4band_b2p_sci_v1.py)
- [code/candidate_k4band_b3m_sci_v1.py](code/candidate_k4band_b3m_sci_v1.py)
- [code/candidate_k4band_b3p_sci_v1.py](code/candidate_k4band_b3p_sci_v1.py)
- [code/fd_parabola_v1.py](code/fd_parabola_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5999](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5999)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6001](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6001)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6003](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6003)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6005](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6005)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6007](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6007)

<details>
<summary>All result files and records (2)</summary>

- [RESULT32_v1.json](RESULT32_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- shipped in: [Submission #337212: combined zero-cost dial optima on #337142, graded 1.4296259e-9](../../../submissions/337212/README.md)

[Research graph](../../../graph/index.html#node=2026-10-19-fourth-cumulant-band-scales-shifted-5-all-worse-fitted) · [Copy provenance](../../../PROVENANCE.json)

