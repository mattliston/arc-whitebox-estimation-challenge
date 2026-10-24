# Previous-state per-unit fourth- and third-cumulant readouts show no zero-weight signal

2026-10-10T05:05Z · experiment · rejected · measured

The lead's readouts o5 / o6 (the per-unit fourth cumulant and the third-cumulant plane with index multiplicities (2,1) (K21) diagonal's third cumulant before the pair map, each read from the 48 message columns of the previous state as one 48-vector per layer) were added to the moment-closure estimator beside batch 6 and certified bit for bit at zero weight; in the one zero-weight table over 36 training networks at candidate generation 42 (S42)'s point they show no signal (readout k4m 11 of 672 weights beyond 3 se, mean t^2 1.29, split-half cosine -0.17; k3m 1 of 672, 1.02, -0.18; no-signal reference 0.3% of weights and t^2 1.0), as do the row-gain readout and the m2 / k3d2 / skew- and kurtosis-corrected transport gate (gk) / gb / eold columns, while the law diagonal at layer 15, local quartic of old pairs (qold) and f15 carry the strongest signal.

Recorded measurement:

    readout k4m 11 of 672 / 1.29 / -0.17; k3m 1 of 672 / 1.02 / -0.18; readout b 24 of 896 / 2.10 / +0.44; rad 9 of 896 / 1.66 / +0.51; row 0 of 104 / 1.28 / +0.15; law diagonal at 15 104 / 44.56 / +1.00; qold 56 / 9.20 / +0.94; f15 26 / 6.43 / +0.85; m2 1 / 1.76; gk 0 / 0.65; gb 4 / 1.82; eold 0 / 0.93; qb 8 / 2.63; xk4 6 / 3.23

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_reexam_20261001_v1/trainer/otr_m.py](code/opus_reexam_20261001_v1/trainer/otr_m.py)
- [code/opus_reexam_20261001_v1/physicist/b6cols.py](code/opus_reexam_20261001_v1/physicist/b6cols.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/opus_reexam_20261001_v1/physicist/b7cols.py](code/opus_reexam_20261001_v1/physicist/b7cols.py)
- [code/opus_reexam_20261001_v1/physicist/f4b.py](code/opus_reexam_20261001_v1/physicist/f4b.py)
- [code/opus_reexam_20261001_v1/trainer/otr_common.py](code/opus_reexam_20261001_v1/trainer/otr_common.py)
- [code/opus_reexam_20261001_v1/trainer/otr_pl.py](code/opus_reexam_20261001_v1/trainer/otr_pl.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L531](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L531)
- [TEAM_LOG_2026-10-01_to_10-22.md#L555](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L555)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-10-previous-state-per-unit-fourth-third-cumulant-readouts-show) · [Copy provenance](../../../PROVENANCE.json)

