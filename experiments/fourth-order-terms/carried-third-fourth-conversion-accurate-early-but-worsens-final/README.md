# Carried third-to-fourth conversion is accurate early but worsens final error by 3.8%

2026-10-02T08:23Z · experiment · rejected · measured

For Gaussian weights the per-unit fourth cumulant has a part (12/n) w_i' N w_i with N the contraction of the fourth cumulant; N gets a source from the third cumulant at every layer (one weight per bank column) and is transported with the columns, so it could be carried free as a second weight vector on the bank. Against the S37 physics on rank 651 the accumulated carrier is +3.8% at strength 1 and +1.5% at 0.5 while the one-layer source alone is -0.1% (layer 6 -1.5%): the transport does not close on one matrix. Later Monte Carlo showed the one-layer conversion explains only 10% / 4% / 1.5% / 0.2% of the matrix's unit-specific energy at layers 5 / 9 / 13 / 15.

Recorded measurement:

    accumulated strength 1: +3.8% (layer 6 -1.8%, layers 12-15 +1.4..+3.8%); 0.5: +1.5%; -0.5: +0.1%; one-layer source -0.1%; conversion explains 10% / 4% / 1.5% / 0.2% of Q's unit-specific energy at layers 5 / 9 / 13 / 15 (slope 1.0-1.35)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/e31c_carrier_vs_truth.py](code/e31c_carrier_vs_truth.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L195](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L195)
- [TEAM_LOG_2026-10-01_to_10-22.md#L209](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L209)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L479](../../../records/SESSION_NOTES_2026-10-01_to_10-03.md#L479)

<details>
<summary>All result files and records (4)</summary>

- [e30_rank0651_first.log](e30_rank0651_first.log) · log
- [e30_dump_rank0035.log](e30_dump_rank0035.log) · log
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [SESSION_NOTES_2026-10-01_to_10-03.md](../../../records/SESSION_NOTES_2026-10-01_to_10-03.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-carried-third-fourth-conversion-accurate-early-but-worsens-final) · [Copy provenance](../../../PROVENANCE.json)

