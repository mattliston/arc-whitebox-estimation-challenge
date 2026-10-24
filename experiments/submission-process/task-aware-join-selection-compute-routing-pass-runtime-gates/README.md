# Task-aware join selection and compute routing pass runtime gates but independent scoring removes the projected gain

2026-10-16T18:02:05Z · experiment · rejected · measured

S4 = S3pb plus option B (task-aware Nystrom join rows at layers 10 and 13, beta 1) and S5 = S4 plus the compute flag on (byte-identical predictions, -0.054% bill) both passed every count gate with P(any fail) 0-0.4% in the estimator submitted as #336412 (S2tr)-family noise model. When the store-32 check showed option B and the other value-only parts near zero, S5-class clean builds projected about 1.476e-9, a tie with #336412, and neither was fired; S5 stayed the SAFE-designation candidate and fire-ready.

Recorded measurement:

    S4 452bda56: slow 41,814/40,304 (margins +886/+896), worst 91.8 s, setup 4.442/4.587, median .3138, P(any/public) family 0.0/0.0%, bill +0.234% vs S2tr -> prereg 1.4707e-9 (~1.4728e-9 8-net). S5 8d1241cc: slow 42,130/40,620 (margins +570/+580), median .3155, worst 92.5 s, P family 0.4/0.4%, bill +0.181% vs S2tr -> 1.4706e-9 with B -0.154 (~1.475 if B null). S5+ms 53b3dd45: median .3116, slow 41,722/40,379, STEP6 fails on the v1 replay precondition. S5 anatomy: 41,846 / 40,553 requests, residual .292, P(loss) 1.9%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_passrel_v1/stack_v1/cand_s4.py](code/opus_passrel_v1/stack_v1/cand_s4.py)
- [code/opus_passrel_v1/stack_v1/cand_s5.py](code/opus_passrel_v1/stack_v1/cand_s5.py)
- [code/opus_passrel_v1/s4_v1/cand_s4base_v1.py](code/opus_passrel_v1/s4_v1/cand_s4base_v1.py)
- [code/opus_passrel_v1/stack_v1/cand_s5ms.py](code/opus_passrel_v1/stack_v1/cand_s5ms.py)
- [code/opus_callform_v1/cand_s5_ms_v1.py](code/opus_callform_v1/cand_s5_ms_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4381](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4381)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4387](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4387)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4430](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4430)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4433](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4433)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4452](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4452)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4471](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4471)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4495](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4495)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4508](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4508)

<details>
<summary>All result files and records (5)</summary>

- [STACK_s4.json](STACK_s4.json) · record
- [STACK_s5.json](STACK_s5.json) · record
- [frag_computeon_v1.py.txt](frag_computeon_v1.py.txt) · code
- [STACK_s5ms.json](STACK_s5ms.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-task-aware-join-selection-compute-routing-pass-runtime-gates) · [Copy provenance](../../../PROVENANCE.json)

