# Disabling grouped strided-read plans added 1.01% bill while reducing backend time by 11.9%

2026-10-17T19:34:26Z · experiment · rejected · measured

Disabling the existing grouped strided-read plans traded a modest arithmetic saving back for lower server time. The plans already consumed wall-clock margin, so neither enabling greater depth nor removing them supplied the required improvement.

Recorded measurement:

    Rank 35 bill +1.01%, requests +602, server backend -11.9%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_fix336687_v1/cheap`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cand_ott20.py](code/cand_ott20.py)
- [code/cand_dcz4e.py](code/cand_dcz4e.py)
- [code/cand_peel16.py](code/cand_peel16.py)
- [code/cand_lr002e.py](code/cand_lr002e.py)
- [code/cand_lr002a.py](code/cand_lr002a.py)
- [code/cand_k31a4.py](code/cand_k31a4.py)
- [code/cand_rrf02.py](code/cand_rrf02.py)
- [code/cand_lr005e.py](code/cand_lr005e.py)
- [code/cand_rs14o0.py](code/cand_rs14o0.py)
- [code/cand_dc25.py](code/cand_dc25.py)
- [code/cand_lcr7.py](code/cand_lcr7.py)
- [code/cand_lcr192.py](code/cand_lcr192.py)

<details>
<summary>Additional programs and dependencies (6)</summary>

- [code/cand_k31a4b11.py](code/cand_k31a4b11.py)
- [code/cand_af005e.py](code/cand_af005e.py)
- [code/cand_lr002m.py](code/cand_lr002m.py)
- [code/cand_final64.py](code/cand_final64.py)
- [code/cand_lcr128.py](code/cand_lcr128.py)
- [code/cand_bf005e.py](code/cand_bf005e.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4910](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4910)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-10-17-disabling-grouped-strided-read-plans-added-1-01-bill) · [Copy provenance](../../../PROVENANCE.json)

