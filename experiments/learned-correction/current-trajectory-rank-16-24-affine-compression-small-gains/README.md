# Current-trajectory rank-16/24 affine compression: small gains vanish in the width-eight edge stack

2026-10-21T00:59:13Z · experiment · rejected · measured

Current-trajectory affine compression gave small negative full-32 effects. Combining the rank-16 variant with the width-eight edge was slightly worse than the exact edge parent, unlike an additive prediction.

Recorded measurement:

    Current-trajectory rank16 full32-.0838225%/held-.125484%; rank24-.0876187%/held-.0431314% versus422, weak/noisy. These are distinct from rejected OLDoffpolicyrank24. Root built exploratory edge8+CURRENT rank16 via make_stack: SCI305d8457/RELb1d6cd4c, D/edge8_current_rank16_stack_v1, exactcomponentAST/literals/importPASS. VM PID50261 launched00:56:03Z, jobs12/shared36, plan7508c502; actualstackgainunknown, no auto fire. Do not sum componentgains.
    Completed closures: corrected bank8 parent16 carriers SCI6995e7db loses +0.2620721% Mac32 and +0.133101% Linux32 (held +0.125422%); originalMac wrapper ended00:59:17Z, resumed old queue/five judges, BEST422 unpinned. Edge8 + CURRENT rank16 SCI305d8457 is -0.207929% Linux32/held -0.193720%, worse than edge8 alone -0.2140704%; do not sum components or promote. Corrected CM4 carrier-only +0.655903% adjusted/held +0.688501%; fixed first-affine + carriers +0.781722%/held +0.877294%; all32/12 guards good. Close this CM4 compression direction.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_oct20_bill_v1/edge8_current_rank16_stack_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/prepare_v1.py](code/prepare_v1.py)
- [code/check_import_v1.py](code/check_import_v1.py)
- [code/tail_exact337422_from_v4a_v1.py](code/tail_exact337422_from_v4a_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6430](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6430)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6437](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6437)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [BUILD_v1.json](BUILD_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-current-trajectory-rank-16-24-affine-compression-small-gains) · [Copy provenance](../../../PROVENANCE.json)

