# Rank-15 rectangular primitive: 4.267895711% fewer FLOPs but 3.169052 times CPU

2026-09-28 · experiment · rejected · measured

The fixed rank15 rectangular multiplication primitive passed twelve arithmetic and error fixtures. Its larger CPU cost was a material risk before full-estimator integration.

Recorded measurement:

    Warm3,501,135,335F versus3,657,221,745F(-4.267895711%);bind651,273,255F;error4.92176669e-5<2e-4;medianCPU.550797s versus.173805s,ratio3.169052.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/hill_climb_hk15_primitive_v1/kernel.py](code/hill_climb_hk15_primitive_v1/kernel.py)
- [code/hill_climb_hk15_primitive_v1/prepare_programs.py](code/hill_climb_hk15_primitive_v1/prepare_programs.py)
- [code/hill_climb_hk15_primitive_v1/run.py](code/hill_climb_hk15_primitive_v1/run.py)
- [code/hill_climb_star_projection32_v1/candidate.py](code/hill_climb_star_projection32_v1/candidate.py)
- [code/teammate_s32_hk_residency_trace_v1/assess.py](code/teammate_s32_hk_residency_trace_v1/assess.py)
- [code/teammate_s32_hk_residency_trace_v1/observer.py](code/teammate_s32_hk_residency_trace_v1/observer.py)
- [code/teammate_s32_hk_residency_trace_v1/prospective_union_price_v1.py](code/teammate_s32_hk_residency_trace_v1/prospective_union_price_v1.py)
- [code/teammate_s32_hk_residency_trace_v1/run.py](code/teammate_s32_hk_residency_trace_v1/run.py)
- [code/teammate_s32_rank15_dispatch_v1/price.py](code/teammate_s32_rank15_dispatch_v1/price.py)

## Evidence

- [HILL_CLIMB_before_hk_train8_closeout_v1.md#L31](../../flop-accounting/base-factor-cache-36-setup-bindings-combined-eight-network/HILL_CLIMB_before_hk_train8_closeout_v1.md#L31)

<details>
<summary>All result files and records (5)</summary>

- [HILL_CLIMB_before_hk_train8_closeout_v1.md](../../flop-accounting/base-factor-cache-36-setup-bindings-combined-eight-network/HILL_CLIMB_before_hk_train8_closeout_v1.md) · record
- [RESULT.md](RESULT.md) · record
- [result.json](result.json) · result
- [assessment.json](assessment.json) · result
- [PROTOCOL.md](PROTOCOL.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-28-rank-15-rectangular-primitive-4-267895711-fewer-flops-but) · [Copy provenance](../../../PROVENANCE.json)

