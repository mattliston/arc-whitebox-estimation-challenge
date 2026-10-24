# Contraction leaf buffers bound before optional frame fusion preserve complete setup coverage

2026-09-25T18:38Z · experiment · adopted · measured

Prioritizing the small leaf buffers preserves prediction and work even when optional fusion reaches the wall guard. Trimming the key list instead increases lazy-binding work.

Recorded measurement:

    2026-09-25 11:38 porter: SETUP FIX VERDICT (stack/setup_variants_report.txt): leaf-buffers-first (ksfirst) vs trim372 on S27r. Bit-identity in process with setup: both predictions identical; ksfirst flops identical, trim +0.030%/+0.009% (16 lazy keys). Paired census vs S27 (same load within each pair): ksfirst setup +0.039 s (3.505 vs 3.466), leaf buffers bound first at 0.031 s, all 410 keys bound, fusion cut to 327/388 by the guard (requests only), bill -3.267%/-3.277%; trim +0.073 s, 394 bound, bill -3.238%/-3.267%. ksfirst = the fix (in S27s/S27t/S28).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_porter_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/build_setup_variants.py](code/build_setup_variants.py)
- [code/census_boundary_setup_v2.py](code/census_boundary_setup_v2.py)
- [code/parity_layers_setup_v2.py](code/parity_layers_setup_v2.py)
- [code/stack/ksfirst_fragment.py](code/stack/ksfirst_fragment.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L1157](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1157)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [setup_variants_report.txt](setup_variants_report.txt) · record

</details>

## Related work

- shipped in: [Submission #332361: middle join rank 320 with refitted learned correction and matched dials](../../../submissions/332361/README.md)

[Research graph](../../../graph/index.html#node=2026-09-25-contraction-leaf-buffers-bound-before-optional-frame-fusion-preserve) · [Copy provenance](../../../PROVENANCE.json)

