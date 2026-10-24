# Caching four five-level left-weight transforms saves 1077536896 FLOPs with identical 14-network outputs

2026-09-19 · experiment · adopted · measured

Explicit weight-owner caches preserved every prediction byte while lowering the actual complete cold bill and qualified as an ancestor of later submissions.

Recorded measurement:

    The fixed four-site D5 weight-transform cache port passes all source, kernel,
    full-output, cost, client, runtime, existing14-panel, contract and package checks.
    Read5 predictions remain byte-for-byte identical on every layer of all14
    networks. Thus this is exclusively a cost reduction. Existing14 noise-corrected
    adjusted score is **3.883207214854186e-9**, versus Read5's3.893504844402904e-9:
    ratio0.9973551774146315,14/14 wins. Public8:4.288924275919752e-9; fresh6:
    3.3422511334334315e-9. This reuses existing panels, not untouched validation.
    Mean complete cold bill is405993892525.8571 FLOPs, versus407071429421.8571:
    mean saving1077536896 FLOPs. The direct local ratio to graded Cost150 projects
    3.917274777205696e-9 officially. That is an estimate, not a graded result or bank.
    The append-only source preserves the complete Read5 prefix and changes only
    four explicit W-left call sites, their prediction lifetime, and the D5 wrapper.
    The first D5 transformed left operand is copied into a separately owned,
    contiguous array; later eligible same-layer uses omit30 transform primitives.
    Additional peak cache capacity is68841472 bytes.
    Native kernel tests use1024x992 and1024x768 left operands with1024/384 RHS
    widths, a non-D5 fallback, foreign scratch overwrite, owner mismatch and layer
    retirement. All outputs match ordinary Read5 exactly. Actual integer bills
    match the creator-copy and omitted-transform formulas. Native35/100/35 plus
    independent cold100 preserve all16 outputs, repetition and warm/cold parity.
    The35/100 savings are1084542144/1082185216 FLOPs in each matching context.
    The real NumPy-free client35/100/35 matches native and Read5 outputs and the
    new exact bills. Full-context elapsed times are39.4934624992,37.5751984157 and
    38.2080836250 seconds, versus40.7486140421,37.5823506247 and38.3617779166.
    Pooled ratio0.98786558605 passes the fixed<=0.99 and<45s gates. This is a

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260919_v1/weight5_reuse_v1/build.py](code/codex_resume_20260919_v1/weight5_reuse_v1/build.py)
- [code/codex_resume_20260919_v1/weight5_reuse_v1/candidate.py](code/codex_resume_20260919_v1/weight5_reuse_v1/candidate.py)
- [code/codex_resume_20260919_v1/weight5_reuse_v1/native.py](code/codex_resume_20260919_v1/weight5_reuse_v1/native.py)
- [code/codex_resume_20260919_v1/weight5_reuse_v1/panel.py](code/codex_resume_20260919_v1/weight5_reuse_v1/panel.py)
- [code/codex_resume_20260919_v1/weight5_reuse_v1/prepare_package.py](code/codex_resume_20260919_v1/weight5_reuse_v1/prepare_package.py)
- [code/codex_resume_20260919_v1/weight5_reuse_v1/qualify_boundary.py](code/codex_resume_20260919_v1/weight5_reuse_v1/qualify_boundary.py)
- [code/codex_resume_20260919_v1/weight5_reuse_v1/qualify_native.py](code/codex_resume_20260919_v1/weight5_reuse_v1/qualify_native.py)
- [code/codex_resume_20260919_v1/weight5_reuse_v1/score_panel.py](code/codex_resume_20260919_v1/weight5_reuse_v1/score_panel.py)
- [code/codex_resume_20260919_v1/weight5_reuse_v1/submit.py](code/codex_resume_20260919_v1/weight5_reuse_v1/submit.py)
- [code/codex_resume_20260919_v1/weight5_reuse_v1/unit.py](code/codex_resume_20260919_v1/weight5_reuse_v1/unit.py)
- [code/astra_phase2_reexamination_v1/current_comp_compact_boxes_v2/rpc37_v1/estimator.py](code/astra_phase2_reexamination_v1/current_comp_compact_boxes_v2/rpc37_v1/estimator.py)
- [code/astra_phase2_reexamination_v1/current_comp_weight_transform_reuse_v1/helper.py](code/astra_phase2_reexamination_v1/current_comp_weight_transform_reuse_v1/helper.py)

<details>
<summary>Additional programs and dependencies (3)</summary>

- [code/astra_phase2_reexamination_v1/current_comp_weight_transform_reuse_v1/port.py](code/astra_phase2_reexamination_v1/current_comp_weight_transform_reuse_v1/port.py)
- [code/codex_resume_20260919_v1/read5_packet_v1/candidate.py](code/codex_resume_20260919_v1/read5_packet_v1/candidate.py)
- [code/codex_resume_20260919_v1/run_guarded.py](code/codex_resume_20260919_v1/run_guarded.py)

</details>

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L4](RESULT.md#L4)
- [RESULT.md#L8](RESULT.md#L8)
- [RESULT.md#L9](RESULT.md#L9)

<details>
<summary>All result files and records (3)</summary>

- [RESULT.md](RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-19-caching-four-five-level-left-weight-transforms-saves-1077536896) · [Copy provenance](../../../PROVENANCE.json)

