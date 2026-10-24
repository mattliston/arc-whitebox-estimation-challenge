# Vectorized pair-polynomial fusion saves 1819768776 FLOPs but misses 12/14 panel-wins gate

2026-09-19 · experiment · rejected · measured

Fusing series work and eliminating unused plane construction passed development scoring but won only eleven panel networks.

Recorded measurement:

    ordinary1B development checks. It does not qualify for a board upload:
    its fixed14-network panel wins11/14, below the predeclared12/14 requirement.
    The older September13 RPC component gate remains failed and unchanged.
    Only the series helper and unused T31 construction/cast in two callers differ
    asymmetry and FP32 association are retained/reordered respectively, with no
    The full-width helper saves125789180F; removing T31 construction/cast saves
    4194304F per pair. Fourteen pair calls yield exactly1819768776F saved per
    35/100/35 plus independent cold100 repeat exactly; first layer matches Lazy.
    Development raw ratios are1.0021265463/1.0033758399; own-cold adjusted ratios
    .9975454711/.9988212287, pooled.9981807591. The development gate passes.
    The fixed existing4M panel was scored after all outputs and bills were sealed.
    | Public8 | 4.2006435396e-9 | 4.2169820297e-9 | .9961255491 | 6/8 |
    | Fresh6 | 3.2722334405e-9 | 3.2819169879e-9 | .9970494234 | 5/6 |
    | All14 | 3.8027534972e-9 | 3.8162398689e-9 | .9964660576 | 11/14 |
    Three ratios exceed1: ranks35(1.0016043017),100(1.0017584351),
    3016503(1.0012674893). Ordinary1B development and4M panel reference estimates
    are distinct; their different rank35/100 outcomes are preserved. These reused
    The3.8029621540e-9 official-parent projection is only arithmetic, not a result.
    Sixteen full predictions total6317049140725F. All17 guarded stages completed
    and retired,500.0443seconds in summed guard elapsed time, peak6518603776B
    including worker self-report, zero added swap. Terminal sessions45656,49823,
    52010,56230 are reaped. No numerical or upload worker remains. About3.8MB of
    raw2.169264529783277e-8. The4e-9 bank is achieved;3e-9,2e-9 and approximately
    1e-9 remain unmet. Goal active. Future qualified Phase2 submissions remain

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/polynomial_fusion_lazy_v1/build.py](code/polynomial_fusion_lazy_v1/build.py)
- [code/polynomial_fusion_lazy_v1/candidate.py](code/polynomial_fusion_lazy_v1/candidate.py)
- [code/polynomial_fusion_lazy_v1/confirmation.py](code/polynomial_fusion_lazy_v1/confirmation.py)
- [code/polynomial_fusion_lazy_v1/fused_helper.py](code/polynomial_fusion_lazy_v1/fused_helper.py)
- [code/polynomial_fusion_lazy_v1/panel.py](code/polynomial_fusion_lazy_v1/panel.py)
- [code/polynomial_fusion_lazy_v1/prepare_package.py](code/polynomial_fusion_lazy_v1/prepare_package.py)
- [code/polynomial_fusion_lazy_v1/qualify_boundary.py](code/polynomial_fusion_lazy_v1/qualify_boundary.py)
- [code/polynomial_fusion_lazy_v1/score_confirmation.py](code/polynomial_fusion_lazy_v1/score_confirmation.py)
- [code/polynomial_fusion_lazy_v1/score_dev.py](code/polynomial_fusion_lazy_v1/score_dev.py)
- [code/polynomial_fusion_lazy_v1/score_panel.py](code/polynomial_fusion_lazy_v1/score_panel.py)
- [code/polynomial_fusion_lazy_v1/unit.py](code/polynomial_fusion_lazy_v1/unit.py)
- [code/lazy_rotation_objective_v1/candidate.py](code/lazy_rotation_objective_v1/candidate.py)

<details>
<summary>Additional programs and dependencies (2)</summary>

- [code/run_guarded.py](code/run_guarded.py)
- [code/weight5_reuse_v1/native.py](code/weight5_reuse_v1/native.py)

</details>

## Evidence

- [NEGATIVE_RESULT.md#L3](NEGATIVE_RESULT.md#L3)
- [NEGATIVE_RESULT.md#L4](NEGATIVE_RESULT.md#L4)
- [NEGATIVE_RESULT.md#L5](NEGATIVE_RESULT.md#L5)
- [NEGATIVE_RESULT.md#L8](NEGATIVE_RESULT.md#L8)
- [NEGATIVE_RESULT.md#L11](NEGATIVE_RESULT.md#L11)
- [REEXAMINATION_INDEX_20260919.md#L1394](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L1394)
- [REEXAMINATION_INDEX_20260919.md#L1399](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L1399)

<details>
<summary>All result files and records (5)</summary>

- [NEGATIVE_RESULT.md](NEGATIVE_RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result
- [dev_score.json](dev_score.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-19-vectorized-pair-polynomial-fusion-saves-1819768776-flops-but-misses) · [Copy provenance](../../../PROVENANCE.json)

