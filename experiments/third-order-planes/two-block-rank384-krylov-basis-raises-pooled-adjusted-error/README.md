# Two-block rank384 Krylov basis raises pooled adjusted error to 1.041615262270436 of the parent

2026-09-19 · experiment · rejected · measured

Two weighted-Gram blocks of 192 columns were jointly orthogonalized at rank384 while the age and physical factors remained fixed.

Recorded measurement:

    # Fixed two-block Krylov384: closed negative
    The192+192 basis retains two weighted-Gram Krylov blocks and a final
    joint QR at rank384. Only the last shared-push basis block and inert
    helpers changed; age4, original entry metric, physical factors, Lazy
    | 35 | 1.870602849007794e-08 | 1.93089673284954e-08 | 1.0322323276 | 1.05057791591 |
    | 100 | 1.842453417527493e-08 | 1.869881698217136e-08 | 1.01488682451 | 1.03257950602 |
    Pooled adjusted ratio: **1.041615262270436**.
    <=.99 gate fails. These are two reused ordinary1B development
    Rank35 complete cold FLOPs: 398080917697 -> 405155903081 (+1.777273%).
    Rank100 complete cold FLOPs: 400893063472 -> 407881895254 (+1.743316%).
    Four actual-kernel algebra cases pass against explicit FP64 dense Grams:
    joining-only, joining plus old core, an FP32-rounded rank32 construction,
    and an exactly zero operator. Rounding means the rank32 construction is
    not claimed to have exact algebraic rank32. Zero exercises exact degeneracy.
    Maximum relative operator error: 4.564946173133269e-06.
    Maximum normalized orthogonality residual: 2.708021508247332e-08.
    Krylov-block inclusion, input bytes, finite FP32 output, six malformed-input
    was not initialized. That exit1 and source are preserved; additive
    unit_v2 fixes only arena setup/result location, with candidate and gates
    Four full predictions pass exact own repeat, warm/cold100 equality,
    first5 parent-layer bytes, finite nonnegative outputs and cache retirement.
    Total full-call charge: 1593873054858 FLOPs.
    Peak RSS including self-reports: 6407585792 bytes; zero added swap.
    General method background: [Musco and Musco2015](https://arxiv.org/abs/1504.05477).
    FP32 nonlinear estimator or its endpoint score.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/block_krylov384_v1/build.py](code/block_krylov384_v1/build.py)
- [code/block_krylov384_v1/candidate.py](code/block_krylov384_v1/candidate.py)
- [code/block_krylov384_v1/finalize.py](code/block_krylov384_v1/finalize.py)
- [code/block_krylov384_v1/krylov_fnp.py](code/block_krylov384_v1/krylov_fnp.py)
- [code/block_krylov384_v1/score_dev.py](code/block_krylov384_v1/score_dev.py)
- [code/block_krylov384_v1/score_dev_v2.py](code/block_krylov384_v1/score_dev_v2.py)
- [code/block_krylov384_v1/unit.py](code/block_krylov384_v1/unit.py)
- [code/block_krylov384_v1/unit_v2.py](code/block_krylov384_v1/unit_v2.py)
- [code/lazy_rotation_objective_v1/candidate.py](code/lazy_rotation_objective_v1/candidate.py)
- [code/weight5_reuse_v1/native.py](code/weight5_reuse_v1/native.py)

## Evidence

- [NEGATIVE_RESULT.md#L3](NEGATIVE_RESULT.md#L3)
- [NEGATIVE_RESULT.md#L1](NEGATIVE_RESULT.md#L1)
- [NEGATIVE_RESULT.md#L4](NEGATIVE_RESULT.md#L4)
- [NEGATIVE_RESULT.md#L5](NEGATIVE_RESULT.md#L5)
- [REEXAMINATION_INDEX_20260919.md#L1381](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L1381)
- [REEXAMINATION_INDEX_20260919.md#L1383](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L1383)

<details>
<summary>All result files and records (5)</summary>

- [NEGATIVE_RESULT.md](NEGATIVE_RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result
- [dev_score.json](dev_score.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-19-two-block-rank384-krylov-basis-raises-pooled-adjusted-error) · [Copy provenance](../../../PROVENANCE.json)

