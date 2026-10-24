# Deferring coefficient rotations saves about 1.9% cold FLOPs but fails the 1e-5 output-identity gate

2026-09-19 · experiment · rejected · measured

Four rotations were deferred without changing rank384, core updates or physical and learned consumers; the later separate objective study preserved this initial failed gate.

Recorded measurement:

    Phase 2 only. The exact preserved lazy-rotation fragment was ported to current
    Weight5 and executed on native35/100/35 plus independent cold100. Complete
    parent took 51.229 seconds at the real client. Weight5's measured local client
    sequence now takes 39.4935/37.5752/38.2081 seconds. That changed runtime premise
    `e96ee6b660a688f767abd2b8300883e52920eace477e35e7b5edaae759ac6e92`.
    `b7c21ac7be5cafa45cd73ac0407605a144a724d62ffc7e2044e1432f5166377c`.
    Weight5 is an exact byte prefix. Six reversible substitutions change only the
    last shared-state, push and read definitions. Rank384, age4, all QR/core updates,
    all reconstructed rows, physical/learned consumers, kernels, arena and Weight5
    cache ownership remain. Rotations are deferred only at layers8/10/12/14 and
    consolidated at9/11/13, with the last map reused at15. At most two ordered factor
    Dense rank384 tests pass 13 comparisons and ten rejection cases. Maximum
    relative Frobenius error is 6.524844351582119e-7 against the unchanged 2e-4
    | Call | Own complete FLOPs | Saved versus matching Weight5 call | Native seconds |
    | Cold35 | 398080917697 | 7466832638 | 29.62456625 |
    | Warm100 | 388883126034 | 7465448812 | 27.72145717 |
    | Repeat35 | 377556934779 | 6337803914 | 29.18781596 |
    | Cold100 | 400893063472 | 7682859772 | 28.00971583 |
    Own cold FLOP ratios are 0.9815882775 and 0.9811960046. Context history affects
    All four outputs are finite, nonnegative FP32 with shape16x1024. Own repeat and
    warm/cold100 are byte exact. The first eight served layers match Weight5 exactly.
    All 15 Weight5 cache creators and39 hits remain, and caches retire each call.
    | Network | Largest all-layer absolute difference | Final-layer RMS difference | Values exceeding 1e-5 |
    | 35 | 2.6047229766845703e-5 | 5.091128473248542e-6 | 115 |
    | 100 | 2.2977590560913086e-5 | 4.8973679118009544e-6 | 104 |

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260919_v1/lazy_rotation_v1/build.py](code/codex_resume_20260919_v1/lazy_rotation_v1/build.py)
- [code/codex_resume_20260919_v1/lazy_rotation_v1/candidate.py](code/codex_resume_20260919_v1/lazy_rotation_v1/candidate.py)
- [code/codex_resume_20260919_v1/lazy_rotation_v1/diagnose_fidelity.py](code/codex_resume_20260919_v1/lazy_rotation_v1/diagnose_fidelity.py)
- [code/codex_resume_20260919_v1/lazy_rotation_v1/kernel.py](code/codex_resume_20260919_v1/lazy_rotation_v1/kernel.py)
- [code/codex_resume_20260919_v1/lazy_rotation_v1/score_dev.py](code/codex_resume_20260919_v1/lazy_rotation_v1/score_dev.py)
- [code/codex_resume_20260919_v1/lazy_rotation_v1/unit.py](code/codex_resume_20260919_v1/lazy_rotation_v1/unit.py)
- [code/astra_phase2_reexamination_v1/current_comp_shared_lazy_rotation_v1/fnp.py](code/astra_phase2_reexamination_v1/current_comp_shared_lazy_rotation_v1/fnp.py)
- [code/codex_resume_20260919_v1/weight5_reuse_v1/candidate.py](code/codex_resume_20260919_v1/weight5_reuse_v1/candidate.py)

## Evidence

- [NEGATIVE_RESULT.md#L3](NEGATIVE_RESULT.md#L3)
- [NEGATIVE_RESULT.md#L4](NEGATIVE_RESULT.md#L4)
- [NEGATIVE_RESULT.md#L12](NEGATIVE_RESULT.md#L12)
- [NEGATIVE_RESULT.md#L13](NEGATIVE_RESULT.md#L13)

<details>
<summary>All result files and records (3)</summary>

- [NEGATIVE_RESULT.md](NEGATIVE_RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-19-deferring-coefficient-rotations-saves-about-1-9-cold-flops) · [Copy provenance](../../../PROVENANCE.json)

