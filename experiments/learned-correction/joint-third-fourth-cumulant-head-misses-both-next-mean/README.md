# Joint third/fourth-cumulant head misses both next-mean gates: ratios 0.917358747 and 0.931169619

2026-09-21 · experiment · rejected · measured

A new three-output model jointly predicted normalized off-diagonal third and fourth cumulant planes after 32 epochs and 1,344 updates.

Recorded measurement:

    The single model finished its prescribed 32 epochs and 1,344 updates. Both
    | MINI ordinal | Weighted T21 | Weighted T31 | Weighted T22 | Joint covariance | Joint next mean |
    | 14 | 0.721598902 | 0.794306777 | 0.853710691 | 0.737520421 | 0.917358747 |
    | 71 | 0.711302170 | 0.783309956 | 0.876363142 | 0.726110964 | 0.931169619 |
    0.90. The three plane ratios are auxiliary reports, not acceptance gates.
    gates pass, while next-mean ratios 0.917358747 and 0.931169619 fail. The
    differences are 1.97220580418e-11,
    2.85751245099e-14 and
    1.68189903192e-15, respectively.
    normalized offdiagonal K21, K31 and K22 corrections at layers 8 through 14.
    Its 140 causal node features include previous-parent H64 transported by W
    K22 is symmetrized. The exact directed response uses the same rho_ij for
    TRAIN MINI ordinals were 57, 68, 45, 59, 3 and 12. Validation ordinals 14
    and 71 had already been used by the previous experiment and are explicitly
    existing 100M public payloads were reused without downloads or sampling.
    Newly decoded fields were pre_m4, pre_M31 and pre_M22, alongside previously
    admitted preactivation moments. No 1B target field was decoded.
    fourth RMS was 1.00262565459, maximum entry
    5.60232540131; limits were 1.5 and 12.
    0.499999050051 FP32 ULPs (limit 2).
    The maximum late-layer FP32 rounding bound was
    0.0224672651498 of the missing-plane RMS
    (limit 0.05). These checks do not remove late-layer Monte Carlo noise or
    parameter ingress measured 1,343,427,584 weighted FLOPs; the conservative
    seven-layer sum was 9,403,993,088, below the 15-billion component cap. There

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/joint_incoming_moment_head_v1/build.py](code/joint_incoming_moment_head_v1/build.py)
- [code/joint_incoming_moment_head_v1/closeout.py](code/joint_incoming_moment_head_v1/closeout.py)
- [code/joint_incoming_moment_head_v1/common.py](code/joint_incoming_moment_head_v1/common.py)
- [code/joint_incoming_moment_head_v1/data.py](code/joint_incoming_moment_head_v1/data.py)
- [code/joint_incoming_moment_head_v1/head.py](code/joint_incoming_moment_head_v1/head.py)
- [code/joint_incoming_moment_head_v1/judge.py](code/joint_incoming_moment_head_v1/judge.py)
- [code/joint_incoming_moment_head_v1/math_core.py](code/joint_incoming_moment_head_v1/math_core.py)
- [code/joint_incoming_moment_head_v1/moments.py](code/joint_incoming_moment_head_v1/moments.py)
- [code/joint_incoming_moment_head_v1/prepare.py](code/joint_incoming_moment_head_v1/prepare.py)
- [code/joint_incoming_moment_head_v1/prepare_fit.py](code/joint_incoming_moment_head_v1/prepare_fit.py)
- [code/joint_incoming_moment_head_v1/prepare_replay.py](code/joint_incoming_moment_head_v1/prepare_replay.py)
- [code/joint_incoming_moment_head_v1/replay.py](code/joint_incoming_moment_head_v1/replay.py)

<details>
<summary>Additional programs and dependencies (11)</summary>

- [code/joint_incoming_moment_head_v1/run_fit.py](code/joint_incoming_moment_head_v1/run_fit.py)
- [code/joint_incoming_moment_head_v1/run_replay.py](code/joint_incoming_moment_head_v1/run_replay.py)
- [code/joint_incoming_moment_head_v1/run_unit.py](code/joint_incoming_moment_head_v1/run_unit.py)
- [code/joint_incoming_moment_head_v1/targets.py](code/joint_incoming_moment_head_v1/targets.py)
- [code/joint_incoming_moment_head_v1/train.py](code/joint_incoming_moment_head_v1/train.py)
- [code/joint_incoming_moment_head_v1/unit.py](code/joint_incoming_moment_head_v1/unit.py)
- [code/full_lower_covariance_edge_packed_v1/candidate.py](code/full_lower_covariance_edge_packed_v1/candidate.py)
- [code/incoming_third_pair_head_v1/common.py](code/incoming_third_pair_head_v1/common.py)
- [code/incoming_third_pair_head_v1/directed_response_audit.py](code/incoming_third_pair_head_v1/directed_response_audit.py)
- [code/incoming_third_pair_head_v1/head.py](code/incoming_third_pair_head_v1/head.py)
- [code/run_guarded.py](code/run_guarded.py)

</details>

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L8](RESULT.md#L8)
- [RESULT.md#L10](RESULT.md#L10)
- [RESULT.md#L11](RESULT.md#L11)
- [REEXAMINATION_INDEX_20260919.md#L516](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L516)
- [REEXAMINATION_INDEX_20260919.md#L523](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L523)

<details>
<summary>All result files and records (5)</summary>

- [RESULT.md](RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result
- [summary.json](summary.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-21-joint-third-fourth-cumulant-head-misses-both-next-mean) · [Copy provenance](../../../PROVENANCE.json)

