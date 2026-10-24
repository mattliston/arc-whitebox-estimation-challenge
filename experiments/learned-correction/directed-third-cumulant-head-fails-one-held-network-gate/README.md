# Directed third-cumulant head fails one held-network gate: next-mean ratio 0.916243773 versus 0.90

2026-09-21 · experiment · rejected · measured

A fixed 24-epoch model was evaluated on weighted plane, covariance-response and next-mean-response losses; its symmetric surrogate was later distinguished from the actual directed consumer.

Recorded measurement:

    The single24-epoch model is complete. At least one of the three required
    | 14 | 0.848462749 | 0.714422508 | 0.893141095 |
    | 71 | 0.856459586 | 0.714472336 | 0.916243773 |
    Each held-out network had to meet<=.85,<=.85,<=.90 respectively. Network14
    passes all three. Network71 fails its weighted-third and next-mean gates.
    These are frozen symmetric-surrogate response losses against public100M
    asymmetric and active-floor fixtures, then measures all56 captured states.
    Maximum covariance asymmetry is6.48097397401e-05.
    3.85716902862e-10 and
    2.69512382411e-08, respectively.
    third-plane loss. That changes the first metric by about.133/.134 because
    weight and its gate stay authoritative. Network71's next-mean ratio still
    exceeds.90 under the exact directed response.
    The model is a directed32-hidden tanh head using140 causal node features,
    It predicts only offdiagonal pre-K21 corrections at layers8..14. The
    TRAIN57,68,45,59,3,12; held-out14,71. All eight current-parent trajectories
    were frozen before new target admission. The three57 qualification calls
    match all16 output rows and complete bills exactly, and the observed
    and rounding-bound checks. The1B final-mean target fields stayed unopened.
    Late-layer sampling error is still present in the100M labels.
    passed. The1024-wide component measured1,189,302,272 weighted FLOPs for one
    seven-layer sum is8,325,115,904. It passed the15-billion component cap.
    All46 guarded groups retired:45 successful and the original import failure.
    The ten unchanged-parent calls consumed4,127,132,456,836
    6,438,207,488 bytes; maximum successful

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260919_v1/incoming_third_pair_head_v1/capture.py](code/codex_resume_20260919_v1/incoming_third_pair_head_v1/capture.py)
- [code/codex_resume_20260919_v1/incoming_third_pair_head_v1/capture_v2.py](code/codex_resume_20260919_v1/incoming_third_pair_head_v1/capture_v2.py)
- [code/codex_resume_20260919_v1/incoming_third_pair_head_v1/closeout.py](code/codex_resume_20260919_v1/incoming_third_pair_head_v1/closeout.py)
- [code/codex_resume_20260919_v1/incoming_third_pair_head_v1/common.py](code/codex_resume_20260919_v1/incoming_third_pair_head_v1/common.py)
- [code/codex_resume_20260919_v1/incoming_third_pair_head_v1/directed_response_audit.py](code/codex_resume_20260919_v1/incoming_third_pair_head_v1/directed_response_audit.py)
- [code/codex_resume_20260919_v1/incoming_third_pair_head_v1/download.py](code/codex_resume_20260919_v1/incoming_third_pair_head_v1/download.py)
- [code/codex_resume_20260919_v1/incoming_third_pair_head_v1/extract_weights.py](code/codex_resume_20260919_v1/incoming_third_pair_head_v1/extract_weights.py)
- [code/codex_resume_20260919_v1/incoming_third_pair_head_v1/freeze_inputs.py](code/codex_resume_20260919_v1/incoming_third_pair_head_v1/freeze_inputs.py)
- [code/codex_resume_20260919_v1/incoming_third_pair_head_v1/head.py](code/codex_resume_20260919_v1/incoming_third_pair_head_v1/head.py)
- [code/codex_resume_20260919_v1/incoming_third_pair_head_v1/judge.py](code/codex_resume_20260919_v1/incoming_third_pair_head_v1/judge.py)
- [code/codex_resume_20260919_v1/incoming_third_pair_head_v1/math_core.py](code/codex_resume_20260919_v1/incoming_third_pair_head_v1/math_core.py)
- [code/codex_resume_20260919_v1/incoming_third_pair_head_v1/prepare.py](code/codex_resume_20260919_v1/incoming_third_pair_head_v1/prepare.py)

<details>
<summary>Additional programs and dependencies (18)</summary>

- [code/codex_resume_20260919_v1/incoming_third_pair_head_v1/prepare_directed_audit.py](code/codex_resume_20260919_v1/incoming_third_pair_head_v1/prepare_directed_audit.py)
- [code/codex_resume_20260919_v1/incoming_third_pair_head_v1/prepare_fit.py](code/codex_resume_20260919_v1/incoming_third_pair_head_v1/prepare_fit.py)
- [code/codex_resume_20260919_v1/incoming_third_pair_head_v1/prepare_replay.py](code/codex_resume_20260919_v1/incoming_third_pair_head_v1/prepare_replay.py)
- [code/codex_resume_20260919_v1/incoming_third_pair_head_v1/prepare_runtime_repair.py](code/codex_resume_20260919_v1/incoming_third_pair_head_v1/prepare_runtime_repair.py)
- [code/codex_resume_20260919_v1/incoming_third_pair_head_v1/qualify_targets.py](code/codex_resume_20260919_v1/incoming_third_pair_head_v1/qualify_targets.py)
- [code/codex_resume_20260919_v1/incoming_third_pair_head_v1/replay_judgment.py](code/codex_resume_20260919_v1/incoming_third_pair_head_v1/replay_judgment.py)
- [code/codex_resume_20260919_v1/incoming_third_pair_head_v1/run_audits.py](code/codex_resume_20260919_v1/incoming_third_pair_head_v1/run_audits.py)
- [code/codex_resume_20260919_v1/incoming_third_pair_head_v1/run_fit.py](code/codex_resume_20260919_v1/incoming_third_pair_head_v1/run_fit.py)
- [code/codex_resume_20260919_v1/incoming_third_pair_head_v1/run_inputs.py](code/codex_resume_20260919_v1/incoming_third_pair_head_v1/run_inputs.py)
- [code/codex_resume_20260919_v1/incoming_third_pair_head_v1/run_inputs_v2.py](code/codex_resume_20260919_v1/incoming_third_pair_head_v1/run_inputs_v2.py)
- [code/codex_resume_20260919_v1/incoming_third_pair_head_v1/train.py](code/codex_resume_20260919_v1/incoming_third_pair_head_v1/train.py)
- [code/codex_resume_20260919_v1/incoming_third_pair_head_v1/unit.py](code/codex_resume_20260919_v1/incoming_third_pair_head_v1/unit.py)
- [code/codex_resume_20260919_v1/incoming_third_pair_head_v1/unit_v2.py](code/codex_resume_20260919_v1/incoming_third_pair_head_v1/unit_v2.py)
- [code/codex_resume_20260919_v1/incoming_third_pair_head_v1/weights_dispatch.py](code/codex_resume_20260919_v1/incoming_third_pair_head_v1/weights_dispatch.py)
- [code/astra_phase2_reexamination_v1/public_third_centering_v1.py](code/astra_phase2_reexamination_v1/public_third_centering_v1.py)
- [code/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py](code/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py)
- [code/codex_resume_20260919_v1/run_guarded.py](code/codex_resume_20260919_v1/run_guarded.py)
- [code/data_gate/target_blind_mini_v2.py](code/data_gate/target_blind_mini_v2.py)

</details>

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L11](RESULT.md#L11)
- [RESULT.md#L12](RESULT.md#L12)
- [RESULT.md#L14](RESULT.md#L14)
- [REEXAMINATION_INDEX_20260919.md#L525](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L525)
- [REEXAMINATION_INDEX_20260919.md#L533](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L533)

<details>
<summary>All result files and records (5)</summary>

- [RESULT.md](RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result
- [summary.json](summary.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-21-directed-third-cumulant-head-fails-one-held-network-gate) · [Copy provenance](../../../PROVENANCE.json)

