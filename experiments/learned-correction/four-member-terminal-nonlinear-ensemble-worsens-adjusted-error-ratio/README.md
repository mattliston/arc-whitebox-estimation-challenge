# Four-member terminal nonlinear ensemble worsens adjusted error: ratio 1.036374342900148, 0/5 folds win

2026-09-19 · experiment · rejected · measured

Bootstrap averaging improved over individual learned members but failed to generalize against the unchanged and affine controls.

Recorded measurement:

    Adjusted ordinary1B-target development errors, including the declared readout
    | Unchanged Read5 |3.989137621921596e-09|
    | Same TRAIN-only affine control |3.974845830335309e-09|
    | Nonlinear ensemble |4.134239881657254e-09|
    | Mean of individual-member scores |4.246380487610553e-09|
    Ensemble/unchanged=1.036374342900148; ensemble/control=
    1.040100687706043. **0/5 folds win** against both baselines.
    The required ratios were<=.95 and<=.97, with4/5 wins. No CHECK access, full-data
    This closes this fixed4,097-parameter terminal nonlinear bagging test, not all
    unchanged estimator. All20 training objectives decrease: final normalized
    bootstrap losses range0.879335391 to0.923288178, while every outer
    FP64 predictions (maximum error5.56e-22).
    The existing Read5 terminal126-input captures and sigma*[1,H15] control are
    reused without another physical prediction. Five outer folds are ordinal%5
    on the fixed20 networks. Every normalizer, residual scale, affine coefficient
    and nonlinear projection comes only from that fold's16 TRAIN networks.
    Each of4 members has126->32->1 SiLU layers and its own presealed initialization
    and16-network bootstrap draw. The four-member mean is the only proposed arm;
    At each of200 full-batch updates, its correction is projected off the common
    held-out predictions. Model arithmetic is FP32, projection/loss reduction FP64,
    with the actual final nonnegative FP32 serve. AdamW/cosine/gradient cap and all
    200 updates were fixed in advance; exactly4000 updates/20 terminal payloads
    All20 initial arrays and bootstraps were sealed before this trainer opened
    original1B target files. Capture/target hashes and canonical weights match
    held-out CHECK4 truth or broader14 panel was opened.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/terminal_bagged_residual_v1/certify.py](code/terminal_bagged_residual_v1/certify.py)
- [code/terminal_bagged_residual_v1/model.py](code/terminal_bagged_residual_v1/model.py)
- [code/terminal_bagged_residual_v1/train.py](code/terminal_bagged_residual_v1/train.py)
- [code/read5_packet_v1/candidate.py](code/read5_packet_v1/candidate.py)
- [code/terminal_representation_v1/capture.py](code/terminal_representation_v1/capture.py)

## Evidence

- [NEGATIVE_RESULT.md#L3](NEGATIVE_RESULT.md#L3)
- [NEGATIVE_RESULT.md#L4](NEGATIVE_RESULT.md#L4)
- [NEGATIVE_RESULT.md#L9](NEGATIVE_RESULT.md#L9)
- [NEGATIVE_RESULT.md#L10](NEGATIVE_RESULT.md#L10)
- [NEGATIVE_RESULT.md#L11](NEGATIVE_RESULT.md#L11)

<details>
<summary>All result files and records (3)</summary>

- [NEGATIVE_RESULT.md](NEGATIVE_RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-19-four-member-terminal-nonlinear-ensemble-worsens-adjusted-error-ratio) · [Copy provenance](../../../PROVENANCE.json)

