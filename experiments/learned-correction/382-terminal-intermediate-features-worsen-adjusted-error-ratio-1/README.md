# 382 terminal intermediate features worsen adjusted error: ratio 1.0253749784, every fold loses

2026-09-19 · experiment · rejected · measured

Encoder inputs and hidden activations were projected away from the existing terminal affine span before fitting the added representation.

Recorded measurement:

    The predeclared20-network, five-fold information test completed and failed.
    |Unchanged Read5|3.989137621921596e-9|
    |Affine [1,H15] control|3.974845830335309e-9|
    |Intermediate-feature extension|4.090361903053607e-9|
    Extended/unchanged1.0253749784 (+2.53750%); extended/control1.0290617744
    (+2.90618%). The extension is worse in ALL five folds. The required gates
    were <=.95 of unchanged, <=.97 of control, and4/5 fold wins against both.
    The382 extra channels are the actual final encoder input126 and its two
    SiLU128 intermediates, before the final64-channel compression. Their
    TRAIN-only weighted SVD projection removes the existing [1,H15] span.
    Every fold retains control rank65 and369 residual columns; maximum
    normalized training cross-correlation is2.33344e-13. Thus duplicate-column
    to maximum absolute2.665e-15. These certify the fitting algebra, not efficacy.
    All20 complete predictions, existing intermediate arrays and own-cold bills
    were sealed before the new target access. Original1B FP32 mc_post means
    weight hashes. All OOF served FP32 predictions were sealed before aggregate
    Its1B TRAIN20 score must not replace Read5's separate4M-NC14 panel result.
    the original gnn_step bytes. Qualification35/100/35 preserves every parent
    The score includes1MF/3MF readout reserves beyond each own-cold parent for
    Resources:22 full parent calls,8,922,656,554,727 metered parent FLOPs;
    one numerical group at a time. Peak monitored process-tree RSS6,403,293,184B,
    fit took2.06s with483.6MB maximum self-reported RSS. Saved research artifacts
    occupy about43.1MB; no downloads or large covariance histories were written.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/terminal_representation_v1/capture.py](code/terminal_representation_v1/capture.py)
- [code/terminal_representation_v1/capture_all.py](code/terminal_representation_v1/capture_all.py)
- [code/terminal_representation_v1/certify.py](code/terminal_representation_v1/certify.py)
- [code/terminal_representation_v1/information.py](code/terminal_representation_v1/information.py)
- [code/read5_packet_v1/candidate.py](code/read5_packet_v1/candidate.py)
- [code/run_guarded.py](code/run_guarded.py)

## Evidence

- [NEGATIVE_RESULT.md#L3](NEGATIVE_RESULT.md#L3)
- [NEGATIVE_RESULT.md#L9](NEGATIVE_RESULT.md#L9)
- [NEGATIVE_RESULT.md#L10](NEGATIVE_RESULT.md#L10)
- [NEGATIVE_RESULT.md#L11](NEGATIVE_RESULT.md#L11)

<details>
<summary>All result files and records (3)</summary>

- [NEGATIVE_RESULT.md](NEGATIVE_RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-19-382-terminal-intermediate-features-worsen-adjusted-error-ratio-1) · [Copy provenance](../../../PROVENANCE.json)

