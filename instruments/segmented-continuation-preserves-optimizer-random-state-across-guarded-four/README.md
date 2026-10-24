# Segmented continuation preserves optimizer and random state across guarded four-update checkpoints

2026-09-30 · tool · informational · measured

The fixed80-update continuation controller ran20 serial four-update segments and saved parameters, Adam state, random state and global step. Exact optimizer reload and per-segment data hashes supported a single final endpoint without intermediate accuracy selection.

Recorded measurement:

    80updates,20guards,batch4,two TRAIN160 passes;each guard600s/7.5GB/zero added swap;all20 guards completed.

## Programs

- [code/controller.py](code/controller.py)
- [code/train_segment.py](code/train_segment.py)
- [code/native_parameters.py](code/native_parameters.py)

## Evidence

- [HILL_CLIMB_CURRENT.md#L442](../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L442)
- [HILL_CLIMB_CURRENT.md#L385](../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L385)

<details>
<summary>All result files and records (5)</summary>

- [HILL_CLIMB_CURRENT.md](../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md) · record
- [protocol.json](../../experiments/learned-correction/eighty-continuation-updates-unfrozen-terminal-head-fail-independent-native/protocol.json) · result
- [training_complete.json](training_complete.json) · result
- [controller_status.json](controller_status.json) · result
- [recipe_evidence.json](../../records/research/2026-09-30-historical-continuation-trained-terminal-head-short-later-trials-had/recipe_evidence.json) · result

</details>

[Research graph](../../graph/index.html#node=2026-09-30-segmented-continuation-preserves-optimizer-random-state-across-guarded-four) · [Copy provenance](../../PROVENANCE.json)

