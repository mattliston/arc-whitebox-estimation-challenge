# Native approximation audit rejects aggressive positive-tail cuts and pair-mean coordinate repair

2026-10-10T21:48Z · experiment · rejected · measured

Beyond the three main native cuts, the audit of the submitted candidate generation 45 (S45) file on networks 651 / 378 priced partial variants: the pair-mean truncation exact at layers 10-14 only -0.78% for +1.31% metered FLOPs, at layer 15 only +0.06% for +0.24%; the old-tier positive tail exact +0.46% for +2.07%, the dead negative tail exact +1.39% for +1.40%; the positive tail at 2.0 instead of 2.5 +14.8% for -0.40% metered FLOPs (the 2.0-2.5 band matters); edge rosters at floor 0.05 +0.13% for -0.149% metered FLOPs and at 0.10 -0.224% metered FLOPs for an implied 0.19-0.23% (break-even); tails at the response-row edge 2.34 +0.64% for -0.143%. A probe of the pair-mean read rank (r_u) = 256 pair-mean rule found it drops only 0.066% / 0.025% of the pair means' energy and the best rotation would still drop 0.022% / 0.013%, so the truncation's 0.5-1% of MSE is not frame energy and has no cheap repair.

Recorded measurement:

    r_u exact L10-14 -0.56 / -0.98% (-0.78%) for +1.31%; L15 only +0.06% for +0.24%; positive tail exact +0.46% for +2.07%; dead tail exact +1.39% for +1.40%; positive tail at 2.0 +14.8% for -0.40%; edge rosters 0.05 +0.13% (implied 0.11%) for -0.149%, 0.10 -0.224% bill for 0.19-0.23% implied; tails at 2.34 +0.64% for -0.143%; r_u 256 drops 0.066% (L10) / 0.025% (L13) of energy

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a53_audit`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/probe_uvp.py](code/probe_uvp.py)
- [code/level_table.py](code/level_table.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L710](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L710)
- [TEAM_LOG_2026-10-01_to_10-22.md#L715](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L715)
- [TEAM_LOG_2026-10-01_to_10-22.md#L729](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L729)

<details>
<summary>All result files and records (2)</summary>

- [RESULT.md](../../arithmetic-kernels/half-zero-strip-kernel-odd-32-dimensions-lowers-charged/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-10-native-approximation-audit-rejects-aggressive-positive-tail-cuts-pair) · [Copy provenance](../../../PROVENANCE.json)

