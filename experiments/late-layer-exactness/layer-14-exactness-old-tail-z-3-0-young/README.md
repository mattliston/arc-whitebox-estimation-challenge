# Layer-14 exactness (old tail z 3.0 + young crop z 3.0): raw -0.352%, bill +0.270%, adjusted -0.079% on 28 ranks

2026-10-19T13:22:30Z · experiment · adopted · measured

Exactness restored at the layer-14 push: the old-tail positive-side threshold (OTT) raised from 2.5 to 3.0 at layer 14 only (raw -0.151% for bill +0.166%; 3.5 no better; adding layer 13 adds nothing) and the young push's dead-row crop threshold (YNT) raised from 2.5 to 3.0 at layer 14 only, including the fusion prepare site (raw -0.234% for bill +0.186%; 3.5 flat). The two pieces add up to the l14x stack, raw -0.352% for bill +0.270% vs the row-cut reference (about -0.15% vs the #337090 physics), reproducible across its YNT 3.5 twin but raw-driven at 1.5 se with a wide per-rank split; a dead-cut 3.5 piece at layer 14 inside it is a no-op.

Recorded measurement:

    ott14_30: raw -0.151% (se ~.07, 12/28 lower), bill +0.166% vs stk1 (~+0.09% vs #337090); ott14_35: raw -0.193% (se .08, 11/28), bill +0.266%; ott1314_30: raw -0.165%, bill +0.276% (adjusted +0.112%); ynt14_30: raw -0.234% (se .10, 13/28), bill +0.186%, held-out 21 adjusted -0.145% (raw -0.333%); ynt14_35: raw -0.236%, bill +0.196%; l14x 803f6602: adjusted -0.079% (se .107, 14/28 lower) = raw -0.352% + bill +0.270%, held-out 21 -0.145% (raw -0.413%), per-rank split -1.4%..+1.0%; l14x35: raw -0.354%, bill +0.280% (adjusted -0.072%, 15/28); l14x3 (with deadcut 3.5 at L14): identical to l14x

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_ott14_30_sci_v1.py](code/candidate_ott14_30_sci_v1.py)
- [code/candidate_ott14_35_sci_v1.py](code/candidate_ott14_35_sci_v1.py)
- [code/candidate_ott1314_30_sci_v1.py](code/candidate_ott1314_30_sci_v1.py)
- [code/candidate_ynt14_30_sci_v1.py](code/candidate_ynt14_30_sci_v1.py)
- [code/candidate_ynt14_35_sci_v1.py](code/candidate_ynt14_35_sci_v1.py)
- [code/candidate_l14x_sci_v1.py](code/candidate_l14x_sci_v1.py)
- [code/candidate_l14x35_sci_v1.py](code/candidate_l14x35_sci_v1.py)
- [code/candidate_l14x3_sci_v1.py](code/candidate_l14x3_sci_v1.py)
- [code/candidate_rel_l14x_v1.py](code/candidate_rel_l14x_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5945](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5945)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5947](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5947)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5948](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5948)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5949](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5949)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5950](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5950)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5951](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5951)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5953](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5953)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5954](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5954)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5958](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5958)

<details>
<summary>All result files and records (2)</summary>

- [RESULT32_v1.json](RESULT32_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- shipped in: [Submission #337126: layer-14 old-tail and young-crop exactness z 3.0 on #337090, graded 1.4332116e-9 (-0.100%)](../../../submissions/337126/README.md)

[Research graph](../../../graph/index.html#node=2026-10-19-layer-14-exactness-old-tail-z-3-0-young) · [Copy provenance](../../../PROVENANCE.json)

