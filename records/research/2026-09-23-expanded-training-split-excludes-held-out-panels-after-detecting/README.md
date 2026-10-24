# Expanded training split excludes held-out panels after detecting three reused panel networks

2026-09-23T13:35Z · incident · informational · measured

A verified 641-network training split excluded the validation set and panel. Earlier training contained three of eight panel networks, so validation and virgin-network results remained the independent checks.

Recorded measurement:

    - 2026-09-23 06:35 analyst: TRAIN640 split READY: fable_analyst_v1/split_train640.json = 641 nets (all observer_fit ranks minus VALID40 and panel8; panel 150 is in VALID40): 157 kept TRAIN160 + 484 new. NOTE: TRAIN160 contains panel nets 35/100/200, so past refits (incl. S6/S6b params_epoch2) were 3/8 in-sample on panel8 (VALID40, virgin6 clean). All 681 nets verified (weights sha = manifest, truth688 mlp_id = manifest, finite); new nets = same distribution (He std .04419, final truth mean .557, KS .065-.115 < .125). 1 epoch = 641 x 48 s = 8.5 h (161 updates at batch 4 = 4x TRAIN160's). refit_s6b.py gained --split + truth688 (bit-identical on old 200); s6bcard unchanged; the physicist's driver needs the same 2 edits. RESULT.md sec 25.

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L233](../../TEAM_LOG_2026-09-22_to_09-26.md#L233)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log
- [split_train640.json](split_train640.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-expanded-training-split-excludes-held-out-panels-after-detecting) · [Copy provenance](../../../PROVENANCE.json)

