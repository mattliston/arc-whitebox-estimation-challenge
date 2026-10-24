# Single scalar gain reverses inside both exact-copy and fourth-order stacks

2026-10-20T21:09:26Z · experiment · rejected · measured

A global scalar change improved the old reference by -.1216%, but actual combined tests harmed their respective parents by +.113620% and +.026192%. The scalar gain was therefore not available as an additive improvement.

Recorded measurement:

    GSK stacks complete vs337393: microKCcopyallGSK -.034981%all32/-.080635%held, but exactparent+.113620%all32; microKCk31GSK -.119560%all32/-.084128%held, but exactparent+.026192%. Both harm ownparents and are not bank candidates. A/GSK_STACKS_COMPLETE_VS_PARENTS_v1.json.
    GSKglobal1.0 (base.9) completed32: -.1216% SE.0622,21/32lower; held25-.1273%, billunchanged. New exactparent stacks at queuefront: micro_kclm_copyall_gskp_astra SCI4f7d3d7619e356840453b8b08642c8c099920d9760632a6361ed229335122d22 REL0c611de5624f8425b290d3c732d70c0e4394e6cf4c7114a4f995eaa27d7cf076; micro_kclm_k31age4_gskp_astra SCI219e9e76d355bc1bd86116e3a8042000206ba4cebfcbbe44b961e3eae1a41ba6 RELcd75b5df0624c260e6227c3aae8688353cd907cd9cf8eb4f4e6059c71047957d. A/GSKP_TWO_STACKS_BUILD_v1.json. Neither combined gain measured.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_micro_kclm_copyall_gskp_astra_sci_v1.py](code/candidate_micro_kclm_copyall_gskp_astra_sci_v1.py)
- [code/candidate_micro_kclm_k31age4_gskp_astra_sci_v1.py](code/candidate_micro_kclm_k31age4_gskp_astra_sci_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6312](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6312)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6270](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6270)

<details>
<summary>All result files and records (3)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [GSK_STACKS_COMPLETE_VS_PARENTS_v1.json](GSK_STACKS_COMPLETE_VS_PARENTS_v1.json) · result
- [GSKP_TWO_STACKS_BUILD_v1.json](GSKP_TWO_STACKS_BUILD_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-single-scalar-gain-reverses-inside-both-exact-copy-fourth) · [Copy provenance](../../../PROVENANCE.json)

