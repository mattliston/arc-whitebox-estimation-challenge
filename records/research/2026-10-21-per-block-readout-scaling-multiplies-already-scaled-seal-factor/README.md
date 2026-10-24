# Per-block readout scaling multiplies an already scaled seal; factor 2 produced absolute strength 3

2026-10-21T21:46:57Z · incident · failed · measured

A block factor of one was a no-op because the loaded coefficients already had strength 1.5. Factor two therefore tested absolute strength 3 and was catastrophic; the corrected strength-one experiment was later retired before useful completion.

Recorded measurement:

    - 2026-10-21T21:46:57Z [fable-lead] PER-BLOCK STRENGTH mis-specified: the block factor multiplies the already x1.5-scaled seal, so fh10 = twin of #337619 (raw identical, no information) and fh20 = fh block at x3.0 absolute -> Mac32 +55% (catastrophic; killed). Proper test launched: hgfh_fh067_c4s8e9_clean (fh block x1.0 absolute, rest x1.5) VM32+fresh. Also running: hgfh_c5s8e9_clean (VM32+fresh), strength x1.35/x1.65 (VM32), ridge .05 hg+fh fit; teammate cap5+8e9 vs cap4+8e9 pair; idea_scout agent ideating.
    - 2026-10-21T21:41:31Z [fable-lead] hgfh_fh10_c4s8e9_clean (first x hidden block x1.0, rest x1.5) VM32 vs 49bd: all32 -0.542% held25 -0.600% raw -0.826% bill +0.287% = raw IDENTICAL to the #337619 twin (-0.826): the fh block's strength does not move the native score; per-block strength arm ~0 (fh20 VM32 21/32, Mac32 running). Board quiet (46 live, none < 1.45e-9).
    - 2026-10-21T21:52:11Z [fable-lead] Mac32 hgfh_fh10_c4s8e9_clean = exact twin of #337619's clean file (block factor 1.0 = no-op): all32 -0.345% held25 -0.446% raw -0.632% bill +0.289% vs 49bd -> the clean Mac32 REFERENCE for #337619 (XSG-on twin -0.423/-0.524, bill +0.211; XSG-off costs +0.078 bill). Candidates must beat all32 -0.345 / held25 -0.446 on Mac32 and VM32 -0.542/-0.600 (fh10 VM32).
    - 2026-10-21T22:12:36Z [fable-lead] Fire-critical prioritization: s135 fresh cohorts were starved (0 ranks) -> killed fh067 (VM32+fresh), r0p05 (VM32+fresh), s165 fresh; teammate told to retire k31r_late_up and skip K4L34. Box 67 GB available; s135 fresh parts admitted; dials at 11-20/32; s12/s105 VM32+fresh queued. Fire job fire_s135_v2 armed (fresh <= -0.45 -> fire once).

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L7568](../../TEAM_LOG_2026-10-01_to_10-22.md#L7568)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7567](../../TEAM_LOG_2026-10-01_to_10-22.md#L7567)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7570](../../TEAM_LOG_2026-10-01_to_10-22.md#L7570)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7577](../../TEAM_LOG_2026-10-01_to_10-22.md#L7577)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-per-block-readout-scaling-multiplies-already-scaled-seal-factor) · [Copy provenance](../../../PROVENANCE.json)

