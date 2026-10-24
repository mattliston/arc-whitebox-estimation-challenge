# Final designated source removes all residual-meter references from three inactive function bodies

2026-10-21T22:26:11Z · decision · adopted · derived

Three inactive residual-routing functions were replaced with constant returns and their description neutralized. The exact edited source passed release gates before submission #337641, with the previous copies retained as provenance.

Recorded measurement:

    - 2026-10-21T22:26:11Z [fable-lead] Matt: "[quote removed]"; asked to remove the dead governor. hgfh_s135_c4s8e9_clean release+science: the three _fast_rescue_live_residual bodies replaced by 'return None' and the describing comment neutralized (35 diff lines; 0 meter references left; parse ok; behavior unchanged: never called). Pre-strip copies kept as *_before_govstrip.py. Gates on the stripped file CLEAR (setup5 PASS 3.36 s, stage A PASS, prefire gc-only): archive e0a79ccb4c52, tag fable_hgfh_s135_c4s8e9_clean_govstrip3_5a4c1540_v1 (two earlier gate attempts were refused by the Mac memory guard while Mac32s ran; Mac32 queue stopped). fire_s135_v3.sh armed: fires once when fresh12 <= -0.45 lands, or at 22:45Z on the Mac32+VM32 evidence.
    - 2026-10-21T22:29:46Z [fable-lead] FIRED #337641 = hgfh_s135_c4s8e9_clean governor-stripped (archive e0a79ccb4c52), pooled ~-0.06% vs #337619 (fresh12 weak at -0.384); 2 shots remain; grading poll on the VM.
    - 2026-10-21T23:55:03Z [fable-lead] DESIGNATION DONE by Matt: [instruction removed]) + #337619 (1.4005566e-9, XSG off) selected for the Phase 2 private re-evaluation. Archives verified from disk: XSG off in all three fired files, 0 governor refs in #337641, imports = flopscope/whestbench + stdlib gc/math/time/zlib/base64; gc block = interpreter plumbing, disclosed in the write-up draft (A0/A9). J2W late fires at 23:49Z: #337650 15/100 projecting ~1.49e-9 (paired +6.6% MSE at our util), #337651 dead (95 exceptions), 3 queued.

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L7579](../../TEAM_LOG_2026-10-01_to_10-22.md#L7579)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7580](../../TEAM_LOG_2026-10-01_to_10-22.md#L7580)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7594](../../TEAM_LOG_2026-10-01_to_10-22.md#L7594)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- shipped in: [Submission #337641: Final 961-column readout at strength 1.35 with residual-meter references removed](../../../submissions/337641/README.md)

[Research graph](../../../graph/index.html#node=2026-10-21-final-designated-source-removes-all-residual-meter-references-three) · [Copy provenance](../../../PROVENANCE.json)

