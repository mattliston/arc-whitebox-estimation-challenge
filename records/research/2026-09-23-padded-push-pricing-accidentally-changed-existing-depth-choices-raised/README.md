# Padded-push pricing accidentally changed existing depth choices and raised metered work

2026-09-23T13:38Z · incident · failed · measured

A shared price parameter changed ordinary push decisions as well as padded-depth choices. Separate prices preserved existing decisions and isolated the intended padding tradeoff.

Recorded measurement:

    - 2026-09-23 07:15 [engineer] CORRECTION S7-lite depth fragments: depthrule_fragment_push5p6_mu06/_mu07.py (sent 06:05) were WRONG for S7-lite. In the v3 rule one MU prices every push decision, so MU 0.6/0.7 also dropped 18 aligned S6b D6 pushes per net to D5: as built they RAISE the bill vs S6b (+0.190% / +0.297% per net, model compute -2.35 / -3.00 s). My quoted -0.40% / -0.28% were the padded moves alone. engineer2 told to stop V06/V07. Fixed: make_depthrule_fragment_v3b.py -> depthrule_fragment_push5p6pad_mu06.py / _mu07.py keep every push5 (S6b) decision at MU 0.4 and add a padded D6 only where it beats that choice at MU_PAD. Verified over all 227 profiled product shapes: only D5->padded-D6 changes (13 classes at 0.6, 7 at 0.7). Per net vs S6b: pad07 3.0 moves, bill -0.277%, model +1.35 s (~+1.8% CPU); pad06 4.8 moves, -0.384%, +2.00 s (~+2.6%).
    - 2026-09-23 07:30 [engineer] S7-lite MU 0.7 end-to-end: fable_engineer_v1/candS7lite_pad07_cb3_mem.py (15d55e77) = S6b recipe + depthrule_fragment_push5p6pad_mu07.py + --caps --caps2 --batch --bindhold --coldbind3 + mem,yright,arena,hold2. In-process vs S6b+coldbind3: util r35 cold -0.084%, r100 -0.206% (model -0.10/-0.27%; padded shapes bind new frames); final MSE -0.043% / -0.110% (rounding level); handles flat. Vs S6b: about -1.9% cold, -0.6% warm (r100). Server CPU (model +1.35 s/net, ~+1.8%) is engineer2's gate.
    - 2026-09-23 06:38 [engineer] TIMESTAMP CORRECTION (my estimates ran fast; the entries themselves stay unedited): the entries stamped 06:25, 06:55, 07:15, 07:30 and 08:05 were written at about 06:04, 06:15, 06:19, 06:23 and 06:37 (anchored to the artifacts' file times). From now on every stamp comes from date +%H:%M.
    2026-09-23 06:48 [engineer2] S7-LITE V07 (S6b + push5p6 MU 0.7 + caps2 + bindhold + coldbind2 + hold2) meets the lead's numbers (paired 20 nets: wall projection -4.8 s, residual -3.6 ms, in-process adjusted -0.40%) but for the WRONG reason: util +0.045% over 20 nets and server CPU -4.2%. Bisection: push5p6 MU 0.7 ALONE raises util vs S6b (+0.33 cold / +0.24 / +0.01 / +0.03 / +0.45% warm repeat), because the fragment's single MU also re-decides push5's 4-vs-5-level pushes (fewer levels at 0.7); coldbind2 composes additively; caps2/bindhold/hold2 exactly bill-neutral. V06 shares the flaw (not gated). Need a fragment with MU 0.4 for 4-vs-5 and a separate price on the padded-D6 option only (asked engineer).

## Programs

- [make_depthrule_fragment_v3b.py](make_depthrule_fragment_v3b.py)
- [depthrule_fragment_push5p6pad_mu06.py](depthrule_fragment_push5p6pad_mu06.py)
- [candS7lite_pad07_cb3_mem.py](candS7lite_pad07_cb3_mem.py)
- [depthrule_fragment_push5p6pad_mu07.py](depthrule_fragment_push5p6pad_mu07.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L230](../../TEAM_LOG_2026-09-22_to_09-26.md#L230)
- [TEAM_LOG_2026-09-22_to_09-26.md#L231](../../TEAM_LOG_2026-09-22_to_09-26.md#L231)
- [TEAM_LOG_2026-09-22_to_09-26.md#L235](../../TEAM_LOG_2026-09-22_to_09-26.md#L235)
- [TEAM_LOG_2026-09-22_to_09-26.md#L237](../../TEAM_LOG_2026-09-22_to_09-26.md#L237)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-padded-push-pricing-accidentally-changed-existing-depth-choices-raised) · [Copy provenance](../../../PROVENANCE.json)

