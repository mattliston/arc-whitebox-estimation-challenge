# Scatter-for-gather disabled in every subsequent submission after fair-accounting review

2026-10-21T20:07:14Z · decision · adopted · derived

The source review found nine scatter-for-gather sites with a lower charged cost than direct gather. All subsequent submitted sources disabled this routing and accepted its measured additional FLOPs; the review otherwise found the listed imports and data-dependent arithmetic within the intended interface.

Recorded measurement:

    - 2026-10-21T20:07:14Z [fable-lead] RULES AUDIT (Matt pasted the official rules; file = hg_r0p03_s15_cap3 = #337614, same lineage as #337565): imports flopscope/flopscope.numpy/whestbench + stdlib base64/zlib/gc/math/time only; no numpy/scipy/ctypes/threads/subprocess/asyncio; no exec/eval/compile/codegen; no assignment into flopscope namespaces (no monkeypatching); Python arithmetic limited to scalar constants and the _np_tables coefficient tables computed at import (network-independent); seeds fixed (_LR_SEED); limits: setup 3.4 s/5, wall 111/120, residual .37/.40, RSS ~3 GB/8. ONE fair-accounting exposure: XSG (scatter-for-gather at 9 sites, _XSG_ON set; -0.077% bill, byte-identical) = a cost-model-priced path chosen for price (sec 5.5 example). Prepared tails/xsg_off.py (_XSG_ON = set()) for clean twins; recommendation to Matt: designate XSG-free files.
    - 2026-10-21T20:10:27Z [fable-lead] Matt: 'fire the clean twin as our 7th submission'. Built hg_r0p03_s15_cap3_clean = #337614's file + tails/xsg_off.py (_XSG_ON = set(); diff = name comment + that line); gates_fire_clean.sh queued behind the running Mac32 window: release_min gates, then fire_once ONCE if GATES CLEAR with only the gc FAIL; serial Mac32 script paused meanwhile.
    - 2026-10-21T20:12:41Z [fable-lead] FIRED #337617 = hg_r0p03_s15_cap3_clean (XSG off; archive 4d8a697f7e0e) as the 7th of the final ten (3 remain). RULE (Matt): all future stacks clean (tails/xsg_off.py appended). Serial Mac32/gates script resumed.
    - 2026-10-21T20:29:18Z [fable-lead] Mac32 hg_r0p03_s15_cap4s8e9 vs 49bd: all32 -0.412% held25 -0.526% raw -0.617% bill +0.206% -> vs #337565 -0.31/-0.13; pooled with VM32 (-0.28/-0.23) and fresh (-0.17): ~-0.26% all / -0.18% held -> ~1.400e-9 XSG-on, ~1.4012 clean. Matt: XSG OFF for ALL future submissions (reaffirmed). Built + launched hgfh_c4s8_k4lc105_clean (hg+fh seal x1.5 + cap4 + 8e9 + _K4L_C x1.05 + XSG off): VM32+fresh running; Mac32+gates queued (k4lc_mac32_gates.sh) behind the clean twins.
    - 2026-10-21T21:52:11Z [fable-lead] Mac32 hgfh_fh10_c4s8e9_clean = exact twin of #337619's clean file (block factor 1.0 = no-op): all32 -0.345% held25 -0.446% raw -0.632% bill +0.289% vs 49bd -> the clean Mac32 REFERENCE for #337619 (XSG-on twin -0.423/-0.524, bill +0.211; XSG-off costs +0.078 bill). Candidates must beat all32 -0.345 / held25 -0.446 on Mac32 and VM32 -0.542/-0.600 (fh10 VM32).

## Programs

- [xsg_off.py](xsg_off.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L7538](../../TEAM_LOG_2026-10-01_to_10-22.md#L7538)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7539](../../TEAM_LOG_2026-10-01_to_10-22.md#L7539)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7540](../../TEAM_LOG_2026-10-01_to_10-22.md#L7540)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7547](../../TEAM_LOG_2026-10-01_to_10-22.md#L7547)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7570](../../TEAM_LOG_2026-10-01_to_10-22.md#L7570)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- shipped in: [Submission #337617: Hidden-times-gate readout and cap 3 with direct gather](../../../submissions/337617/README.md)
- shipped in: [Submission #337619: 961-column on-policy readout, grouped-read cap 4 and 8e9 size threshold](../../../submissions/337619/README.md)
- shipped in: [Submission #337641: Final 961-column readout at strength 1.35 with residual-meter references removed](../../../submissions/337641/README.md)

[Research graph](../../../graph/index.html#node=2026-10-21-scatter-gather-disabled-every-subsequent-submission-after-fair-accounting) · [Copy provenance](../../../PROVENANCE.json)

