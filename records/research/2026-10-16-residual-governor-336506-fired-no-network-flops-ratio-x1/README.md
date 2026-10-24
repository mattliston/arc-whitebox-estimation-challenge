# Residual governor on #336506 fired on no network (flops ratio <= x1.004): zero projection margin at the 0.40 s threshold

2026-10-16T14:29:35Z · finding · informational · derived

Paired per network against #336412 (the estimator submitted as #336412 (S2tr), same 100 networks), the pubmaxapg2 run #336506 lost four public networks to the 0.4 s residual cap (MLP0 .402, MLP1 .423, MLP4 .424 on cold first networks, MLP12 .415 warm) while no network showed the +1.2-1.7% flops of an early FULL or EXACT routing switch, so the residual governor (a residual-time-triggered routing switch) never rescued them. The shim check showed the flopscope BudgetContext is open in the participant process so the live read works; the projection r(L)/FRAC[L] crosses the public FULL threshold .40 only at layer 3 and a back-loading of the residual accrual by 0.6-5.7% hides each failure through layer 12, which is what cold first-call lazy binds of the read grid and pushSM families do. Verdict: mis-projection with no margin, not an inert governor. A residue-code canary (per-network flops_used mod 65521 encoding the trigger layer) was built and left untested when the user dropped dynamic routing.

Recorded measurement:

    Median flops ratio 0.9965 vs #336412 on 15 networks, per-network excess -0.56..+0.68% (nets 4 +0.68%, 1 +0.41%); residual x1.043 median, failures x1.12-1.17; server overhead x1.03 median, x1.15-1.20 on MLP4/8/12/18; kernel x1.06; completed median .373 vs S2tr .356. Final: 11 failed (9 public), residual median x1.037, kernel x1.062. Back-loading needed to hide: 0.6 / 5.5 / 5.7 / 3.7% at .40 (3.1 / 7.9 / 8.1 / 6.1% at v4g .39). Canary cost <= 65,520 FLOPs and 2 requests per network.

## Programs

- [analyze_governor_live_v1.py](analyze_governor_live_v1.py)
- [test_canary_plain.py](test_canary_plain.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4215](../../TEAM_LOG_2026-10-01_to_10-22.md#L4215)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4217](../../TEAM_LOG_2026-10-01_to_10-22.md#L4217)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4219](../../TEAM_LOG_2026-10-01_to_10-22.md#L4219)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4221](../../TEAM_LOG_2026-10-01_to_10-22.md#L4221)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4224](../../TEAM_LOG_2026-10-01_to_10-22.md#L4224)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4228](../../TEAM_LOG_2026-10-01_to_10-22.md#L4228)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4229](../../TEAM_LOG_2026-10-01_to_10-22.md#L4229)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4238](../../TEAM_LOG_2026-10-01_to_10-22.md#L4238)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4269](../../TEAM_LOG_2026-10-01_to_10-22.md#L4269)

<details>
<summary>All result files and records (4)</summary>

- [eval_336506-71a8dfce.json](../../../submissions/336506/eval_336506-71a8dfce.json) · result
- [eval_336506_partial_1421Z.json](../../../submissions/336506/eval_336506_partial_1421Z.json) · result
- [fragment_govcanary_v1.py.txt](fragment_govcanary_v1.py.txt) · code
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- enables: [Dynamic residual routing dropped: residual time optimised statically under a hard request-count fire gate](../2026-10-16-dynamic-residual-routing-dropped-residual-time-optimised-statically-under/README.md)

[Research graph](../../../graph/index.html#node=2026-10-16-residual-governor-336506-fired-no-network-flops-ratio-x1) · [Copy provenance](../../../PROVENANCE.json)

