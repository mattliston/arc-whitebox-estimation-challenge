# Dynamic residual routing dropped: residual time optimised statically under a hard request-count fire gate

2026-10-16T14:32:33Z · decision · adopted · measured

After the #336506 losses the user directed the team to drop the residual governor (dynamic routing) and to optimise residual wall time statically like the earlier clean builds (public residual mean .32-.35 s). A hard fire gate was set on request counts: first network <= 43,800 and warm <= 42,700 requests on both the normal and the 1.3x-slow boundary, then reset at 14:44Z to warm <= 41,200 (median <= .32 s), first <= 42,700, projected worst wall <= 100 s (corrected to 106 s at 14:54Z after a re-fit on the estimator submitted as #336412 (S2tr)-family runs), setup <= 4.60 s. The user also ruled out leaderboard mining and any retraining or refits.

Recorded measurement:

    Gate v1: first <= 43,800, warm <= 42,700 requests (0.4 s cap / ~7.8 us per request / 1.2x slow margin); S2tr was 47,336 / 45,646 (over). Gate v2: warm <= 41,200, first <= 42,700, worst wall <= 110 then 100 s, setup <= 4.60 s. Re-fit on S2tr-family runs (336381/389/392/397/412, 500 nets): P(any fail) 0% at median .315, 1.3% .32, 2.9% .325, 6.4% .33, 40% .34; wall P(any > 120 s) ~0% at median <= 96.6 s, 2.9% at 100 -> worst <= 106 s.

## Programs

- [count_gates_v2.py](count_gates_v2.py)
- [count_gates_v4.py](count_gates_v4.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4236](../../TEAM_LOG_2026-10-01_to_10-22.md#L4236)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4240](../../TEAM_LOG_2026-10-01_to_10-22.md#L4240)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4249](../../TEAM_LOG_2026-10-01_to_10-22.md#L4249)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4251](../../TEAM_LOG_2026-10-01_to_10-22.md#L4251)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4260](../../TEAM_LOG_2026-10-01_to_10-22.md#L4260)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4271](../../TEAM_LOG_2026-10-01_to_10-22.md#L4271)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Residual governor on #336506 fired on no network (flops ratio <= x1.004): zero projection margin at the 0.40 s threshold](../2026-10-16-residual-governor-336506-fired-no-network-flops-ratio-x1/README.md)

[Research graph](../../../graph/index.html#node=2026-10-16-dynamic-residual-routing-dropped-residual-time-optimised-statically-under) · [Copy provenance](../../../PROVENANCE.json)

