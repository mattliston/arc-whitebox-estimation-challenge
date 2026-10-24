# Residual margin loss on #337065 traced to slow-worker episodes (server kernel wall +7-11%), not to new lazy binds

2026-10-19T07:58:29Z · finding · informational · measured

The near-cap networks of #337065 were cold first-of-worker networks and a few warm ones whose residual rose 7-10% vs #337061. An own-key census of the file showed the column squeeze added no uncovered prebind keys (a refit saves only ~80-160 requests on the first network), while on the platform those networks' server kernel wall rose 7-11% with the residual/kernel ratio unchanged: slow workers, the known noise regime. The byte-identical own-list refit file 5d097ef1 was held as the release base for the next step, not fired alone.

Recorded measurement:

    idx 4 residual .3627 -> .3972, wall 107.8 -> 117.6 s; idx 1 .3634 -> .3769; idx 17 .3429 -> .3662; kernel wall idx 4 98.0 -> 107.0 s, idx 9 86.4 -> 96.2, idx 17 97.9 -> 105.5; residual/kernel median .00382 vs .00378; census 13 nets: old list first-net lazy-bind requests 3122 fit / 3920 hold, later nets 1516 / 1951; refit file: byte-identical on 7 ranks, flops -0.003..-0.033% (mean -0.014%), requests -24..-135/net, lazy tree binds -2..-4/net, archive 4c949d9eeda5

## Programs

- [candidate_rel_ab47_kc1536_pb2_v1.py](candidate_rel_ab47_kc1536_pb2_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5926](../../TEAM_LOG_2026-10-01_to_10-22.md#L5926)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5927](../../TEAM_LOG_2026-10-01_to_10-22.md#L5927)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5928](../../TEAM_LOG_2026-10-01_to_10-22.md#L5928)

<details>
<summary>All result files and records (4)</summary>

- [quick_fit.json](quick_fit.json) · result
- [quick_hold.json](quick_hold.json) · result
- [refit.log](refit.log) · log
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-19-residual-margin-loss-337065-traced-slow-worker-episodes-server) · [Copy provenance](../../../PROVENANCE.json)

