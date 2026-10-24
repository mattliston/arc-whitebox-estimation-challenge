# Tree request census identifies exact strided-operation fusion as the next residual reduction

2026-09-25T07:03Z · tool · informational · measured

Additions and subtractions dominate remote requests. Prebinding paired strided views during guarded setup permits exact operation fusion without imposing a cold prediction penalty.

Recorded measurement:

    - 2026-09-25 00:03 [engineer] E-71 REQUEST CENSUS of S23 (paper, nothing built; RESULT.md E-71): 31.7k requests/net warm (32.2k cold) + ~2.7k free trips; add 9.4k + subtract 7.9k = 55%. Tree kernel 19.4k (61%): 188 single products (S23's s1=0 unbatched C32@w), ~106 requests each. Priced: (1) level-op fusion (paired strided ops, 3->2 / 3->2 / 6->4 per level; bit-identical, bill 0) -3.1k (-9.7%); (2) reuse repeated A transforms -0.7k, bill down to -0.3%; (3) batch same-shape products -0.6k; (4) level-1 copies -0.6k; (5) physics body (engineer2) est. -1..-1.5k. Lane total -5.0k (-15.6%), with the body diet -19..-20%. Awaiting the lead's GO.
    - 2026-09-25 00:03 lead: E-71 REQUEST CENSUS (engineer, S23): r100 31.7k requests + 2.7k free trips; add/subtract 55%. The tree kernel is 61% (188 single products, ~106 requests each); the physics body 39%.
      Priced groups:
      - (1) level-op fusion via strided pair views: -3.1k (-9.7%), bit-identical, bill 0;
      - (2) forward-A transform reuse: -0.7k, bill -0.3% if reordered;
      - (3) same-shape batching: -0.6k;
      - (4) level-1 copies: -0.6k;
      - (5) body (engineer2): -1..-1.5k.
      Groups 1-4 total ~-15.6% requests (~-33 ms residual).
      GO group 1 with the pair views bound in setup() under the 3.0 s guard, not at first predict (that would add ~+50 ms to each worker's cold MLP). Gates: bit-identity, bill identity, request census, residual pair vs S23, setup wall. Engineer and engineer2 agree on ownership of the _ipt_* functions before editing.

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L777](../../TEAM_LOG_2026-09-22_to_09-26.md#L777)
- [TEAM_LOG_2026-09-22_to_09-26.md#L778](../../TEAM_LOG_2026-09-22_to_09-26.md#L778)
- [TEAM_LOG_2026-09-22_to_09-26.md#L779](../../TEAM_LOG_2026-09-22_to_09-26.md#L779)
- [TEAM_LOG_2026-09-22_to_09-26.md#L780](../../TEAM_LOG_2026-09-22_to_09-26.md#L780)
- [TEAM_LOG_2026-09-22_to_09-26.md#L781](../../TEAM_LOG_2026-09-22_to_09-26.md#L781)
- [TEAM_LOG_2026-09-22_to_09-26.md#L782](../../TEAM_LOG_2026-09-22_to_09-26.md#L782)
- [TEAM_LOG_2026-09-22_to_09-26.md#L783](../../TEAM_LOG_2026-09-22_to_09-26.md#L783)
- [TEAM_LOG_2026-09-22_to_09-26.md#L784](../../TEAM_LOG_2026-09-22_to_09-26.md#L784)
- [TEAM_LOG_2026-09-22_to_09-26.md#L785](../../TEAM_LOG_2026-09-22_to_09-26.md#L785)
- [TEAM_LOG_2026-09-22_to_09-26.md#L786](../../TEAM_LOG_2026-09-22_to_09-26.md#L786)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-tree-request-census-identifies-exact-strided-operation-fusion-next) · [Copy provenance](../../../PROVENANCE.json)

