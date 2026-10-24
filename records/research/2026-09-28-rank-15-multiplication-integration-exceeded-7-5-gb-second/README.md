# Rank-15 multiplication integration exceeded 7.5 GB on the second network

2026-09-28 · incident · failed · measured

The first integration completed rank35 but hit the memory guard on rank100. The partial first-network score did not establish a complete-screen gain, and a separate memory-only blocked implementation followed.

Recorded measurement:

    Rank35 adjusted-1.37094%,bill-1.71088%,raw+.34585%;rank100 RSS7,507,116,032B at51.072641s;zero swap;no rank100 output.

## Programs

- [candidate.py](candidate.py)

## Evidence

- [HILL_CLIMB_before_hk_train8_closeout_v1.md#L35](../../../experiments/flop-accounting/base-factor-cache-36-setup-bindings-combined-eight-network/HILL_CLIMB_before_hk_train8_closeout_v1.md#L35)

<details>
<summary>All result files and records (4)</summary>

- [HILL_CLIMB_before_hk_train8_closeout_v1.md](../../../experiments/flop-accounting/base-factor-cache-36-setup-bindings-combined-eight-network/HILL_CLIMB_before_hk_train8_closeout_v1.md) · record
- [RESULT.md](RESULT.md) · record
- [build.json](build.json) · result
- [PROTOCOL.md](PROTOCOL.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-28-rank-15-multiplication-integration-exceeded-7-5-gb-second) · [Copy provenance](../../../PROVENANCE.json)

