# Passed kernel gate was not communicated, delaying the next evaluation about four hours

2026-09-25T15:30Z · incident · informational · measured

The candidate had passed its residual gate, but its all-pass notification never reached the build owner. Evaluation resumed when the missing handoff was identified.

Recorded measurement:

    - 2026-09-25 08:30 [engineer] E-74 K-SPLIT v4 ALL GATES PASS (RESULT.md top). S26 fragment ksplit_fragment_v4.py (9e63ad1fa92417f5); porter confirmed at 08:30 (the all-pass after the 04:06 residual PASS never went out: my miss, S26 idled ~4 h). v4 = v2's leaf-sum design + S26 setup literals (_PB_WALL_S 3.5, _PB_BATCHED ()) + prebind taken off S23's census (chunk keys in the porter's census-derived list; phase 3 = 22 leaf buffers). Gates on S24 + emulated S26 list + v4 + treuse vs S24+E-73: residual cold +0.005 s PASS (r100 +0.007, r37 +0.001), requests +877..+1,087; parity bill -0.76..-1.07% (4-net mean -0.93%), MSE ratio 0.997-1.002 (rounding-level); user CPU +5.15% (v2 pair), expected worst 73.7 s; setup 2.14-2.18 s quiet, 3.507 s under heavy load (<= 3.6); handles +718. v4w (wide, -1.08%, +~380 requests) shelved with group 5. Coverage study: ~15% of chunk keys are net-specific whatever the list (~0.16% bill/net in lazy binds).
    - 2026-09-25 08:31 lead: MATT BACK: "keep going - we're within a hair of the podium not the time to ease up". S24 2.3726e-9 vs marius 2.299e-9 = -3.1% needed.
      STALL FOUND: the team idled from 03:41 to ~08:28. The engineer's e74d (k-split v4) residual gate PASSED at 04:06 (cold +0.005 s), but the all-pass never reached the porter, so S26 never launched. Launched now.

## Programs

- [ksplit_fragment_v4.py](ksplit_fragment_v4.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L1001](../../TEAM_LOG_2026-09-22_to_09-26.md#L1001)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1002](../../TEAM_LOG_2026-09-22_to_09-26.md#L1002)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1003](../../TEAM_LOG_2026-09-22_to_09-26.md#L1003)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-passed-kernel-gate-not-communicated-delaying-next-evaluation-about) · [Copy provenance](../../../PROVENANCE.json)

