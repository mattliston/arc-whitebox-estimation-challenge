# Native evaluation queue replaces dependent shell chains and fixes shell word splitting

2026-10-19T16:29:42Z · tool · adopted · measured

Chained judge cascades were replaced by a queue runner taking a lane count, a reference and tags: v2 died at start under macOS bash 3.2 with set -u on an empty array expansion, v3 removed set -u, v4 paired with the relaxed judge copy, v5 became file-driven. The first zsh launcher never started its judges because `set -- $spec` does not word-split in zsh, so shasum got 'path tag' as one file; it was relaunched from bash.

Recorded measurement:

    queue3 relaunched 09:08Z; four lanes ~1.1 ranks/min/lane

## Programs

- [code/run_queue34_v1.sh](code/run_queue34_v1.sh)
- [code/run_queue_v2.sh](code/run_queue_v2.sh)
- [code/run_queue_v3.sh](code/run_queue_v3.sh)
- [code/run_queue_v4.sh](code/run_queue_v4.sh)
- [code/run_queue_v5.sh](code/run_queue_v5.sh)
- [code/kill_tree_v1.sh](code/kill_tree_v1.sh)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5929](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5929)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5977](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5977)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5979](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5979)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5982](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5982)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6003](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6003)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-10-19-native-evaluation-queue-replaces-dependent-shell-chains-fixes-shell) · [Copy provenance](../../PROVENANCE.json)

