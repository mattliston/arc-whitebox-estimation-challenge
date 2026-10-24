# Single-submission tool records one archive and polls its evaluation receipt

2026-10-19T03:06:01Z · tool · adopted · measured

Every submission in this window was sent with fire_once.py (tag, archive path, note) after the three mandatory gates (official setup5 with its start envelope, stage A einsum-signature audit of the archive, prefire check) and polled every 150 s by poll_eval.sh into fire/<tag>/ (receipt.json, response.json, eval_<id>.json, poll.log); a paired-telemetry script compares two evaluation records network by network.

## Programs

- [code/fire_once.py](code/fire_once.py)
- [code/poll_eval.sh](code/poll_eval.sh)
- [code/paired_telemetry_v1.py](code/paired_telemetry_v1.py)
- [code/prefire_check_v1.py](code/prefire_check_v1.py)
- [code/compare_evals.py](code/compare_evals.py)
- [code/read_eval_summary_astra_v1.py](code/read_eval_summary_astra_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5897](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5897)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5918](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5918)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5893](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5893)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-10-19-single-submission-tool-records-one-archive-polls-evaluation-receipt) · [Copy provenance](../../PROVENANCE.json)

