# Single-attempt submission uploader records one qualified archive, receipt and evaluation per tag

2026-10-01T20:56Z · tool · informational · measured

The lead's fire script uploads one qualified archive to the board once with the client patched to a single HTTP attempt per request (no silent re-posts), a tag usable once, the API key handed only to the child process, and writes attempt, response and receipt records; a poller then stores the evaluation summary and the full per-network evaluation record for the projection checks. All seven submissions of 2026-10-02 went through it.

Recorded measurement:

    seven fires s36..s42 on 2026-10-02; eval_<id>.json about 94 kB each

## Programs

- [code/fire_once.py](code/fire_once.py)
- [code/poll_eval.sh](code/poll_eval.sh)
- [code/compare_evals.py](code/compare_evals.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L82](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L82)
- [TEAM_LOG_2026-10-01_to_10-22.md#L96](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L96)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-10-01-single-attempt-submission-uploader-records-one-qualified-archive-receipt) · [Copy provenance](../../PROVENANCE.json)

