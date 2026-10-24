# Paired telemetry review: the lost networks are participant-side spikes and slow-worker episodes, not a platform shift

2026-10-16T22:22:01Z · finding · informational · measured

After the user objected that the failures were code, not platform, the lead reviewed the paired telemetry of every run since #336342: per-worker kernel medians were 0.94-1.04 of #336342 and 66-73% of networks had lower residual, so there was no uniform shift. The majority class is one-network participant spikes (residual +7..27% with kernel within +-8%) on networks where the Python did more work (lazy view binds range 13-72 per network, median 32; cold first nets bind most); the second class is worker-speed episodes (one worker +10-22% kernel) that are fatal only without margin. The standard became: no per-network lazy work and a residual median <= .33 s.

Recorded measurement:

    Per-worker kernel medians 0.94-1.04 vs #336342; 66-73% of nets lower residual (median ratio 0.96-0.97). Class B spikes: S3prime 58, S3pbt 88, S4 79/85, S4gb 1/49/71, S2tr 93, V-A-noB 3 (cold first net), S2t 18/24/42; binds per network 13-72, median 32. Class A: b709 worker 4 first three nets kernel +10-22% (two TIME_EXHAUSTED at 120 s, residual .484); S2t worker 1 +6% whole session (4 of its 7 losses).

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4520](../../TEAM_LOG_2026-10-01_to_10-22.md#L4520)

<details>
<summary>All result files and records (3)</summary>

- [eval_336570.json](../../../submissions/336570/eval_336570.json) · result
- [eval_336412.json](../../../submissions/336412/eval_336412.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← supersedes: [Residual noise on Oct 16: per-run sd .023 to .036 and spikes of +10-28% on 1-6 networks; P(any fail) 89% at median .355](../2026-10-16-residual-noise-oct-16-per-run-sd-023-036/README.md)

[Research graph](../../../graph/index.html#node=2026-10-16-paired-telemetry-review-lost-networks-participant-side-spikes-slow) · [Copy provenance](../../../PROVENANCE.json)

