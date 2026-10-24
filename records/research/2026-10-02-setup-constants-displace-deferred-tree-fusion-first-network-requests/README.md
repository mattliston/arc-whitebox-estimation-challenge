# Setup constants displace deferred tree fusion: first-network requests fall 381 while warm requests rise 107

2026-10-02T17:15Z · finding · informational · measured

Moving persistent constants into setup helps the first network but uses time previously spent fusing trees. Under the platform setup clock 47 deferred fusions were displaced, increasing every warm network by 107 requests; the binding-phase cap moved to 18,600 requests with the slow-setup gate retained.

Recorded measurement:

    first / warm requests -381 / +107; 490 requests for constants; deferred fusions 266 -> 219; binding cap 18,100 -> 18,600; setup work displaces 0.15 to 0.25 requests per network per setup request

## Programs

- [../2026-10-02-designation-caps-set-43-800-first-network-requests-42/variant_table.py](../2026-10-02-designation-caps-set-43-800-first-network-requests-42/variant_table.py)
- [../2026-10-02-submission-designation-requires-successful-grading-measured-wall-residual/designation_judge.py](../2026-10-02-submission-designation-requires-successful-grading-measured-wall-residual/designation_judge.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/project.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/project.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L392](../../TEAM_LOG_2026-10-01_to_10-22.md#L392)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-setup-constants-displace-deferred-tree-fusion-first-network-requests) · [Copy provenance](../../../PROVENANCE.json)

