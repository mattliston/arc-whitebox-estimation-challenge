# Literal-only candidate verification uses source equivalence and one boundary execution

2026-10-02T08:30Z · tool · informational · measured

A dial candidate whose source differs from the last fully qualified file only in float literals, packed arrays or docstrings (proved by walking both syntax trees) needs one boundary session instead of the hour-long chain: under the platform clock every count of a session (requests, handles, calls per operation, billed flops per family, wire bytes) repeats exactly, so the new file inherits the qualified file's wall, residual and memory projection if its census is equal. Tested with a passing pair (seven literals moved) and a failing pair (equal column-shift strengths take a value-dependent shortcut: 38 requests and 80 kflop fewer). The S42 file was fired through it.

Recorded measurement:

    PASS case: census identical on setup and networks 35 / 100 (requests 42,797 / 40,927; bill 258,268,828,893 / 257,655,715,185 flops), except 16 bytes of int64; FAIL case: -38 requests, bill -80,512 / -80,128 flops; gates: residual / kernel instructions +-3% / +-2% calm, +-20% / +-10% loaded (REPEAT beyond); sessions 2-3 minutes quiet, 12-15 at load 20; S38 bill by family: matmul 82.00%, add 7.40%, subtract 5.63%, multiply 1.51%

## Programs

- [code/fast_path.sh](code/fast_path.sh)
- [code/literal_diff.py](code/literal_diff.py)
- [code/selftest_literal_diff.py](code/selftest_literal_diff.py)
- [code/variants/s38_7c1a67f4_dialsA/candidate.py](code/variants/s38_7c1a67f4_dialsA/candidate.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L176](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L176)
- [TEAM_LOG_2026-10-01_to_10-22.md#L177](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L177)
- [TEAM_LOG_2026-10-01_to_10-22.md#L178](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L178)
- [TEAM_LOG_2026-10-01_to_10-22.md#L179](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L179)
- [TEAM_LOG_2026-10-01_to_10-22.md#L199](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L199)
- [RESULT.md#L979](../release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/RESULT.md#L979)

<details>
<summary>All result files and records (3)</summary>

- [fast_path.json](fast_path.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [RESULT.md](../release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/RESULT.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-10-02-literal-only-candidate-verification-uses-source-equivalence-one-boundary) · [Copy provenance](../../PROVENANCE.json)

