# BLAS environment changes final error up to 0.31% per network despite identical arithmetic counts

2026-09-23T13:51Z · finding · informational · measured

Native parity differences were traced to floating-point order in the BLAS backend rather than a changed estimator or builder layer. Comparisons require the same environment.

Recorded measurement:

    2026-09-23 06:51 porter: S7 bisection RESOLVED. No layer breaks bit-identity. S7-full-cb3 (9011916a) is BIT-EQUAL to candS7 (5bf2a07e) at all 16 layers on r35/r100 in engineer2's OpenBLAS env and reproduces the quoted +0.04/-0.09% vs S6b there. The -0.31/+0.18% was the .venv (macOS system BLAS), where S6b's own MSE differs from OpenBLAS by up to 0.31%/net; FLOPs are identical across envs. Gate and VALID40 held for the lead.

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L240](../../TEAM_LOG_2026-09-22_to_09-26.md#L240)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-blas-environment-changes-final-error-up-0-31-per) · [Copy provenance](../../../PROVENANCE.json)

