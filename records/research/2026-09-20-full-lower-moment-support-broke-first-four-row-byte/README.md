# Full lower-moment support broke the first-four-row byte gate: score-sorted vs natural summation differs by up to 6.87e-7

2026-09-20 · finding · informational · measured

The full lower-moment candidate reproduced only the first two parent output rows instead of the predeclared four, with maximum changes of 1.1324882507324219e-6 / 1.0132789611816406e-6 at layer index 2 on nets 35/100. Source review found the premise wrong rather than the candidate: the selector returns score-sorted indices even when all 1024 rows survive, the compact covariance and star operators sum in that order while the new full helpers use natural order, and permuted indices differ by up to 6.873160600662231e-7 (covariance) and 1.4751204812198581e-8 (star); the original gate was kept as failed and a separately sealed objective study was declared.

Recorded measurement:

    first-row agreement 2 of required 4; layer-2 max change 1.1324882507324219e-6 (net 35) / 1.0132789611816406e-6 (net 100); permuted-index helper differences covariance 6.873160600662231e-7, star 1.4751204812198581e-8; six estimator calls 2368162176070 F

## Evidence

- [QUALIFICATION_RESULT.md#L3](QUALIFICATION_RESULT.md#L3)
- [QUALIFICATION_RESULT.md#L12](QUALIFICATION_RESULT.md#L12)
- [QUALIFICATION_RESULT.md#L13](QUALIFICATION_RESULT.md#L13)
- [QUALIFICATION_RESULT.md#L15](QUALIFICATION_RESULT.md#L15)
- [QUALIFICATION_RESULT.md#L19](QUALIFICATION_RESULT.md#L19)
- [QUALIFICATION_RESULT.md#L20](QUALIFICATION_RESULT.md#L20)

<details>
<summary>All result files and records (4)</summary>

- [QUALIFICATION_RESULT.md](QUALIFICATION_RESULT.md) · result
- [SOURCE_REVIEW.md](SOURCE_REVIEW.md) · record
- [unit_result.json](unit_result.json) · result
- [PROTOCOL.md](PROTOCOL.md) · config

</details>

[Research graph](../../../graph/index.html#node=2026-09-20-full-lower-moment-support-broke-first-four-row-byte) · [Copy provenance](../../../PROVENANCE.json)

