# Partial server-marks JSON caused native evaluation failure; complete-line transport preserved numeric behavior

2026-10-20T23:37:21Z · incident · informational · measured

A client read a partially written server-marks line and failed with JSONDecodeError. A complete-line transport repair passed partial JSON, UTF-8, newline and timeout fixtures and preserved numerical behavior without changing active sources or kernel guards.

Recorded measurement:

    Concreteinfrastructurefailure: nonlinear96 v7full32 aborted23:16:02 after clientmark() read partialserver_marks.jsonl (JSONDecodeError unterminatedstring); no model/OOMverdict. Other7activegroupsretired, pending9-12unrun, incomplete12rowsnotfull32. FIX A/cloud_linux_markline_v1: client1beba822cd02cebdfe9f8b30d0e08fb498a870a85fecaeccbe2fa38782098455; group e2004b88; scheduler315bc522; grader602cfb76; canonicalcomplete-linefragmentca5f41ca. PartialJSON/UTF8/newline/timeout fixtures+officialNumPyfreeimport+ASTnumericparityPASS. UsesABSOLUTE originalv7guard884199a8/server491dc628/moverunchanged; no kernelguard changes or activefilemutations.

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6383](../../TEAM_LOG_2026-10-01_to_10-22.md#L6383)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-partial-server-marks-json-caused-native-evaluation-failure-complete) · [Copy provenance](../../../PROVENANCE.json)

