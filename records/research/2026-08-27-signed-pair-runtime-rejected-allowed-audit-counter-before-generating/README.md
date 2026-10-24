# Signed-pair runtime rejected an allowed audit counter before generating any row

2026-08-27 · incident · failed · measured

Runtime V3 had independent static GO and consumed its one-use claim, then failed before row generation when an overbroad substring firewall rejected the exact-zero target-blind audit counter as a forbidden `manifest` field. Failure e64f998f says published_bundle=false and status terminal post-claim/no-retry. This closes V3 only, not the signed-pair source or accuracy law.

Recorded measurement:

    Runtime V3 had independent static GO and consumed its one-use claim, then failed before row generation when an overbroad substring firewall rejected the exact-zero target-blind audit counter as a forbidden `manifest` field. Failure e64f998f says published_bundle=false and status terminal post-claim/no-retry. This closes V3 only, not the signed-pair source or accuracy law.

## Programs

- [capture_adapter.py](capture_adapter.py)
- [parent_capture.py](parent_capture.py)
- [runtime_token.py](runtime_token.py)
- [signed_pair_literal.py](signed_pair_literal.py)
- [signed_pair_reference.py](signed_pair_reference.py)

## Evidence

- [INDEPENDENT_STATIC_REVIEW_V3.json#L1](INDEPENDENT_STATIC_REVIEW_V3.json#L1)
- [THREE_BANK_RUN.CLAIM.json#L1](THREE_BANK_RUN.CLAIM.json#L1)
- [THREE_BANK_RUN.FAILURE.json#L1](THREE_BANK_RUN.FAILURE.json#L1)
- [V3_TERMINAL_FAILURE_LINEAGE_V4.json#L1](V3_TERMINAL_FAILURE_LINEAGE_V4.json#L1)

<details>
<summary>All result files and records (4)</summary>

- [INDEPENDENT_STATIC_REVIEW_V3.json](INDEPENDENT_STATIC_REVIEW_V3.json) · result
- [THREE_BANK_RUN.CLAIM.json](THREE_BANK_RUN.CLAIM.json) · result
- [THREE_BANK_RUN.FAILURE.json](THREE_BANK_RUN.FAILURE.json) · result
- [V3_TERMINAL_FAILURE_LINEAGE_V4.json](V3_TERMINAL_FAILURE_LINEAGE_V4.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-27-signed-pair-runtime-rejected-allowed-audit-counter-before-generating) · [Copy provenance](../../../PROVENANCE.json)

