# Automatic approval review blocked two qualified archive uploads for lacking exact-payload authorization

2026-09-19T18:43:07Z · incident · failed · measured

The agent's upload step refused both the qualified Cost150 archive (exact kernel-cost schedule card) and later the Cost75 archive (the same with a 750k-FLOP request penalty) because its automatic review could not find an authorization naming that specific payload and destination; a prior approval of another archive did not transfer. No submission id existed until the owner approved each exact archive, after which the identical pinned Cost150 archive was accepted as #331584.

Recorded measurement:

    Cost150 archive SHA 6356d5d81d14aced04611c5279f44e70f955abd2dbabe3d0f073ecc5132702e2 (770155 bytes); Cost75 archive SHA 4facbcf450861a0eaa5c28186bd34045d6b7953919ed5e367217050f4692a4ff, rejection 2026-09-19T18:43:07Z

## Evidence

- [RECEIPTS.md#L138](../../../experiments/birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md#L138)
- [RECEIPTS.md#L266](../../../experiments/birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md#L266)
- [RECEIPTS.md#L171](../../../experiments/birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md#L171)

<details>
<summary>All result files and records (3)</summary>

- [cost150_submission_review_rejection.json](cost150_submission_review_rejection.json) · record
- [cost75_submission_review_rejection.json](cost75_submission_review_rejection.json) · record
- [RECEIPTS.md](../../../experiments/birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md) · record

</details>

## Related work

- enables: [Five named archive uploads approved; the strongest qualified delayed-rotation archive selected](../2026-09-19-five-named-archive-uploads-approved-strongest-qualified-delayed-rotation/README.md)

[Research graph](../../../graph/index.html#node=2026-09-19-automatic-approval-review-blocked-two-qualified-archive-uploads-lacking) · [Copy provenance](../../../PROVENANCE.json)

