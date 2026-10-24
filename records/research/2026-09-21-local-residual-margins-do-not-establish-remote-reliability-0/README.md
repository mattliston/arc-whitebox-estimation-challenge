# Local residual margins do not establish remote reliability: 0.209 s local maximum, 0.4047 s remote failure at 0.4 s cap

2026-09-21T15:32:39Z · lesson · informational · measured

The readout composition #331759 passed the local ordinary-client checks with a maximum candidate residual of 0.209131764 s and equal complete bills, yet on the platform network 4 failed RESIDUAL_WALL_TIME_EXHAUSTED at 0.4047110810000021 s (predict wall 113.493815336 s and 418771789071 FLOPs, neither exhausted) and ten more networks failed on residual or total time. The record's rule: a local residual margin is not evidence of remote reliability, and an unchanged archive must not be resubmitted to select favourable runtime noise.

Recorded measurement:

    #331759: 89 successful, 11 failed (4 residual-time, 7 total-time), adjusted 0.020395830787829743; the sibling #331760 fired 20 s later passed 100/100 with maximum residual 0.3824777899999958 s

## Evidence

- [approved_uploads_checkpoint102.md#L16](../2026-09-21-exact-archive-upload-approvals-readout-composition-refit-mean-encoder/approved_uploads_checkpoint102.md#L16)
- [approved_uploads_checkpoint102.md#L26](../2026-09-21-exact-archive-upload-approvals-readout-composition-refit-mean-encoder/approved_uploads_checkpoint102.md#L26)
- [bank_and_signed_fit_checkpoint103.md#L9](../../../submissions/331759/bank_and_signed_fit_checkpoint103.md#L9)

<details>
<summary>All result files and records (3)</summary>

- [approved_uploads_checkpoint102.md](../2026-09-21-exact-archive-upload-approvals-readout-composition-refit-mean-encoder/approved_uploads_checkpoint102.md) · record
- [official_completion_331759.json](../../../submissions/331759/official_completion_331759.json) · record
- [bank_and_signed_fit_checkpoint103.md](../../../submissions/331759/bank_and_signed_fit_checkpoint103.md) · record

</details>

## Related work

- measured by: [Submission #331759: copied 65-coefficient readout on the covariance-edge estimator; 11 of 100 networks failed](../../../submissions/331759/README.md)
- ← enables: [Submission #331759: copied 65-coefficient readout on the covariance-edge estimator; 11 of 100 networks failed](../../../submissions/331759/README.md)

[Research graph](../../../graph/index.html#node=2026-09-21-local-residual-margins-do-not-establish-remote-reliability-0) · [Copy provenance](../../../PROVENANCE.json)

