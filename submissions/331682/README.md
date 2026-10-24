# Submission #331682: lossless static parameter packing of the 65-coefficient final-head refit; 100/100 completed

2026-09-20T19:53:50Z · submission · adopted · measured

The failed readout source #331676 was repackaged with a lossless static packing of its FP32 parameter literals (a 999625-character GNN literal and a 713384-character older final-head literal), preserving every parameter byte, output byte, numerical operation order and complete FLOP bill. Five matched cold NumPy-free client pairs gave a mean cold residual ratio of 0.672566421412 (maximum 0.213847026 s) against the parent, but a later prediction-only probe showed that gain was almost entirely module loading, which the installed SDK performs outside the timed predict. Accepted 2026-09-20T19:53:50.713Z, it completed all 100 networks with official residual wall min/mean/max 0.28950262200000054/0.3232998421000002/0.3946758339999974 s and was graded adjusted 3.600753016428718e-9, 0.849608% better than #331672; the record states that completion does not prove packing repaired the timeouts.

Recorded measurement:

    official adjusted 3.600753016428718e-9, raw 1.9759227321003436e-8; 100/100 completed, 0 failures; official residual s min 0.28950262200000054 mean 0.3232998421000002 max 0.3946758339999974; local mean cold residual ratio 0.672566421412, max 0.213847026 s, total-wall ratio 0.994861771476; prediction-only residual ratio 0.996339965486; archive 345810 bytes sha256 d68af333ba02bb0d5ac71487736af4624f7d9558b14bca9121f34be36a73eb52; source sha256 822f96fa01cb9d2ff8cfeb191cf845fc779c9739782bf395ac63d16e769b3811

Public score: 3.600753016428718e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [OFFICIAL_RESULT.md#L3](OFFICIAL_RESULT.md#L3)
- [OFFICIAL_RESULT.md#L4](OFFICIAL_RESULT.md#L4)
- [OFFICIAL_COMPLETION_331682.md#L3](OFFICIAL_COMPLETION_331682.md#L3)
- [PROFILE_FINDINGS.md#L10](../../records/research/2026-09-20-cold-call-client-timing-included-module-loading-32-7/PROFILE_FINDINGS.md#L10)
- [PROFILE_FINDINGS.md#L11](../../records/research/2026-09-20-cold-call-client-timing-included-module-loading-32-7/PROFILE_FINDINGS.md#L11)
- [REEXAMINATION_INDEX_20260919.md#L875](../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L875)
- [REEXAMINATION_INDEX_20260919.md#L876](../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L876)
- [REEXAMINATION_INDEX_20260919.md#L866](../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L866)
- [REEXAMINATION_INDEX_20260919.md#L867](../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L867)

<details>
<summary>All result files and records (13)</summary>

- [OFFICIAL_RESULT.md](OFFICIAL_RESULT.md) · record
- [OFFICIAL_COMPLETION_331682.md](OFFICIAL_COMPLETION_331682.md) · record
- [RESULT_v2.md](RESULT_v2.md) · record
- [PACKING_PROTOCOL.md](PACKING_PROTOCOL.md) · config
- [PROFILE_FINDINGS.md](../../records/research/2026-09-20-cold-call-client-timing-included-module-loading-32-7/PROFILE_FINDINGS.md) · record
- [official_best_331682.json](official_best_331682.json) · record
- [official_result_331682.json](official_result_331682.json) · record
- [closeout_official_v1.json](closeout_official_v1.json) · record
- [official_comparison_331682_331684.json](official_comparison_331682_331684.json) · result
- [official_objective_gap_331682.json](official_objective_gap_331682.json) · result
- [official_completion_331682.json](official_completion_331682.json) · result
- [official_evaluation_331682.json](official_evaluation_331682.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

## Related work

- ← rejects: [Cold-call client timing included module loading: the 32.7% packing residual gain shrinks to 0.366% prediction-only](../../records/research/2026-09-20-cold-call-client-timing-included-module-loading-32-7/README.md)
- ← shipped in: [Standing approval for all qualified Phase 2 uploads recorded 2026-09-20 after the exact-archive approval of #331672](../../records/research/2026-09-20-standing-approval-all-qualified-phase-2-uploads-recorded-2026/README.md)
- ← enables: [Copying 65 readout coefficients onto the covariance-edge estimator gives eight-network adjusted ratio 0.995858110143](../../experiments/learned-correction/copying-65-readout-coefficients-onto-covariance-edge-estimator-gives/README.md)
- ← enables: [Submission #331676: 65-literal final-head ridge refit of #331672 failed: two networks exceeded the 0.4 s residual cap](../331676/README.md)
- measured by: [Cold-call client timing included module loading: the 32.7% packing residual gain shrinks to 0.366% prediction-only](../../records/research/2026-09-20-cold-call-client-timing-included-module-loading-32-7/README.md)
- supersedes: [Submission #331676: 65-literal final-head ridge refit of #331672 failed: two networks exceeded the 0.4 s residual cap](../331676/README.md)
- ← supersedes: [Submission #331684: packed frozen covariance-edge correction at layers 8-14 with full lower-moment support](../331684/README.md)
- ← enables: [Submission #331759: copied 65-coefficient readout on the covariance-edge estimator; 11 of 100 networks failed](../331759/README.md)

[Research graph](../../graph/index.html#node=submission-331682) · [Copy provenance](../../PROVENANCE.json)

