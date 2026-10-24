# True post-mean feedback revealed error transfer through the covariance outer product

2026-09-04 · finding · informational · measured

MEAN-LEAK ORACLE (RETRO_MEANLEAK_TRUTH: the true post-mean error delta fed into the recursion; independent 4M; operator OFF numbers): the post covariance is C = M11 - m m^T, so a per-unit post-mean error leaks into it as -(m d^T + d m^T) - d d^T: rank 2, paired +/- spectrum, mean-aligned = the T-100 residual's signature. Covariance-only correction diverges (3.3e-6; r100 1e2); with the corrected mean ALSO carried forward: r35 8.62e-8 -> 2.13e-8 (-75%), r100 7.70e-8 -> 2.37e-8 (-69%) with NO operator; with the deployed operator on top it double-counts (3.8e-8 / 6.0e-8). The operator estimates exactly this delta per layer (64.5% held-out) but is only patched onto the output rows: the recursion never sees it ([instruction removed]).

Recorded measurement:

    - T-103 MEAN-LEAK ORACLE (RETRO_MEANLEAK_TRUTH: the true post-mean error delta fed into the recursion; independent 4M; operator OFF numbers): the post covariance is C = M11 - m m^T, so a per-unit post-mean error leaks into it as -(m d^T + d m^T) - d d^T: rank 2, paired +/- spectrum, mean-aligned = the T-100 residual's signature. Covariance-only correction diverges (3.3e-6; r100 1e2); with the corrected mean ALSO carried forward: r35 8.62e-8 -> 2.13e-8 (-75%), r100 7.70e-8 -> 2.37e-8 (-69%) with NO operator; with the deployed operator on top it double-counts (3.8e-8 / 6.0e-8). The operator estimates exactly this delta per layer (64.5% held-out) but is only patched onto the output rows: the recursion never sees it ([instruction removed]).

## Evidence

- [LANE_g3_tracker.md#L783](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L783)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-true-post-mean-feedback-revealed-error-transfer-through-covariance) · [Copy provenance](../../../PROVENANCE.json)

