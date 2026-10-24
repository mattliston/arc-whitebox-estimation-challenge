# Isolated symmetric covariance-mean supply double-counts the existing third-order covariance read

2026-09-23T01:40Z · finding · informational · measured

The symmetric covariance-mean error is real, but injecting that term alone omits cancelling parts of the true plane error. Removing it from an otherwise true state worsened next-layer variance sevenfold.

Recorded measurement:

    2026-09-22 18:40 analyst: hub_planes_*_honest are the hub-carrier MODEL (26-38% off 8M MC truth), not truth. Vs MC truth the card's Sym(C,mu) plane error is real (45-64% of error energy L8-14), per-unit k3 already carries the gain term (card beta = truth beta). One-layer pair-map test: truth plane improves v_{l+1} ~10%, kappa*Sym(C,mu) alone worsens it 3-4x (the rest of the true error cancels its covariance read) -> Sym-only injections are physically incomplete.
    2026-09-22 19:10 physicist: pair map with ALL-truth inputs (8M) reproduces true post cov at the MC noise floor; removing kappa*Sym from TRUTH K21 on the true state -> v_{l+1} 7x worse, so the card's K21 already holds a read-equivalent stand-in (Sym supply double-counts). Truth K22/K31 do not compensate (1-3e-4 v change per layer).

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L6](../../TEAM_LOG_2026-09-22_to_09-26.md#L6)
- [TEAM_LOG_2026-09-22_to_09-26.md#L8](../../TEAM_LOG_2026-09-22_to_09-26.md#L8)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-isolated-symmetric-covariance-mean-supply-double-counts-existing-third) · [Copy provenance](../../../PROVENANCE.json)

