# Two-sided 48-by-48 star core: initial gain fails eight-network expansion

2026-09-26T23:11Z · experiment · rejected · measured

The wide final star projection and singular-value decomposition were replaced by a 48-by-48 QR core while preserving ranks48/32 and the power step. This approximation omits the right-subspace complement and failed expanded scoring, so the conditional combined candidate was never built.

Recorded measurement:

    TRAIN2 adjusted-0.50026%; TRAIN8 adjusted-0.199676%,95% interval[-0.623171%,+0.272431%],6/8 wins; raw+0.400516%; bill-0.594413%; additional6 adjusted-0.11181% with interval crossing zero.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_two_sided_star_core_v1.py](code/candidate_two_sided_star_core_v1.py)

## Evidence

- [RESULTS.md#L772](../all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md#L772)

<details>
<summary>All result files and records (1)</summary>

- [RESULTS.md](../all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-26-two-sided-48-48-star-core-initial-gain-fails) · [Copy provenance](../../../PROVENANCE.json)

