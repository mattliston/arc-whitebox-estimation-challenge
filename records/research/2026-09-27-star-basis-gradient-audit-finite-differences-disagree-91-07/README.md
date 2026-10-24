# Star-basis gradient audit: finite differences disagree by 91.07%

2026-09-27 · finding · inconclusive · measured

Frozen- and full-basis primal outputs were identical, but derivative evidence failed because fixed-step finite differences were unstable and row/column branches changed. No optimizer or training update was qualified.

Recorded measurement:

    Six full-network passes on TRAIN35; full/frozen final tangents differ111.10%; finite differences disagree91.07%; full/frozen errors against smaller step96.19%/102.71%; repaired peak5.270GB,zero added swap.

## Evidence

- [RESULTS.md#L1105](../../../experiments/third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md#L1105)

<details>
<summary>All result files and records (2)</summary>

- [RESULTS.md](../../../experiments/third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md) · record
- [star_basis_gradient_closeout_v1.json](star_basis_gradient_closeout_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-27-star-basis-gradient-audit-finite-differences-disagree-91-07) · [Copy provenance](../../../PROVENANCE.json)

