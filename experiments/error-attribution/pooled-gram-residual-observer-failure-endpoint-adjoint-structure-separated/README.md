# Pooled-Gram residual observer failure: endpoint-adjoint structure separated hard cases

2026-08-24 · experiment · informational · measured

The read-only K-versus-J replay shows that pooled Gram/context expansion is not a hard-case discriminator: K improves the four hard rows but regresses the ordinary 684, and scalar future-gain scores have near-zero discrimination. A fixed endpoint-adjoint diagnostic instead finds 9.42x ordinary and 3.77x hard residual enrichment over a random rank-eight subspace; the full oracle pullback norm is 1.586x larger on hard rows and correlates 0.764 with J mean squared error (MSE). This licenses exactly one full-width learned backward-adjoint successor while rejecting scalar and fixed low-rank variants.

Recorded measurement:

    The read-only K-versus-J replay shows that pooled Gram/context expansion is not a hard-case discriminator: K improves the four hard rows but regresses the ordinary 684, and scalar future-gain scores have near-zero discrimination. A fixed endpoint-adjoint diagnostic instead finds 9.42x ordinary and 3.77x hard residual enrichment over a random rank-eight subspace; the full oracle pullback norm is 1.586x larger on hard rows and correlates 0.764 with J MSE. This licenses exactly one full-width learned backward-adjoint successor while rejecting scalar and fixed low-rank variants.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_full688_bidirectional_adjoint_smoother_design_v1/reproduce_readonly_diagnostic.py](code/public_full688_bidirectional_adjoint_smoother_design_v1/reproduce_readonly_diagnostic.py)
- [code/public_full688_local_innovation_transport_v1/dataset.py](code/public_full688_local_innovation_transport_v1/dataset.py)
- [code/public_full688_transport_energy_coherence_cv_v1/run_experiment.py](code/public_full688_transport_energy_coherence_cv_v1/run_experiment.py)

## Evidence

- [CV_RESULTS.json#L1](CV_RESULTS.json#L1)
- [CV_RESULTS-2a8a5695.json#L1](CV_RESULTS-2a8a5695.json#L1)
- [VERIFICATION.json#L1](VERIFICATION.json#L1)
- [K_VS_J_OOF_DECOMPOSITION_AND_SUCCESSOR.json#L156](K_VS_J_OOF_DECOMPOSITION_AND_SUCCESSOR.json#L156)
- [DIAGNOSTIC_RESULT.json#L32](../../learned-correction/learned-backward-adjoint-residual-observer-rejected-full-width-cross/DIAGNOSTIC_RESULT.json#L32)
- [DESIGN_RECEIPT.json#L26](../../learned-correction/learned-backward-adjoint-residual-observer-rejected-full-width-cross/DESIGN_RECEIPT.json#L26)

<details>
<summary>All result files and records (6)</summary>

- [CV_RESULTS.json](CV_RESULTS.json) · result
- [CV_RESULTS-2a8a5695.json](CV_RESULTS-2a8a5695.json) · result
- [VERIFICATION.json](VERIFICATION.json) · result
- [K_VS_J_OOF_DECOMPOSITION_AND_SUCCESSOR.json](K_VS_J_OOF_DECOMPOSITION_AND_SUCCESSOR.json) · result
- [DIAGNOSTIC_RESULT.json](../../learned-correction/learned-backward-adjoint-residual-observer-rejected-full-width-cross/DIAGNOSTIC_RESULT.json) · result
- [DESIGN_RECEIPT.json](../../learned-correction/learned-backward-adjoint-residual-observer-rejected-full-width-cross/DESIGN_RECEIPT.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-24-pooled-gram-residual-observer-failure-endpoint-adjoint-structure-separated) · [Copy provenance](../../../PROVENANCE.json)

