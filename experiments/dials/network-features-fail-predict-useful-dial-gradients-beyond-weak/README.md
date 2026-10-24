# Network features fail to predict useful dial gradients beyond a weak permutation-null result

2026-10-02T12:37Z · experiment · rejected · measured

Whether a dial's per-network gradient can be predicted from what the card knows for free (27 features from the weights, layer variances and true means) was tested on 72 networks' exact gradients: 6 of 45 dials beat the 95th percentile of their own null with one feature, but shuffled gradients give 2.2 such dials on average and 6 or more in 6% of shuffles, and none of the four principal components of the gradients is predictable. The three pre-registered relations tested on 291 other networks gave r = +0.007, +0.03 and -0.13 (opposite sign) where a real -0.40 would have given t = -7. Conditional dials are closed for these features.

Recorded measurement:

    first pass: edge R2 0.18 (p 0.004), s_new / s_carry R2 0.13 / 0.12 (p 0.010 / 0.013, r -0.40); null: 2.2 dials expected, >= 6 in 6% of shuffles; retest on 291 networks: s_new r +0.007 (t +0.1), s_carry +0.03 (t +0.5), k4a -0.13 (t -2.2); per-network gradient sd 2.0 (s_new), 4.5 (s_carry)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/analyst`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/a17_conditional_dials.py](code/a17_conditional_dials.py)
- [code/a17_retest_oldcard.py](code/a17_retest_oldcard.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L173](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L173)
- [TEAM_LOG_2026-10-01_to_10-22.md#L310](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L310)
- [RESULT.md#L957](../../../records/research/2026-10-01-torch-research-estimator-lowers-final-layer-error-0-73/RESULT.md#L957)

<details>
<summary>All result files and records (3)</summary>

- [a17_conditional_dials.md](a17_conditional_dials.md) · result
- [RESULT.md](../../../records/research/2026-10-01-torch-research-estimator-lowers-final-layer-error-0-73/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-network-features-fail-predict-useful-dial-gradients-beyond-weak) · [Copy provenance](../../../PROVENANCE.json)

