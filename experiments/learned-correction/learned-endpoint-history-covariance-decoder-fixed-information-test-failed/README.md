# Learned endpoint-history covariance decoder: fixed information test failed

2026-09-06T08:42Z · experiment · rejected · measured

Fixed65 endpoint features (64 learned states plus constant), symmetric Theta0/1/2 shared over late layers8..14, multiplied by R^0/R/R^2. Six train35/100/150/200/250/300, held-out400/500. Information gate required BOTH ratios<=.75 on EACH net.

Recorded measurement:

    Fixed65 endpoint features (64 learned states plus constant), symmetric Theta0/1/2 shared over late layers8..14, multiplied by R^0/R/R^2. Six train35/100/150/200/250/300, held-out400/500. Same100-epoch objective, initialization/seeds/LR/ridge as designed. Final epoch only,7.83s MPS.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/train_covariance_innovation_probe_v1.py](code/train_covariance_innovation_probe_v1.py)
- [code/train_learned_pair_history_v1.py](code/train_learned_pair_history_v1.py)

## Evidence

- [LEARNED_PAIR_HISTORY_INFORMATION_V1_CLOSED.md#L1](LEARNED_PAIR_HISTORY_INFORMATION_V1_CLOSED.md#L1)

<details>
<summary>All result files and records (3)</summary>

- [LEARNED_PAIR_HISTORY_INFORMATION_V1_CLOSED.md](LEARNED_PAIR_HISTORY_INFORMATION_V1_CLOSED.md) · record
- [learned_pair_history_verdict_v1.json](learned_pair_history_verdict_v1.json) · result
- [LEARNED_PAIR_HISTORY_INFORMATION_DESIGN_V1.md](LEARNED_PAIR_HISTORY_INFORMATION_DESIGN_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-learned-endpoint-history-covariance-decoder-fixed-information-test-failed) · [Copy provenance](../../../PROVENANCE.json)

