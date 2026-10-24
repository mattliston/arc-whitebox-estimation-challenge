# Exact variance correction: error 1.466e-8 versus deployed 3.197e-8

2026-09-02 · experiment · informational · measured

dv_{l+1} = diag(W^T dC_post_l W): the pushed error of the closure's post-covariance. Exact v = 1.466e-8 vs deployed 3.197e-8 (10-net leave-one-out (LOO)). Form is proven optimal given inputs (N1.1), so the whole channel is INPUT error: third-order plane (N1.2, the big leaf), covariance (N1.3, small).

Recorded measurement:

    dv_{l+1} = diag(W^T dC_post_l W): the pushed error of the closure's post-covariance. Exact v = 1.466e-8 vs deployed 3.197e-8 (10-net LOO). Form is proven optimal given inputs (N1.1), so the whole channel is INPUT error: third-order plane (N1.2, the big leaf), covariance (N1.3, small). HONEST FRAME (T-36, independent 4M truth, r35, F2J-on, deployed 2.704e-8, +-3 points): the 47.4-point 'truth v' ceiling was ~35 points of same-sample noise-matching (12.4 honest, itself depressed by the per-layer truth's own noise); truth planes through the consumer 24.4; the engine's v-arm 8.5 (β 0.4). The v channel is real but smaller than priced on 08-31; the whole v-pool lives in layers 8-15 (shallow 'truth' targets are 76-98% noise).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_d11_util_halving_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/estimator_cap.py](code/estimator_cap.py)

## Evidence

- [MECHANISM_TREE.md#L2](MECHANISM_TREE.md#L2)

<details>
<summary>All result files and records (1)</summary>

- [MECHANISM_TREE.md](MECHANISM_TREE.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-02-exact-variance-correction-error-1-466e-8-versus-deployed) · [Copy provenance](../../../PROVENANCE.json)

