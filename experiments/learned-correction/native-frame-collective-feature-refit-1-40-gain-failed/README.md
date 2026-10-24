# Native-frame collective-feature refit: 1.40% gain failed the preregistered halving requirement

2026-08-31 · experiment · rejected · measured

Collective features were added in the native estimator frame to resolve an earlier frame mismatch. The first fold improved slightly against its matched baseline and failed the preregistered requirement to halve the remaining error.

Recorded measurement:

    {
      "baseline_fold_final_mse": 3.154306195654431e-08,
      "matched_baseline": "F2J fold-0 banked artifact",
      "mean_delta": 4.426129994206047e-10,
      "most_nets_improving": true,
      "networks": 147,
      "networks_improved": 121,
      "networks_worsened": 26,
      "node": "P2R0G19B_REFIT",
      "reduction_vs_matched_fold": 0.01403202390530045,
      "refit_fold_final_mse": 3.110044895712371e-08,
      "sentinel_fold": 0,
      "status": "sentinel-fail"
    }
    

Program source is in [`code/`](code/), preserving the relative layout below `.`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_f2q_native_collective_cv_v1/run_experiment.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_f2q_native_collective_cv_v1/run_experiment.py)
- [code/estimator.py](code/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_collective_card_v2/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_collective_card_v2/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_collective_card_v2/export_coll_features.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_collective_card_v2/export_coll_features.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_full688_local_innovation_transport_v1/dataset.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_full688_local_innovation_transport_v1/dataset.py)

## Evidence

- [SENTINEL_VERDICT.json#L1](SENTINEL_VERDICT.json#L1)

<details>
<summary>All result files and records (4)</summary>

- [SENTINEL_VERDICT.json](SENTINEL_VERDICT.json) · result
- [REFIT_RESULT.json](REFIT_RESULT.json) · result
- [PREREGISTER.md](PREREGISTER.md) · record
- [fold_0.json](fold_0.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-31-native-frame-collective-feature-refit-1-40-gain-failed) · [Copy provenance](../../../PROVENANCE.json)

