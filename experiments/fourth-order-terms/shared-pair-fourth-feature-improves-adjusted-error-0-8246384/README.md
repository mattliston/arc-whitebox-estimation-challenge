# Shared-pair fourth feature improves adjusted error 0.8246384% but misses the 5% magnitude gate

2026-09-20 · experiment · rejected · measured

A single terminal fourth-order feature added a small transferable signal in fixed five-fold cross-validation.

Recorded measurement:

    The fixed scalar feature improves the exploratory20-network cross-validation
    | Unchanged Lazy | 3.907171531554795e-09 |
    | Terminal66-channel control | 3.892198872749762e-09 |
    | Control plus one shared-pair fourth feature | 3.874951494461463e-09 |
    Extension/unchanged **0.9917536159257102** (0.8246384% better)
    and extension/control **0.9955687314928708** (0.4431269% better).
    Both magnitude gates, <=.95 and <=.97, FAIL. Four of five folds beat both
    | 0 | 0.9900179496 | 0.9950965752 |
    | 1 | 0.9944645367 | 0.9985450865 |
    | 2 | 1.0024407362 | 0.9963426169 |
    | 3 | 0.9900128097 | 0.9943701145 |
    | 4 | 0.9834876217 | 0.9936131329 |
    ranks are66 and the one new residual column is retained in all folds.
    augmented-solve discrepancy2.6645352591003757e-15). Network exploration and
    The1B reference corpus is different from the official aggregate; no official
    Gaussian epsilon and the stored t/s gap. It does not add K31/K22, fourth-to-
    Native35/100/35 qualification preserves all16 parent output bytes and
    All repeated feature arrays are byte-exact. Feature bills are269441344,
    267997376,269441344F, approximately0.07% overhead. Small FNP tensor/gap/read
    certificates pass with maximum scaled error8.660606953814209e-16. Parent
    Twenty independently cold TRAIN records include reused qualification35.
    There were22 full predictions in total, including qualification100/repeated35.
    All22 supervised stages completed and retired; total capture bill
    8744037044961F, peak RSS6672793600B,
    zero added swap. The information score includes measured featureF and a2MF

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/shared_pair_fourth_feature_v1/capture.py](code/shared_pair_fourth_feature_v1/capture.py)
- [code/shared_pair_fourth_feature_v1/capture_all.py](code/shared_pair_fourth_feature_v1/capture_all.py)
- [code/shared_pair_fourth_feature_v1/feature_fnp.py](code/shared_pair_fourth_feature_v1/feature_fnp.py)
- [code/shared_pair_fourth_feature_v1/information.py](code/shared_pair_fourth_feature_v1/information.py)
- [code/shared_pair_fourth_feature_v1/unit.py](code/shared_pair_fourth_feature_v1/unit.py)
- [code/lazy_rotation_objective_v1/candidate.py](code/lazy_rotation_objective_v1/candidate.py)
- [code/run_guarded.py](code/run_guarded.py)

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L8](RESULT.md#L8)
- [RESULT.md#L9](RESULT.md#L9)
- [RESULT.md#L10](RESULT.md#L10)

<details>
<summary>All result files and records (3)</summary>

- [RESULT.md](RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-20-shared-pair-fourth-feature-improves-adjusted-error-0-8246384) · [Copy provenance](../../../PROVENANCE.json)

