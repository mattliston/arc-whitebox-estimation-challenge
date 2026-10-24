# Gaussian-shadow disagreement: closed

2026-09-06T10:00Z · experiment · rejected · measured

All nine small tests exactly matched the established zero-cumulant Gaussian map. Width-1024 smoke ranks 35/100 reproduced the entire shadow exactly. 2026-09-06 03:00 PDT, Phase 2.

Recorded measurement:

    All nine small tests exactly matched the established zero-cumulant Gaussian map. Width-1024 smoke ranks 35/100 reproduced the entire shadow exactly. All 24 pilot rows preserved the previously certified bank scalar inputs and complete output bytes. Rows were frozen without labels; 24-row generation took 6.783 seconds using two low-priority workers. One matched 40-epoch final-only fit on 20 training nets, with ranks 37/39/42/48 held out for this fit, used existing 1e9-sample final means. Those four nets are reused research holdouts, not fresh competition nets. | Rank | Current bank raw | Zero-extra control raw | Shadow-source raw | |---|---:|---:|---:| | 37 | 2.1281180954e-8 | 2.1433378428e-8 | 2.1144534821e-8 | | 39 | 1.7861628910e-8 | 1.7952818048e-8 | 1.8170240428e-8 | | 42 | 2.3224433217e-8 | 2.3265062914e-8 | 2.2867032417e-8 | | 48 | 2.2439381374e-8 | 2.2438353077e-8 | 2.1937080957e-8 | Pooled source/bank raw ratio 0.9918905411; source/control 0.9885917468; 3/4 bank wins. Required ratios were <=0.65 and <=0.70 with >=3 wins. Thus a 0.81% raw improvement is nowhere near enough to justify the added covariance trajectory. No fitted checkpoint/seed was selected. Maximum NumPy/MPS final-delta discrepancy was 4.366e-11. Fit/evaluation time 1.227s; peak host RSS 445,333,504 bytes (not whole-system memory).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/train_final_variance_source_probe_v1.py](code/train_final_variance_source_probe_v1.py)
- [code/compact_final_head_v1.py](code/compact_final_head_v1.py)
- [code/train_final_cohort_source_probe_v1.py](code/train_final_cohort_source_probe_v1.py)
- [code/train_gaussian_shadow_disagreement_probe_v1.py](code/train_gaussian_shadow_disagreement_probe_v1.py)

## Evidence

- [GAUSSIAN_SHADOW_DISAGREEMENT_V1_CLOSED.md#L1](GAUSSIAN_SHADOW_DISAGREEMENT_V1_CLOSED.md#L1)
- [GAUSSIAN_SHADOW_DISAGREEMENT_V1_CLOSED.md#L10](GAUSSIAN_SHADOW_DISAGREEMENT_V1_CLOSED.md#L10)
- [GAUSSIAN_SHADOW_DISAGREEMENT_V1_CLOSED.md#L21](GAUSSIAN_SHADOW_DISAGREEMENT_V1_CLOSED.md#L21)

<details>
<summary>All result files and records (3)</summary>

- [GAUSSIAN_SHADOW_DISAGREEMENT_V1_CLOSED.md](GAUSSIAN_SHADOW_DISAGREEMENT_V1_CLOSED.md) · record
- [gaussian_shadow_disagreement_probe_verdict_v1.json](gaussian_shadow_disagreement_probe_verdict_v1.json) · result
- [GAUSSIAN_SHADOW_DISAGREEMENT_DESIGN_V1.md](GAUSSIAN_SHADOW_DISAGREEMENT_DESIGN_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-gaussian-shadow-disagreement-closed) · [Copy provenance](../../../PROVENANCE.json)

