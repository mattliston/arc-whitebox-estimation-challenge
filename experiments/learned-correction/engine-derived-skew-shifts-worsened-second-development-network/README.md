# Engine-derived skew shifts worsened the second development network

2026-09-04 · experiment · informational · measured

ENGINE-CLASS SHIFTS FAIL r100 (operator ON, beta 0.5, independent 4M; deployed 2.779e-8): engine k3u shift 3.261e-8 (+17%), engine dv-hat + k3u 2.917e-8 (+5%); the alignment of the engine's k3 shift on r100 decays with depth (+0.82 / +0.69 / +0.56 at L8 / L12 / L14). => Matt's 2e-8 bar needs supplies aligned >~ 0.9 on EVERY net; the full-tensor engine (0.8 on r35, 0.6-0.8 on r100) is not enough even if it were deployable. The remaining route with real odds: the born fourth-order/third-order plane content via matrix-matrix multiplication (GEMM)-class block-pattern feeds (T-49: direction corr 0.95-0.97 with the full path contribution at n=32, magnitude 33-49%) with learned per-pair multipliers fit on truth planes across many nets -> planes at few-% accuracy -> the pair consumer's half-scale variance correction. Premise test next: alignment of the block-pattern feed with the actual push residual at 1024 (push_residual_rank0035_L{8,12}.npz).

Code availability: The shared retrofit instrument retains the engine-skew injection hooks. A dedicated launch script for the exact tracker comparison was not identified.

Recorded measurement:

    - T-82 ENGINE-CLASS SHIFTS FAIL r100 (operator ON, beta 0.5, independent 4M; deployed 2.779e-8): engine k3u shift 3.261e-8 (+17%), engine dv-hat + k3u 2.917e-8 (+5%); the alignment of the engine's k3 shift on r100 decays with depth (+0.82 / +0.69 / +0.56 at L8 / L12 / L14). => Matt's 2e-8 bar needs supplies aligned >~ 0.9 on EVERY net; the full-tensor engine (0.8 on r35, 0.6-0.8 on r100) is not enough even if it were deployable. The remaining route with real odds: the born fourth-order/third-order plane content via GEMM-class block-pattern feeds (T-49: direction corr 0.95-0.97 with the full path contribution at n=32, magnitude 33-49%) with learned per-pair multipliers fit on truth planes across many nets -> planes at few-% accuracy -> the pair consumer's half-scale variance correction. Premise test next: alignment of the block-pattern feed with the actual push residual at 1024 (push_residual_rank0035_L{8,12}.npz).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_g20_emission_retrofit_v1/estimator_retro.py](code/public_g20_emission_retrofit_v1/estimator_retro.py)
- [code/public_g20_emission_retrofit_v1/run_retro_screen.py](code/public_g20_emission_retrofit_v1/run_retro_screen.py)
- [code/public_g28_mech_features_v1/mechfeat15.py](code/public_g28_mech_features_v1/mechfeat15.py)
- [code/public_g29_bond_kernel_v1/k22_chain.py](code/public_g29_bond_kernel_v1/k22_chain.py)

## Evidence

- [LANE_g3_tracker.md#L724](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L724)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-engine-derived-skew-shifts-worsened-second-development-network) · [Copy provenance](../../../PROVENANCE.json)

