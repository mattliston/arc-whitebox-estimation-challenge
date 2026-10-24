# Same-sign terminal birth-slot cubic feature (65 columns) explains 6.2%/7.9% of the affine residual vs the 20% gate

2026-09-20 · experiment · rejected · measured

A formal tangent feature that adds the same coefficient to both cubic slots of the last newborn (layer 14) births, removes the complete coordinate-repeated plane and contracts through W15 with three power-product matrices gives 65 columns a*[H14,1]; same-network least-squares oracles on nets 35/100 show the source accounts for only 6.2369%/7.9301% of the affine-head residual (required 20%) and total privileged reductions of 16.6301%/19.2061% (required 30%), so no shared fit or candidate followed. The record also corrects a stale assumption: the older A1-preserving common-cubic physical fit had already run and failed CHECK4 (adjusted 1.0288877817, 0 wins).

Recorded measurement:

    extended/parent 0.833698802476 (net 35) / 0.807938720482 (net 100) vs <=0.70; extended/affine 0.937631134111 / 0.920699089095 vs <=0.80; parent raw 1.87060284901e-8 / 1.84245341753e-8; affine oracle 1.66325466208e-8 / 1.61680344245e-8; extended oracle 1.55951935513e-8 / 1.4885894567e-8; 544 selected last-birth columns on both nets

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/terminal_common_slot_information_v1/capture.py](code/terminal_common_slot_information_v1/capture.py)
- [code/terminal_common_slot_information_v1/prepare.py](code/terminal_common_slot_information_v1/prepare.py)
- [code/terminal_common_slot_information_v1/compute_features.py](code/terminal_common_slot_information_v1/compute_features.py)
- [code/terminal_common_slot_information_v1/score.py](code/terminal_common_slot_information_v1/score.py)
- [code/terminal_common_slot_information_v1/unit.py](code/terminal_common_slot_information_v1/unit.py)
- [code/lazy_rotation_objective_v1/candidate.py](code/lazy_rotation_objective_v1/candidate.py)
- [code/run_guarded.py](code/run_guarded.py)
- [code/terminal_common_slot_information_v1/common.py](code/terminal_common_slot_information_v1/common.py)
- [code/terminal_common_slot_information_v1/feature.py](code/terminal_common_slot_information_v1/feature.py)
- [code/terminal_common_slot_information_v1/tangent.py](code/terminal_common_slot_information_v1/tangent.py)
- [code/terminal_covariance_information_v1/capture.py](code/terminal_covariance_information_v1/capture.py)
- [code/terminal_covariance_tangent_readout_v1/tangent.py](code/terminal_covariance_tangent_readout_v1/tangent.py)

## Evidence

- [NEGATIVE_RESULT.md#L3](NEGATIVE_RESULT.md#L3)
- [NEGATIVE_RESULT.md#L9](NEGATIVE_RESULT.md#L9)
- [NEGATIVE_RESULT.md#L10](NEGATIVE_RESULT.md#L10)
- [NEGATIVE_RESULT.md#L12](NEGATIVE_RESULT.md#L12)
- [NEGATIVE_RESULT.md#L13](NEGATIVE_RESULT.md#L13)
- [NEGATIVE_RESULT.md#L29](NEGATIVE_RESULT.md#L29)
- [NEGATIVE_RESULT.md#L30](NEGATIVE_RESULT.md#L30)

<details>
<summary>All result files and records (4)</summary>

- [NEGATIVE_RESULT.md](NEGATIVE_RESULT.md) · result
- [SOURCE_REVIEW.md](SOURCE_REVIEW.md) · record
- [result.json](result.json) · result
- [PROTOCOL.md](PROTOCOL.md) · config

</details>

[Research graph](../../../graph/index.html#node=2026-09-20-same-sign-terminal-birth-slot-cubic-feature-65-columns) · [Copy provenance](../../../PROVENANCE.json)

