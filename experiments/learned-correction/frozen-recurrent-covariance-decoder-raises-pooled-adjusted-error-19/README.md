# Frozen recurrent covariance decoder raises pooled adjusted error 19.7306%

2026-09-21 · experiment · rejected · measured

Three saved symmetric 65-by-65 matrices decoded a separate recurrent history state and replaced the existing covariance correction at layers8-14.

Recorded measurement:

    Replacing the bank's covariance edge at layers 8 through 14 with the stored
    history decoder increases pooled adjusted error by **19.7306%** on networks
    36 and 38. Both networks regress by more than the permitted 5%. This fixed
    | 36 | 1.983987470157e-08 | 2.250638762958e-08 | 420159401355 | 425163919053 | 1.147913558190 |
    | 38 | 1.522485110651e-08 | 1.899279269174e-08 | 414511808707 | 419516326405 | 1.262547520987 |
    The pooled adjusted ratio is **1.197306034380315**. Candidate cold FLOPs increase
    by 5,004,517,698 on each network, about 1.2%. The score uses each arm's own
    complete cold bill and max(0.1, FLOPs / 2^41). Independent math.fsum checks
    agree with the raw MSE calculations within 1.75e-16 relative error. These
    P0 encoder on its current physical inputs. Its three saved symmetric 65 by
    65 matrices form a covariance correction modulated by 1, R, and R squared.
    - Three deterministic kernel fixtures passed NumPy FP32/FP64, permutation,
    - Disabled calls on 35, 100, 35 match all 16 original output rows and each
      repeated 36 and separate cold 38 reproduce output and capture bytes.
    - All 60 saved recurrent states match the original NumPy equations exactly.
      All 28 source subsets match NumPy FP32 exactly, and FP64 within
      9.175790200699e-07 relative error.
      1e-10. Maximum full-source relative asymmetry is
      1.712759068666e-05,
      within the predeclared 3e-5 tolerance. Native numeric caches retired.
    2.171328543497e-07
    2,872,944,342,572 measured FLOPs. Peak sampled group RSS was
    6,503,514,112 bytes and peak process RSS was
    6,503,530,496 bytes. Added swap was zero. About
    41.213 MB of artifacts were

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260919_v1/frozen_pair_history_endpoint_v1/build.py](code/codex_resume_20260919_v1/frozen_pair_history_endpoint_v1/build.py)
- [code/codex_resume_20260919_v1/frozen_pair_history_endpoint_v1/candidate.py](code/codex_resume_20260919_v1/frozen_pair_history_endpoint_v1/candidate.py)
- [code/codex_resume_20260919_v1/frozen_pair_history_endpoint_v1/common.py](code/codex_resume_20260919_v1/frozen_pair_history_endpoint_v1/common.py)
- [code/codex_resume_20260919_v1/frozen_pair_history_endpoint_v1/finalize.py](code/codex_resume_20260919_v1/frozen_pair_history_endpoint_v1/finalize.py)
- [code/codex_resume_20260919_v1/frozen_pair_history_endpoint_v1/legacy_replay.py](code/codex_resume_20260919_v1/frozen_pair_history_endpoint_v1/legacy_replay.py)
- [code/codex_resume_20260919_v1/frozen_pair_history_endpoint_v1/native.py](code/codex_resume_20260919_v1/frozen_pair_history_endpoint_v1/native.py)
- [code/codex_resume_20260919_v1/frozen_pair_history_endpoint_v1/preflight.py](code/codex_resume_20260919_v1/frozen_pair_history_endpoint_v1/preflight.py)
- [code/codex_resume_20260919_v1/frozen_pair_history_endpoint_v1/prepare.py](code/codex_resume_20260919_v1/frozen_pair_history_endpoint_v1/prepare.py)
- [code/codex_resume_20260919_v1/frozen_pair_history_endpoint_v1/production.py](code/codex_resume_20260919_v1/frozen_pair_history_endpoint_v1/production.py)
- [code/codex_resume_20260919_v1/frozen_pair_history_endpoint_v1/qualify.py](code/codex_resume_20260919_v1/frozen_pair_history_endpoint_v1/qualify.py)
- [code/codex_resume_20260919_v1/frozen_pair_history_endpoint_v1/reference.py](code/codex_resume_20260919_v1/frozen_pair_history_endpoint_v1/reference.py)
- [code/codex_resume_20260919_v1/frozen_pair_history_endpoint_v1/run_native.py](code/codex_resume_20260919_v1/frozen_pair_history_endpoint_v1/run_native.py)

<details>
<summary>Additional programs and dependencies (18)</summary>

- [code/codex_resume_20260919_v1/frozen_pair_history_endpoint_v1/run_prepare.py](code/codex_resume_20260919_v1/frozen_pair_history_endpoint_v1/run_prepare.py)
- [code/codex_resume_20260919_v1/frozen_pair_history_endpoint_v1/score.py](code/codex_resume_20260919_v1/frozen_pair_history_endpoint_v1/score.py)
- [code/codex_resume_20260919_v1/frozen_pair_history_endpoint_v1/seal_native.py](code/codex_resume_20260919_v1/frozen_pair_history_endpoint_v1/seal_native.py)
- [code/codex_resume_20260919_v1/frozen_pair_history_endpoint_v1/unit.py](code/codex_resume_20260919_v1/frozen_pair_history_endpoint_v1/unit.py)
- [code/astra_phase2_reexamination_v1/generate_covariance_innovation_probe_v1.py](code/astra_phase2_reexamination_v1/generate_covariance_innovation_probe_v1.py)
- [code/astra_phase2_reexamination_v1/generate_learned_pair_history_row_v1.py](code/astra_phase2_reexamination_v1/generate_learned_pair_history_row_v1.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_binary_compact.py](code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [code/astra_phase2_reexamination_v1/radial_cubic.py](code/astra_phase2_reexamination_v1/radial_cubic.py)
- [code/astra_phase2_reexamination_v1/radial_features.py](code/astra_phase2_reexamination_v1/radial_features.py)
- [code/astra_phase2_reexamination_v1/radial_gnn_numpy.py](code/astra_phase2_reexamination_v1/radial_gnn_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)
- [code/astra_phase2_reexamination_v1/train_covariance_innovation_probe_v1.py](code/astra_phase2_reexamination_v1/train_covariance_innovation_probe_v1.py)
- [code/astra_phase2_reexamination_v1/train_learned_pair_history_v1.py](code/astra_phase2_reexamination_v1/train_learned_pair_history_v1.py)
- [code/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py](code/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py)
- [code/codex_resume_20260919_v1/run_guarded.py](code/codex_resume_20260919_v1/run_guarded.py)

</details>

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L4](RESULT.md#L4)
- [RESULT.md#L5](RESULT.md#L5)
- [RESULT.md#L12](RESULT.md#L12)
- [REEXAMINATION_INDEX_20260919.md#L572](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L572)
- [REEXAMINATION_INDEX_20260919.md#L575](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L575)

<details>
<summary>All result files and records (6)</summary>

- [RESULT.md](RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result
- [score.json](score.json) · result
- [summary.json](summary.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-21-frozen-recurrent-covariance-decoder-raises-pooled-adjusted-error-19) · [Copy provenance](../../../PROVENANCE.json)

