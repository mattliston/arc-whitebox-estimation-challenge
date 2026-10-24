# Derived-host score projection retracted after counting all fourth-order channels

2026-09-04 · experiment · informational · measured

RETRACTION OF THE 1.45e-8 ESTIMATE (15:40): the chain without its fourth-cumulant triple-single plane (K31) channel costs +8% raw (r35 3.35e-8, r100 3.87e-8 fitted transport-energy residual correction (F2J) off); the chain supplying ALL fourth-order objects (fourth-cumulant canonical-polyadic carrier (CP4) + quartic star removable) costs +35-40% raw (T-91 runs chain_all: 4.18e-8 / 5.20e-8). Priced honestly: (a) chain fourth-cumulant two-pair plane (K22) + CP4/quartic kept + f32: util ~0.71, refit F2J-on 2.88/2.90e-8 -> adjusted ~2.05e-8; (b) chain-all + f32: util ~0.52, raw 4.2-5.2e-8 -> ~2.2e-8; both WORSE than the bank (1.78e-8). The chain's host gain is the same fourth-order fix the deployed operator already performs on the old host (in-span, T-73).

Code availability: The shared retrofit instrument and fourth-order chain implementation were recovered. This entry retracts a derived cost/error projection; no separate program for the projection itself was identified.

Recorded measurement:

    - T-92 RETRACTION OF THE 1.45e-8 ESTIMATE (15:40): the chain without its K31 channel costs +8% raw (r35 3.35e-8, r100 3.87e-8 F2J off); the chain supplying ALL fourth-order objects (CP4 + quartic star removable) costs +35-40% raw (T-91 runs chain_all: 4.18e-8 / 5.20e-8). Priced honestly: (a) chain K22 + CP4/quartic kept + f32: util ~0.71, refit F2J-on 2.88/2.90e-8 -> adjusted ~2.05e-8; (b) chain-all + f32: util ~0.52, raw 4.2-5.2e-8 -> ~2.2e-8; both WORSE than the bank (1.78e-8). The chain's host gain is the same fourth-order fix the deployed operator already performs on the old host (in-span, T-73). Only the f32 pack banks (1.70e-8, -4.9%, below the bar). No chain-card package. The 4e-9 target has no measured block in the corpus; open threads: SOL (3,1)-plane ask, transported state corrector gate (feat dumps on 8 nets exist: feat_derived_retro-*.npz).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_g20_emission_retrofit_v1/estimator_retro.py](code/public_g20_emission_retrofit_v1/estimator_retro.py)
- [code/public_g20_emission_retrofit_v1/run_retro_screen.py](code/public_g20_emission_retrofit_v1/run_retro_screen.py)
- [code/public_g28_mech_features_v1/mechfeat15.py](code/public_g28_mech_features_v1/mechfeat15.py)
- [code/public_g29_bond_kernel_v1/k22_chain.py](code/public_g29_bond_kernel_v1/k22_chain.py)

## Evidence

- [LANE_g3_tracker.md#L764](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L764)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-derived-host-score-projection-retracted-after-counting-all-fourth) · [Copy provenance](../../../PROVENANCE.json)

