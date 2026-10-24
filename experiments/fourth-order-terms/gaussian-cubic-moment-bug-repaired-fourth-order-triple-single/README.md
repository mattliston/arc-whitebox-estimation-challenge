# Gaussian cubic moment bug repaired: fourth-order triple-single plane correlation rose 0.33 to 0.998

2026-09-04 · experiment · rejected · measured

One-step certificate with truth POST planes (mc_post_planes_mps.py, 4M): from truth pre inputs, fourth_emission_1024 gives fourth-cumulant two-pair plane (K22) (rel 7-9%, corr 0.998), k4 (1%), third-cumulant repeated-index plane (K21) (1.5%) right, but fourth-cumulant triple-single plane (K31) at ratio 7.8-11 with corr 0.33 because mu31 used the Gaussian E[phi^3] while every other moment carried Edgeworth corrections; the connected (3,1) plane is 8-9% of its disconnected part, so a first-order-in-kappa3 moment error is amplified 10x. With FOURTH_R3FIX=1 (r3 = c3[0] + (k3/6)/sd^3 c3[3] + (k4/24)/sd^4 c3[4] + (k3^2/72)/sd^6 c3[6]): ratio 1.000, corr 0.998 / 0.995, rel 6-10% at L8 / L12. Same fix as teammate A's T-49 g24fix.

Recorded measurement:

    One-step certificate with truth POST planes (mc_post_planes_mps.py, 4M): from truth pre inputs, fourth_emission_1024 gives K22 (rel 7-9%, corr 0.998), k4 (1%), K21 (1.5%) right, but K31 at ratio 7.8-11 with corr 0.33 because mu31 used the Gaussian E[phi^3] while every other moment carried Edgeworth corrections; the connected (3,1) plane is 8-9% of its disconnected part, so a first-order-in-kappa3 moment error is amplified 10x. With FOURTH_R3FIX=1 (r3 = c3[0] + (k3/6)/sd^3 c3[3] + (k4/24)/sd^4 c3[4] + (k3^2/72)/sd^6 c3[6]): ratio 1.000, corr 0.998 / 0.995, rel 6-10% at L8 / L12. Same fix as teammate A's T-49 g24fix.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g29_bond_kernel_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/mc_post_planes_mps.py](code/mc_post_planes_mps.py)

## Evidence

- [LANE_g3_tracker.md#L699](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L699)
- [FOURTH_SPLIT_rank0035.json#L47](FOURTH_SPLIT_rank0035.json#L47)

<details>
<summary>All result files and records (2)</summary>

- [LANE_g3_tracker.md](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record
- [FOURTH_SPLIT_rank0035.json](FOURTH_SPLIT_rank0035.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-gaussian-cubic-moment-bug-repaired-fourth-order-triple-single) · [Copy provenance](../../../PROVENANCE.json)

