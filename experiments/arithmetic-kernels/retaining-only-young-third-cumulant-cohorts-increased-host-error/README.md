# Retaining only young third-cumulant cohorts increased host error 3.6 to 7.5 times

2026-09-04 · experiment · informational · measured

AGE-TRUNCATED BANK PROBE (plain card, r35, emulated via RETRO_LOWRANK_CP3 with rank 0 for old cohorts): youngest 3 cohorts -> fitted transport-energy residual correction (F2J)-off host 6.42e-7 (7.5x the full 8.62e-8), youngest 5 -> 3.09e-7 (3.6x). The third-order plane content is NOT recent-dominated (the ladder's "born in the last pair map" was the variance ERROR's birth, not the plane's content); with the closure at ~90% the loop would still be 2-4x the full card's raw for ~30-40% less cost -> worse adjusted. Design fails its necessary condition; no round.

Code availability: The shared retrofit instrument retains the cohort-truncation control. The tracker states the rank-zero schedules and measurements, but a dedicated run script for this exact comparison was not identified.

Recorded measurement:

    - T-122 DESIGN NOTE (no compute; for Matt's decision): what a cheap third-order carry must satisfy, from the receipts. (1) The card USES only planes (K21, and the k3 diagonal) and the fourth-order planes; the tensor is carried only to produce the next layer's planes. (2) The plane push needs the tensor's factors (K21_ab of a pushed CP term = (W^T p)_a^2 (W^T q)_b); the youngest cohorts' transport factors are full rank (T-95b), so any factor-form carry costs O(rank x 1024^2) per layer - the arena is already the cheapest dispatch (T-97). (3) The competitor's util 0.22 ~ our Gaussian card (0.16-0.18) + ~0.04, i.e. his third-order content costs ~1e11 flops total ~ one or two 1024^3 matmuls per layer: he does NOT carry a factor bank. Candidates for a 1-2 matmul/layer third-order closure: (a) a Gaussian recursion with an analytic per-layer third-order correction of the MEAN only (k3 diagonal via the cubic star push, 9% - already tested as 'diag k3': the covariance recursion still needs the planes, host 2.5e-6, dead); (b) a correction of the COVARIANCE recursion that uses a learned or analytic low-rank plane surrogate (the K21 plane is rank 64-128) built from the Gaussian state and the weights at each layer (not carried): K21_l ~ f(W_l, m, C) rank-r; the question is whether a per-layer surrogate (no memory across layers) can reproduce the accumulated plane - T-115i's ladder says 2/3 of the final variance error is born inside the last pair map, i.e. the plane's effect is dominated by RECENT layers -> a short-memory surrogate (last 2-3 layers' births, exact, rank 1024 each, pushed 1-3 times = 2-6 matmuls/layer) may capture most of the third-order content. This is the first falsifiable test of a new family: truncate the CP bank by AGE (keep only the youngest k cohorts; RETRO_LOWRANK_CP3 can emulate an age cutoff by setting old ranks to 0) and measure the honest host and the co-adapted loop vs k. Cost of a k-cohort bank = k x 1024 rank -> arena ~ k/15 of 45% -> k=3: ~9% util; card ~0.3 with the fourth-order machinery replaced by the closure. If the co-adapted loop at k=3 holds within ~1.5x of the full card's raw, adjusted ~1e-8. One round (~1 h) - awaiting Matt's go.
    
    - T-122 AGE-TRUNCATED BANK PROBE (plain card, r35, emulated via RETRO_LOWRANK_CP3 with rank 0 for old cohorts): youngest 3 cohorts -> F2J-off host 6.42e-7 (7.5x the full 8.62e-8), youngest 5 -> 3.09e-7 (3.6x). The third-order plane content is NOT recent-dominated (the ladder's "born in the last pair map" was the variance ERROR's birth, not the plane's content); with the closure at ~90% the loop would still be 2-4x the full card's raw for ~30-40% less cost -> worse adjusted. Design fails its necessary condition; no round. No compute running.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_g20_emission_retrofit_v1/estimator_retro.py](code/public_g20_emission_retrofit_v1/estimator_retro.py)
- [code/public_g20_emission_retrofit_v1/run_retro_screen.py](code/public_g20_emission_retrofit_v1/run_retro_screen.py)
- [code/public_g28_mech_features_v1/mechfeat15.py](code/public_g28_mech_features_v1/mechfeat15.py)
- [code/public_g29_bond_kernel_v1/k22_chain.py](code/public_g29_bond_kernel_v1/k22_chain.py)

## Evidence

- [LANE_g3_tracker.md#L910](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L910)
- [LANE_g3_tracker.md#L911](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L911)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-retaining-only-young-third-cumulant-cohorts-increased-host-error) · [Copy provenance](../../../PROVENANCE.json)

