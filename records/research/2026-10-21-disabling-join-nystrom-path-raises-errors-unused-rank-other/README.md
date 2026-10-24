# Disabling the join Nystrom path raises errors; unused rank and other arithmetic tails are ineffective

2026-10-21T22:43:48Z · finding · informational · measured

The recorded sweep summary rejects the disabled Nystrom path and identifies parameter overrides without an effective deployed reader. These checks have no claimed numeric score when the source did not execute successfully.

Recorded measurement:

    - **Closed switches:** `_ASTRA_SM40_ON = False` fails at setup (restored-after-failure) - not a supported switch; `_SMX_ON = False` = +0.50% adj, wall +1.4% (Smirnov-X is itself a wall optimisation); `_JOIN_NYSTROM = False` raises at every join; `_SLR_RANK` has no reader (dead constant); rank0only shape filter = cap4 twin.
    Today's VM32 sweep ran 60-odd native 32-network evaluations of the deployed #337506 science file (sha 49bd2ba3...) on the 48-core GCP box, every variant an appended tail on the exact deployed bytes, scored by the cross-half U at the final layer times metered FLOPs (score_pair_v3.py), accuracy and bill paired against the lone reference run and, for wall, against a control of the same file launched concurrently. What moved: the Smirnov (SM40) read cap, a predict-time wrapper that returns no plan after N reads, is the only wall lever found: cap3 (the fired #337565 tail) -11% VM wall at +0.36% bill, cap4 -5% at +0.18%, with accuracy unchanged; dropping the 9 largest SM40 plans (m*k*n >= 8.0e9) adds -4.7% wall for +0.07% bill on its own and, combined with cap4, matches cap3 on the slow-10 ranks at 0.16% less bill (cap4+8e9 vs cap3, concurrent: adj -0.118%, bill -0.160%), which became the final margin tail; the lead's on-policy first128 refit was worth about -0.3% adjusted at any strength from 0.5x to 2.0x (s1.5 best at -0.385%, se 0.16). What did not move: the size-threshold family below 5.5e9 (-5.6% wall at +0.44% bill once measured concurrently; the earlier -22% was load), 7.0e9 under cap4 (+0.14% bill, no wall), cap5 variants (worse bill, no wall), SM40 off (setup fails), SMX off (+0.50%), the Nystrom switch (raises), LB3/HCF/square-depth tails (byte-identical no-ops), the cohort-age cuts (K4L age 1 +0.92%, F4C age 1 +2.31%, NP k3 age 1 fails), every threshold and strength dial (row cut 2.5/3.5, kc +-256 and the kc sets, YNT/OTT, k4 bands, NP strength, K31R late rows +0.17%), with the single apparent dial win (_K4L_C x1.05, -0.144% all32) inside the +-0.09% per-run noise, flanked by -0.076% at x0.95 and -0.033% at x1.10, and not transferring onto the readout stack (-0.1% worse when stacked). Measurement laws for next time: predictions, FLOPs, requests and puts are byte-identical across any load, but wall jitters +-20% per rank and +-50% in the mean with concurrency, so only the 32-rank mean against a concurrent control is readable; the admission guard retires runs when MemAvailable drops under its 19 GB reserve, so four 12-job launches is the box's ceiling. Box utilization during the sweep: 80-93% user CPU with three to four launches in flight, 50-67% with two (each launch is 12 Python-bound server processes with 4 BLAS threads), 1-2% when idle between the lead's holds.

## Programs

- [cand_lb3_off.py](cand_lb3_off.py)

## Evidence

- [VM_SWEEP_RESULTS_2026-10-21.md#L109](../../VM_SWEEP_RESULTS_2026-10-21.md#L109)

<details>
<summary>All result files and records (1)</summary>

- [VM_SWEEP_RESULTS_2026-10-21.md](../../VM_SWEEP_RESULTS_2026-10-21.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-disabling-join-nystrom-path-raises-errors-unused-rank-other) · [Copy provenance](../../../PROVENANCE.json)

