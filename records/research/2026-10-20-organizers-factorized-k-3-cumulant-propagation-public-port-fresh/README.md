# Organizers' factorized K=3 cumulant propagation (public port) on a fresh 1024x16 net: U 4.43e-8 vs card 1.32e-8

2026-10-20T02:33:00Z · experiment · rejected · measured

The organizers' paper (arXiv 2605.05179: cumulant propagation to order K with power cumulants, an odd-K trace term, and a factorized K>=3 representation of size n^(K-1) costing about L^2 n^K) and a public NumPy port were run on fresh seed 7100005 at K=2 and K=3 against the 64M truth to test the hypothesis that the leaders run the published scheme at the challenge budget: the vanilla K=3 factored method is 3.4x worse than the card (K=2 far worse), so the leaders' -17% is their own engineering. The paper's cost law puts factored K=3 at about 2.7e11 flops at 1024x16, util about 0.12, which is the leaders' utilisation.

Code availability: The log names scratchpad ext/run_kprop_fresh_v1.py, but that exact runner is absent from the frozen repository. No different K-propagation program was substituted; this result remains log-only.

Recorded measurement:

    kprop K=3 factored on 7100005: final U 4.43e-8 (per layer 2.4e-9 at L3, 1.2e-8 at L7, 2.9e-8 at L11) vs card 1.32e-8; K=2 3.7e-6; phase-1 port 256x8 k_max=3 MSE 1.2e-6 at 1.7e10 flops; cost law L^2 n^3 = 2.7e11 flops = util ~0.12

The measured result is retained here. A separate executable program was not identified; this entry is indexed as a result record rather than a code directory.

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6113](../../TEAM_LOG_2026-10-01_to_10-22.md#L6113)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6115](../../TEAM_LOG_2026-10-01_to_10-22.md#L6115)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6117](../../TEAM_LOG_2026-10-01_to_10-22.md#L6117)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-organizers-factorized-k-3-cumulant-propagation-public-port-fresh) · [Copy provenance](../../../PROVENANCE.json)

