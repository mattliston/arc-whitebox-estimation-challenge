# Norm conditioning removes 87–90% of skew variance and mean kurtosis while leaving kurtosis spread

2026-10-01T19:20Z · finding · informational · measured

Monte Carlo with 16M samples in 16 norm bins on networks 100, 37, 39 (and 35) measured how much of the per-unit third- and fourth-cumulant content is explained by conditioning on the activation norm (the gain mode). At layer 15 norm conditioning removes 87-90% of the skew variance and the mean kurtosis, while the kurtosis spread across units stays. A later correction by the lead states that the gain mode accounts for about two thirds to four fifths of the mean kurtosis, part of the zero coming from the fixed-norm constraint within bins.

Recorded measurement:

    layer 15: skew variance -87..-90%, mean kurtosis removed; kurtosis spread stays 0.013-0.015 of 0.015-0.016; correction: within-bin mean kurtosis -0.0057 at layer 0, 0.012-0.019 remains at layers 6-15 on network 35; runtimes 508 / 531 / 638 s

## Programs

- [e1_gain_conditioning_mc.py](e1_gain_conditioning_mc.py)
- [e1_analyze.py](e1_analyze.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1](../../TEAM_LOG_2026-10-01_to_10-22.md#L1)
- [TEAM_LOG_2026-10-01_to_10-22.md#L10](../../TEAM_LOG_2026-10-01_to_10-22.md#L10)
- [TEAM_LOG_2026-10-01_to_10-22.md#L28](../../TEAM_LOG_2026-10-01_to_10-22.md#L28)

<details>
<summary>All result files and records (3)</summary>

- [e1_analyze_r35.txt](e1_analyze_r35.txt) · result
- [e1_analyze_r100.txt](e1_analyze_r100.txt) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-01-norm-conditioning-removes-8790-skew-variance-mean-kurtosis-while) · [Copy provenance](../../../PROVENANCE.json)

