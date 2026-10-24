# True norm-coupling matrix raises (3,1) plane accuracy from a third to a half of kink-row energy

2026-10-02T09:00Z · finding · informational · measured

GPU Monte Carlo of the full matrix Ntilde_ij = Cov(z_i z_j, |z|^2) - 2 (C^2)_ij on network 35 (2 x 8M samples, layers 5-15): as (3/n) Ntilde_ij with nothing fitted it explains the true (3,1) fourth-cumulant plane (K31) on the kink rows with R2 0.50-0.53 where the card's isotropic form gives 0.31-0.41 (fitted coefficient 0.92-0.97 of 3/n); as a feed into births it adds 6-7 points beyond the exact hub-local terms (not 10: the two overlap). Its spectrum minus the isotropic forms is neither diagonal nor low rank (rank 16 keeps 55-78%, rank 64 85-95%), and the free column weights hold more of the feed than matrix plus hub-local terms, so the largest unexplained part of the feed is carried content along the bank's own columns (fourth chaos).

Recorded measurement:

    K31 kink rows R2 layers 5 / 9 / 11 / 13: isotropic 0.41 / 0.38 / 0.35 / 0.31, (3/n) Ntilde 0.52 / 0.53 / 0.50 / 0.50 (noise ceiling 0.87 / 0.96 / 0.97 / 0.98); feed share: isotropic 0.27 / 0.21 / 0.23 / 0.21, Ntilde v_k / tr C 0.34 / 0.30 / 0.33 / 0.33, hub-local 0.34, matrix + hub-local 0.40 / 0.41 (layers 9 / 13), free weights 0.49 / 0.60; D = Ntilde minus isotropic forms 16-20% of energy, rank 1 23-27%, rank 4 33-58%, rank 16 55-78%, rank 64 85-95%; block agreement 0.993-0.999

## Programs

- [../2026-10-02-norm-coupling-contraction-explains-36-46-per-unit-fourth/t19_second_chaos_mc.py](../2026-10-02-norm-coupling-contraction-explains-36-46-per-unit-fourth/t19_second_chaos_mc.py)
- [t19b_analysis.py](t19b_analysis.py)
- [t19c_feed_combined.py](t19c_feed_combined.py)
- [t19d_spectrum.py](t19d_spectrum.py)
- [../../../experiments/moment-propagation/second-order-skew-correction-worsens-error-learned-correction-but/code/opus_reexam_20261001_v1/physicist/pcard_light.py](../../../experiments/moment-propagation/second-order-skew-correction-worsens-error-learned-correction-but/code/opus_reexam_20261001_v1/physicist/pcard_light.py)
- [../2026-10-01-carried-content-accounts-5079-third-order-kink-plane-error/t2_lib.py](../2026-10-01-carried-content-accounts-5079-third-order-kink-plane-error/t2_lib.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L207](../../TEAM_LOG_2026-10-01_to_10-22.md#L207)
- [TEAM_LOG_2026-10-01_to_10-22.md#L211](../../TEAM_LOG_2026-10-01_to_10-22.md#L211)
- [TEAM_LOG_2026-10-01_to_10-22.md#L218](../../TEAM_LOG_2026-10-01_to_10-22.md#L218)
- [RESULT.md#L1244](../2026-10-01-kink-row-third-order-terms-account-6374-one-step/RESULT.md#L1244)

<details>
<summary>All result files and records (3)</summary>

- [t19b_analysis.log](t19b_analysis.log) · log
- [RESULT.md](../2026-10-01-kink-row-third-order-terms-account-6374-one-step/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-true-norm-coupling-matrix-raises-3-1-plane-accuracy) · [Copy provenance](../../../PROVENANCE.json)

