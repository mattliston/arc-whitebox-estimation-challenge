# 15 fitted zero-bill dials re-centred by finite differences (pair-law strength, star beta, row factors): all within 0.05%

2026-10-20T00:50:00Z · experiment · rejected · measured

The remaining fitted dials were re-centred by +-10% finite differences on 28 store ranks at zero bill: pair-law strengths (all layers), the fourth-order ages 1-2 coefficient, star beta, birth skew strength (a shallow bowl), the band-3 pair, the pair-law k3 by-layer strengths, the K31R row factor and its entry 1, the head-column factor, and the GNN knobs (iso and k21n and edge and rc are no-ops because the per-layer table overrides them; k31 is live but flat). Every optimum is within 0.05% of the fitted value, confirming the torch-fitted dials transferred to the native card without bias; the small optima were folded into the micro-stack.

Recorded measurement:

    NP_STRENGTH x0.9 +0.04% (held-out -0.09%), x1.1 +0.21%, optimum x0.974 for -0.01% (per-net oracle -0.21%); K4L_C x0.9 -0.03% / x1.1 +0.22%, optimum x0.93 for -0.04% (k4lc_m all32 -0.06%, held-out -0.13%); SS_BETA 0.80 +0.10% / 0.90 +0.05%; BSK 0.2 +0.14%, 0.5 -0.06%, 0.7 +0.15%, 1.0 +0.24%, optimum ~0.45 for -0.03%; band 3 pair x0.95 +0.59% / x1.05 +1.02%, optimum x0.993 (-0.01%); K3L by-layer x0.9 -0.04% / x1.1 +0.13%, x0.943 (-0.02%); K31R row factor x0.9 +0.57% / x1.1 +0.04% (held-out -0.08%), x1.04 (-0.05%); K31R entry 1 x0.9 +0.10%; hcfa x0.9 -0.05% (held-out -0.11%), optimum ~x0.965 (-0.03%); GN k31 x0.8 +0.17% / x1.2 -0.015%, ~x1.1 (-0.02%); GN iso/k21n/edge/rc no-ops; GBK stays 0 (torch feed +0.18% on 10-02)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_npstr_m_sci_v1.py](code/candidate_npstr_m_sci_v1.py)
- [code/candidate_npstr_p_sci_v1.py](code/candidate_npstr_p_sci_v1.py)
- [code/candidate_k4lc_m_sci_v1.py](code/candidate_k4lc_m_sci_v1.py)
- [code/candidate_k4lc_p_sci_v1.py](code/candidate_k4lc_p_sci_v1.py)
- [code/candidate_ssbeta_080_sci_v1.py](code/candidate_ssbeta_080_sci_v1.py)
- [code/candidate_ssbeta_090_sci_v1.py](code/candidate_ssbeta_090_sci_v1.py)
- [code/candidate_bsk_070_sci_v1.py](code/candidate_bsk_070_sci_v1.py)
- [code/candidate_bsk_100_sci_v1.py](code/candidate_bsk_100_sci_v1.py)
- [code/candidate_npk3lbl_m_sci_v1.py](code/candidate_npk3lbl_m_sci_v1.py)
- [code/candidate_npk3lbl_p_sci_v1.py](code/candidate_npk3lbl_p_sci_v1.py)
- [code/candidate_k31r1_m_sci_v1.py](code/candidate_k31r1_m_sci_v1.py)
- [code/candidate_k31r1_p_sci_v1.py](code/candidate_k31r1_p_sci_v1.py)

<details>
<summary>Additional programs and dependencies (7)</summary>

- [code/candidate_hcfa_m_sci_v1.py](code/candidate_hcfa_m_sci_v1.py)
- [code/candidate_hcfa_p_sci_v1.py](code/candidate_hcfa_p_sci_v1.py)
- [code/candidate_gn_isom_sci_v1.py](code/candidate_gn_isom_sci_v1.py)
- [code/candidate_gn_k31m_sci_v1.py](code/candidate_gn_k31m_sci_v1.py)
- [code/candidate_gn_k31p_sci_v1.py](code/candidate_gn_k31p_sci_v1.py)
- [code/candidate_gn_rcm_sci_v1.py](code/candidate_gn_rcm_sci_v1.py)
- [code/candidate_gn_edgem_sci_v1.py](code/candidate_gn_edgem_sci_v1.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6009](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6009)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6011](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6011)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6013](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6013)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6015](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6015)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6017](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6017)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6029](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6029)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6031](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6031)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6069](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6069)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6085](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6085)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6091](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6091)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-15-fitted-zero-bill-dials-re-centred-finite-differences) · [Copy provenance](../../../PROVENANCE.json)

