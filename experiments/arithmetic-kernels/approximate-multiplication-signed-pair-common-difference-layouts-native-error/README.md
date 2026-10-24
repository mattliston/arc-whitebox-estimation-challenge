# Approximate multiplication with signed-pair and common/difference layouts: native error rises by about 4.3%

2026-10-17T00:21:08Z · experiment · rejected · measured

An approximate (Bini, epsilon 2^-8) multiplication family was engineered to cut the push and read bill: a signed-pair layout that cancels the common-left antisymmetric-right error (f32 pair-difference error 0.237x, signed K3 error 0.47x on a saved late push), a pair routing repair that brought the STORE35 MSE from +3.578% to +0.128%, all-orientation packing (v4), inner-pair cancellation on the young read (v5, K21 error ratios 0.64-0.70) and a common/difference (CD) push mode (v6). Native tests rejected it: v4 on fresh networks 150/16/250 raw +11.858/+22.726/+15.790%; the CD push on STORE150 +4.3052603% MSE for -0.4889194% bill (adjusted +4.8152289%); the read-only variant pooled about -0.22% but with W3 and zb4 it read +0.76% on 8. Roundoff, not truncation, dominates and no epsilon rescues it.

Recorded measurement:

    APA signed-pair layout (saved W14.T@J13): f32 pair difference error .237485x, signed K3 error .470538x (1.13667% -> .534847%), kink K21 .438930x. Pair routing repair v3: STORE35 +3.57771% -> +0.12836% MSE, first-run bill +0.65918%. v4 fresh 150/16/250: raw +11.858/+22.726/+15.790%; isolation 150 push-only +10.884%, read-only -1.397%. v5 read-only diag 3: raw -.976/+2.013/-1.430%, pooled -.218289%. Paired eps-8 1.8210%/2.5880% signed K3/K21 error, CD-8 1.0457%/1.4037%. CD push native STORE150: U 1.5647778e-8 vs 1.5001907e-8 = +4.3052603%, bill +.4889194%, adjusted +4.8152289%. ReadonlyBini+W3+zb4 fixed 8: raw +.759515% SE .52343, 2/8.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_crunch_apa_pairlayout_v1/bini_actual_factory_v6.py](code/astra_crunch_apa_pairlayout_v1/bini_actual_factory_v6.py)
- [code/astra_crunch_apa_pairlayout_v1/fragment_pair_routing_v1.py](code/astra_crunch_apa_pairlayout_v1/fragment_pair_routing_v1.py)
- [code/astra_crunch_apa_readpair_v1/bini_actual_factory_v7.py](code/astra_crunch_apa_readpair_v1/bini_actual_factory_v7.py)
- [code/astra_crunch_apa_pairmode_v1/candidate_bini_cdpush_e8_readpair_v6.py](code/astra_crunch_apa_pairmode_v1/candidate_bini_cdpush_e8_readpair_v6.py)
- [code/astra_crunch_apa_pairlayout_v1/candidate_bini_pair_combined_e8_v4.py](code/astra_crunch_apa_pairlayout_v1/candidate_bini_pair_combined_e8_v4.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4532](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4532)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4536](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4536)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4538](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4538)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4539](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4539)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4541](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4541)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4542](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4542)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4543](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4543)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4545](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4545)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4547](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4547)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4548](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4548)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4549](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4549)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4551](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4551)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4553](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4553)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4554](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4554)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4555](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4555)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4559](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4559)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4560](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4560)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4561](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4561)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4574](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4574)

<details>
<summary>All result files and records (5)</summary>

- [SOURCE_READY_v6.json](SOURCE_READY_v6.json) · record
- [ASSESSMENT_v1.json](ASSESSMENT_v1.json) · result
- [COMPARE_CDPUSH150_v1.json](COMPARE_CDPUSH150_v1.json) · result
- [CLOSURE_CDPUSH_v1.json](CLOSURE_CDPUSH_v1.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-17-approximate-multiplication-signed-pair-common-difference-layouts-native-error) · [Copy provenance](../../../PROVENANCE.json)

