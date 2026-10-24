# Age-dependent low-rank cohort schedule doubled raw error without saving compute against the arena

2026-09-04 · experiment · rejected · measured

LR-BANK age schedule (dense ages 1-3, 256 at age 4, 128 at age>=5), host only, r35: util 0.6087 (the LR path's full-rank 0.77; the arena 0.62) but host 1.67e-7 vs 8.6e-8 full-rank: 2x worse raw for no cost gain over the arena. Old-cohort compression DEAD again (T-97 stands).

Recorded measurement:

    - T-132: reference for banks: the DEPLOYED card's grader panel on the 6 virgin nets = adjusted 1.645e-8, raw 2.712e-8 (panel_deployed_virgin.jsonl). CAP256_v2 14-net panel running. Testing the T-97 low-rank cohort bank with an AGE schedule that keeps the youngest 3 cohorts dense and compresses age 4 to rank 256, age >= 5 to rank 128 (RETRO_LR_BANK="1024,1024,1024,256,128;..."), closure ON, r35: raw vs 2.407e-8 (1e9 frame) and util vs 0.622.
    
    - T-132b LR-BANK age schedule (dense ages 1-3, 256 at age 4, 128 at age>=5), host only, r35: util 0.6087 (the LR path's full-rank 0.77; the arena 0.62) but host 1.67e-7 vs 8.6e-8 full-rank: 2x worse raw for no cost gain over the arena. Old-cohort compression DEAD again (T-97 stands).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_d11_util_halving_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/estimator_cap.py](code/estimator_cap.py)

## Evidence

- [LANE_g3_tracker.md#L940](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L940)
- [LANE_g3_tracker.md#L941](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L941)

<details>
<summary>All result files and records (2)</summary>

- [LANE_g3_tracker.md](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record
- [panel_deployed_virgin.jsonl](panel_deployed_virgin.jsonl) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-age-dependent-low-rank-cohort-schedule-doubled-raw-error) · [Copy provenance](../../../PROVENANCE.json)

