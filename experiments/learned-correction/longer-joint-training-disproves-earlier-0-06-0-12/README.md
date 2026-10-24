# Longer joint training disproves the earlier 0.06-0.12% remaining-gain projection

2026-09-25T15:34Z · experiment · informational · measured

The projection incorrectly treated physics dials as converged. Three more joint epochs delivered a confirmed native raw-error gain of 0.527%, despite narrowly missing the original small-set gate.

Recorded measurement:

    - 2026-09-25 08:34 analyst: LEARNING CURVE + S25 ATTRIBUTION (fable_analyst_v1/RESULT.md sections 41-42). (1) Training alone cannot close the gap. On a fixed card each successive epoch gains q = .21-.45 of the previous one (B2 .21, loopR3->R4 .34, arm A .45 within one cosine). From arm A's end (VALID16 1.63990e-8) the projected remaining gain is .06-.12% (.22% at q = .6). Arm A holds 80-88% of the asymptotic gain from the loopR3 end, and 90% needs 2-3 epochs. Learned-only beats joint per epoch from the same start: arm A's first epoch -0.32% vs loopR4 (joint, full 1-epoch cosine) -0.18%. Dials have converged (drift <= 0.3 GN steps); 2x width adds .08 pt; TRAIN640's 4x nets gave ~0. Recommendation: loopR5 learned-only with a 2-3 epoch cosine from arm A, expect about -0.1% VALID16. (2) S25 (watcher, 02:58): raw +4.455% = all R320; util -4.583% (R320 -4.40%, treuse+j3 -0.18%); adjusted -0.343% (21/40); R320 alone -0.164% adjusted, per-net raw -1.19 / +4.63 / +10.46%. Per-net R320 on/off ORACLE -1.10% vs S25 (20/40 keep R320), but no routing signal (the energy signal failed, section 39). S26/S27 attribution watcher armed (attrib_pair.py).
    - 2026-09-25 08:35 lead: ANALYST §42 (learning curve): each successive epoch gains q = .21-.45 of the previous; arm A already holds 80-88% of the asymptotic gain from loopR3, so continued training projects only ~-0.06..-0.12% (at most -0.22%). The learned parts are information-limited (width +.08, TRAIN640 ~0). loopR5 (joint 3 ep) expected ~-0.1%.
    - 2026-09-25 09:10 analyst: CORRECTION to my section 42 (08:34 line): the learning-curve projection was wrong for JOINT training. loopR5 (joint, 3-epoch cosine from arm A) is at -0.473% after 2 epochs, about 4x my projected -0.06..-0.12%. I wrongly took the dials as converged from loopR3/R4 drift; c0 keeps drifting (.694 -> .643). Joint training is not exhausted; judge by loopR5's epochs. RESULT section 42 carries the correction.
    - 2026-09-25 09:21 [physicist] loopR5 DONE (joint, 3 epochs TRAIN160, fresh cosine, from covhunter's arm A endpoint + S24 dials; s6bcard_loop4j, MPS, lock held): VALID16 start 1.63984e-8 -> ep0 1.63702e-8 (-0.172%, 9/16) -> ep1 1.63208e-8 (-0.473%, 14/16, median -0.41%) -> ep2 1.63206e-8 (-0.474%, 14/16, median -0.34%, range -1.74..+0.18%). Pre-registered bar 1.63164e-8: FAILS narrowly (both ep1/ep2 miss by 0.026-0.027 pt, inside noise). No overfit (train 1.671 -> 1.664 -> 1.660e-8). Cumulative vs S24 params on VALID16 (MPS): -0.94% (arm A -0.463%, then loopR5 -0.474%). Checkpoints refit6b/s6b_loopR5_joint_3ep/params_epoch{1,2}.pt + dials. CPU float64 check on the S24-structure card (armA vs R5ep1 vs R5ep2) running for the S28-vs-S27 call.
    - 2026-09-25 09:21 lead: loopR5 DONE (physicist; joint, 3-epoch cosine from arm A, MPS): epoch 1 -0.473%, epoch 2 -0.474% vs its start (14/16 each); misses the pre-registered -0.5% bar by 0.026 pt (noise). Cumulative vs S24's params ~-0.94% VALID16 (arm A 1.64753 -> 1.63990 -> loopR5 1.63206e-8). No overfit. The analyst corrected §42: joint training is NOT exhausted (c0 still moving).
      DECISION: the best loopR5 checkpoint (per the physicist's CPU f64 check on the S24-structure card) -> S28, native VALID40 vs the S27 rows.
    - 2026-09-25 09:44 [physicist] loopR5 on the DEPLOYMENT structure (no refit, VALID16, CPU float64, S24-structure card s6bcard_loop5j with star-Nystrom; refit6b/r5check/): S24 1.64837e-8; arm A (S27 base) 1.64071e-8 = -0.465% (14/16); loopR5 ep1 1.63307e-8 = -0.928% vs S24, -0.465% vs arm A (12/16); loopR5 ep2 1.63293e-8 = -0.936% vs S24 (14/16), -0.474% vs arm A (13/16, median -0.37%, range -1.79..+0.16%). Matches MPS. S28 CANDIDATE for the porter (lead's call): S27 structure + loopR5 ep2 literals = refit6b/s6b_loopR5_joint_3ep/params_epoch2.pt (sha256 dd9a084a935d6b6a...) + dials_epoch2.json (c0 .640, k4a .9835, k4b .9678, k4c .957, bscale 1.0427, X0 .0109; joint2t native-shared export). Native VALID40 vs S27 is the judge (the MPS VALID16 bar missed by 0.027 pt).
    - 2026-09-25 09:44 lead: loopR5 epoch 2 CONFIRMED on CPU f64 (physicist; VALID16, S24-structure card, no refit): -0.474% vs arm A (13/16), -0.936% vs S24 (14/16). It is S28's fallback params (params_epoch2.pt dd9a084a + dials_epoch2.json). A 3-epoch MPS run takes about 36 min, so the GPU goes back to back: plain loopR6 now, from loopR5 epoch 2 on S27's structure. This supersedes the 09:38 "skip loopR6" line. Then loopR7s at the rank schedule from loopR6's endpoint, if the held-out check passes. S28 params: loopR7s if it passes its CPU check by about 13:00, else loopR6 if it beats loopR5 epoch 2, else loopR5 epoch 2.
    2026-09-25 10:27 porter: S27p5 (S27 + loopR5 ep2 literals/dials) VALID40 vs S27 rows: adjusted (pooled): -0.526%  95% CI [-0.721%, -0.330%]  wins 33/40; per-net min -1.77% median -0.55% max +1.16% | raw raw (pooled MSE ratio): -0.527%  95% CI [-0.723%, -0.330%]  wins 33/40 (ties 0)
    - 2026-09-25 10:27 lead: PARAMS STEP NATIVE (porter S27p5 = S27 + loopR5 epoch 2 literals + dials, VALID40 vs S27 rows): raw -0.527% [CI -0.723, -0.330], 33/40; util +0.001%; adjusted -0.526%. That is the full VALID16 CPU effect (-0.474%) or slightly more: a 100% native transfer, against arm A's 64%. loopR6 epoch 1 is at -0.383% vs its start (MPS); its last epoch lands ~10:38.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_analyst_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/attrib_pair.py](code/attrib_pair.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L1017](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1017)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1018](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1018)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1042](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1042)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1043](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1043)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1044](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1044)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1045](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1045)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1066](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1066)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1067](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1067)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1083](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1083)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1084](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1084)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [RESULT-1a40d4c3.md](../../third-order-planes/gain-mixture-correction-half-strength-native-adjusted-ratio-1/RESULT-1a40d4c3.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-longer-joint-training-disproves-earlier-0-06-0-12) · [Copy provenance](../../../PROVENANCE.json)

