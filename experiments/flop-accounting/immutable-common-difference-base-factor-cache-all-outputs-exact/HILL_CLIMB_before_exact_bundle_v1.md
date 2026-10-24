# Current Phase 2 hill-climb state

Updated 2026-09-28T04:04:26.777580+00:00. Goal active and unmet. No numerical jobs, grading, upload or approval pending. Derivative/JVP V11 remains parked.

## Official bank

**S32 #332738: adjusted2.1399773263156476e-9**,1.548972% better than S31;100/100 complete, zero failures. Source `hill_climb_star_projection32_v1/candidate.py`, SHA256 `f226a45750e6c3851e50967ba96f1742bb7bf563d43c1adfc2988fd674041aef`. Official receipt `official_bank_S32_v1.json`. Do not reupload.

User policy: **bank fully validated aggregate gains>=1%**; submissions already authorized. Smaller validated components may accumulate into a bundle. No additional permission is required. Current bank has not reached2e-9 or approximately1e-9.

## Retained follow-up components

1. **Exact UVP base B cache**, source `hill_climb_uvp_base_bcache_v1/candidate.py`, SHA256 `0308a99fd89823cfd67e19c3ad00c946d4211b6c69d956519267e142aa205fd5`. Native2 bill-0.1158296%, all16 bytes exact; actual worker2 and boundary2 exact bytes/bills and resource checks passed.13 hits/3 paid copies per network,337MB cache fully retired,319,701,476 saved FLOPs and296 fewer requests. Peak groupRSS5.997GB, handles48,637, zero added swap. Receipt `hill_climb_uvp_base_bcache_v1/retained_component.json`.
2. **Nine D3 setup bindings**, source `hill_climb_star32_fused_d3_v1/candidate.py`, SHA256 `61a1f40b8a6e50d6c5e1b403a2200b0979d3d02794c0f522e29b98436c513c6e`. Worker2/boundary2/setup3 passed, all16 bytes exact, requests-996 cold/-870 warm. Cold bill-23,761,920F from public shape-only bindings; warm bill identical. Original failed equal-bill assertion and exact accounting correction preserved in `probe_assessment_v2.json`.

These are separate sources, not yet composed. Neither has fullworker20/broad validation or earns another submission by itself. Preserve them for a future bundle.

## Completed current-source studies

- Direct starK40/RNone/Q1: TRAIN2 adjusted+0.0436595%,1/2wins, raw-0.359184%, bill+0.404409%. Rejected; no selective-rank inference from global trajectories.
- Birth+32 atL1–10: passed first2(-0.245412%) but **rejected fixed16, adjusted+1.046220%,4wins,95%CI[+0.516582,+1.578596]**. Raw-0.703650%,bill+1.752359%,additional14+1.206259%. All guards retired, peak6.105GB, zero swap; no follow-up scope/rank sweep. `hill_climb_birth_plus32_v1/train16_result.json`.
- Exact S32 rank35 cost census: fullbill276,549,645,399F and all16 bytes exact. Push43.4479%, read34.3864%, covstar11.0386%; KS24.8960%. KS chunk partition leaves multiplication+accumulation arithmetic invariant; padding/output copies only0.0263% of total. No new chunk-size candidate. `hill_climb_s32_bill_census_v1/`.

Historical cache correction: earlier old-factor cache was not implemented. Fable skipped a1.1GB resource forecast; S31 reuse census stopped at a former1.5% implementation threshold. Do not turn these into measured runtime failures. The new bounded cache now supplies actual evidence. Seven historical mask cuts all had unfavorable adjusted forecasts; global/late larger birth expansions failed economics.

Other closed neighbors remain closed: star24, common192, late negativecut2, terminalcut3, covariance diagonal64, common256/320 atL7–9, energy-selected common coordinates, no-power star, repeated GN scalar fits and duplicate LoopR8s. Details and previous state: `HILL_CLIMB_before_uvp_bcache_v1.md`, `RESULTS.md`. Do not resume derivativeV11.

Restart receipt: `continuation_uvp_bcache_v1.json`. Use current S32 controls `hill_climb_star_projection32_train2_v1`, `...train6_v1`, `...extra8_candidate_v1`. Root owns serial numerical execution under7.5GB guards with zero added swap; use only the three existing teammates for source work.
