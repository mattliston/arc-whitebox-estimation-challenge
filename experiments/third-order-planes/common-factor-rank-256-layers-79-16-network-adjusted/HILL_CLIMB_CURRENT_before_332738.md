# Current Phase 2 hill-climb state

Official bank remains S31 #332517, adjusted 2.1736465056908115e-9. Frozenstar32 broadVALID40 is running. The user explicitly changed the submissionminimum to fullyvalidated gains >=1%; see submission_policy_20260928_v1.json. Goalactive,submissionsauthorized.

## Retained experimental parent

`hill_climb_star_projection32_v1/candidate.py`, SHA256 f226a45750e6c3851e50967ba96f1742bb7bf563d43c1adfc2988fd674041aef. This contains S31, Thin, late common256, and the direct rank32 deterministic-column star projection with one power iteration.

- Versus S31 on matched TRAIN8: adjusted -1.352971%,8/8 wins, paired95%CI[-1.795945%,-0.928311%].
- Incremental versus common256 on matched TRAIN16: adjusted -0.499758%,12/16 wins, paired95%CI[-0.826831%,-0.171318%]. Raw +0.057483%,compute -0.557997%.
- Original TRAIN8 incremental confidence gate failed narrowly and remains recorded. A single predeclared unchanged-source extra8 panel improved0.631778%,5/8 wins; all TRAIN16 extension gates passed. No further panel was added.
- These are training results. There is no same-roster TRAIN16 S31 comparison; do not add percentages across panels or claim a leaderboard gain.

## Closed adjacent steps

| Candidate | Increment versus retained parent | Outcome |
|---|---:|---|
| Common192 versus common256 | +0.447604%,3/8 wins | Rejected |
| Star projection24 versus32 | +0.240479%,0/2 wins | Rejected |
| Negative cutoff2.0 only at layers13/14 | +0.740673%,0/2 wins; compute unchanged | Rejected |
| Thin depth4 | Microbench only0.0261%bill saving | No full candidate |
| Terminal cutoff3.0 | TRAIN16 -0.060625%,10wins,CI crosses0 | Rejected |
| Five native GN scalar fit | forecastgain0.035488%,LOOgain-0.136715%,2/6wins | No candidate exported |
| Float64 covariance diagonal | TRAIN16 +0.117174%,8wins,CI crosses0 | Rejected |
| Common256 extended toL7–9 | TRAIN16 +0.180310%,7wins,CI crosses0 | Rejected |
| Common coordinate-energy selection | TRAIN16 -0.085413%,7wins,CI crosses0 | Not retained |

Diagonal and common256L7 original smoke-gate failures were preserved, then one explicitly recorded fixed14-network extension tested each unchanged source. The larger panels rejected them. Common-energy passed its smoke; its predeclared16-panel rejected retention. Do not silently reopen or claim their small initial effects as validated gains.

Saved-factor capacity inspection in hill_climb_common_energy_capacity_v1 found that common-energy selection changes few coordinates and barely reduces discarded energy. Selected192 still loses roughly5–6times as much coefficient energy as leading256; no energy192 network test was launched. Historical uvp0r rotating-frame native S28 test was found and already unfavorable on price/mixed accuracy; not repeated.

## Current running experiment

Rootsession59301: four serial10-network broad-validation blocks on unchanged retainedstar32. Studyhill_climb_star32_broad_v1; outputhill_climb_star32_broad_v1_bank_validation_v1. This explicitly broadens measurement below the old2%training-admissionbar; the submissionminimum remains2%. Newadmissionrecordexplainswhy. Genericteam_bank_validation_v1.py stillgatesVALID40gain>=2%,>=30wins,pairedCIupper<0. No automaticupload.

Common320L7–9 followup finished: TRAIN16increment-0.134968%,9wins,CI[-0.358910%,+0.080031%]; failsretention. First2had-0.957%,butthatdidnotholdonexpansion. Sourcea4a1c050preserved,notretained. Noadditionalearlyrankneighbors.

An optionalasyncquestion asks whether userwantsfullyvalidated>=1%banksasOpusdid orkeepgoal>=2%. Noanswerrecordedyet. Maintain2%untiluserexplicitlychanges. Useralreadyauthorizessubmissions.

All completed numericalgroupsretiredwithzeroaddedswap. Current4VALIDblocksare serialunder7.5GB/360s. Recoveryteammateisread-onlyreviewingreleasefields; otherteammatesidle. Derivative/JVP V11 remainsunfinishedandparked.

Broadvalidationnetworksare reused,notpristine. A qualifyingresultstillneedsactualworker20,16layernative/billparity,handles/cachecaps,setup,boundary/runtimeprojectionandcontractchecksbeforeauthorizedupload. Releaseinstructionsareteammate_recovery_20260927_v1/release_readiness.md.

Latest policy: user answered "Bank fully validated gains ≥1%". This supersedes all older2%barreferencesabove. PreservelegacyVALIDcontrollerresultandwritecurrentpolicyadmissionafterall40; noextraapprovalneeded. SeeHANDOFF_CURRENT.md for live59301 andthree source-only releasepreparations.

## Latest: broad validation passed; qualification running

Frozenstar32f226a457 has VALID40 adjustedgain1.793775%,39/40wins,95%CIgain1.485544–2.098453%; raw+0.192799%,CPU+4.832423%. Userexplicitlyauthorizesfullyvalidated>=1%banks. Sourceunchanged. Legacy2%resultpreserved; newpolicyadmissionpassed. AllVALIDguardsretiredzeroSwap/peak6.159GB;59301reaped. Rootsession11330runningserialworker/boundary/setup/contractqualificationunderreviewedreleasefoldercontroller. ThenpotentialCPUmatchedcontrolresolution,assessor,isolatedpackage,onceupload. No furtheruserpermissionneeded. Allagentsdone;noothernumericaljobs.

**2026-09-28T02:40:10.638244+00:00: FROZEN STAR32 RESIDUAL EXCEPTION UNDER REVIEW; NO LIVE NUMERICAL JOB.**

VALID40 passes user >=1% policy: adjusted -1.793775%, 39/40 wins, CI [-2.09845,-1.48554]. Candidate f226a457 unchanged. Worker20, native/RPC parity, setup and contract completed; session11330 reaped. Original qualification_assessment.json failed historical +10ms residual heuristic; preserve it. Fresh paired boundary session35990 now reaped exit0, both groups retired zero added swap: S31 .1239/.1185 s, candidate .1415/.1342 s. Fresh deltas +.0176/+.0157 s; worse cached deltas +.0254/+.0253 s. Official residual cap .4s; S31 max 0.236215087s. Root explicit versioned headroom exception residual_resolution_v1.json: maximum observed delta times4 plus S31 official max = 0.337815087s, require <=.35s and retain 90s runtime gate/all other gates. Original config/assessment unchanged; accuracy teammate preparing assess_qualification_v2.py for independent verification, root must review/run. No FIRE_CHECKS/archive/upload yet. Recovery writing one-shot polling helper, no API/key reads. Submissions already authorized at fully validated >=1%; no confirmation needed. Goal active/unmet; derivative V11 parked.

