# Paired reports automatically compare new candidates with the current best valid submission

2026-10-20T20:32:34Z · tool · informational · measured

A reference pointer and scorer were added after submission #337393 graded. They preserve older baselines for provenance while preventing historical component gains from being reported as new incremental gains.

Recorded measurement:

    MATT NEW INSTRUCTION: "please always compare all new adj gains to our new best." BEST GRADED = #337393, exactpublicscore1.4272337950135872e-9, VALID100/100,zero failures. Actual improvement versus previousbest#337259board1.4284668696282146e-9 was.08632%, smallerthanlocal.33741%. J2W1.416013781559781e-9 nowrequires.786137%furtherimprovement. Newreportingpointer K/BEST_GRADED_REFERENCE_v1.json maps#337393 toscience32_gram_pair_kclm_copy_astra_m15 SCI d03b2500/RELfd873958. Use K/compare_best_v1.py for ALL NEW USER-FACING READS; saved VS_BEST_337393_v1.json. Frozen compare_relative_v1/v4a evidence kept forqualification/provenance; oldgainmustnotbeclaimedasincrementalbankgain. Automaticallyadvancereportingpointerwhenabettervalidsubmissiongrades.

## Programs

- [code/compare_best_v1.py](code/compare_best_v1.py)
- [code/compare_relative_v1.py](code/compare_relative_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6277](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6277)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [BEST_GRADED_REFERENCE_v1.json](BEST_GRADED_REFERENCE_v1.json) · result

</details>

[Research graph](../../graph/index.html#node=2026-10-20-paired-reports-automatically-compare-new-candidates-current-best-valid) · [Copy provenance](../../PROVENANCE.json)

