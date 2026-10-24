# Scoring an incomplete native run creates its merge directory and prevents the final merge

2026-10-19T11:10:58Z · incident · informational · measured

Running the partial grader on a lane still in progress created candidate_all32/ so the judge's own final merge failed with FileExistsError; the partial numbers stand but RESULT32 is never written. Rule: never grade a running lane; read early bill verdicts from part rows only.

## Programs

- [grade_partial_fable_v1.py](grade_partial_fable_v1.py)
- [../../../experiments/birth-terms/variance-weighted-same-count-birth-ranking-8-store-networks/code/lead/runs/astra_crunch_physics_v1/guard_memory50_v1.py](../../../experiments/birth-terms/variance-weighted-same-count-birth-ranking-8-store-networks/code/lead/runs/astra_crunch_physics_v1/guard_memory50_v1.py)
- [../../../experiments/arithmetic-kernels/expanded-union-layer-5-products-32-network-adjusted-0/code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/run_shared32_continuation_v1.py](../../../experiments/arithmetic-kernels/expanded-union-layer-5-products-32-network-adjusted-0/code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/run_shared32_continuation_v1.py)
- [../../../experiments/arithmetic-kernels/expanded-union-layer-5-products-32-network-adjusted-0/code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/run_guarded_product_census_v4.py](../../../experiments/arithmetic-kernels/expanded-union-layer-5-products-32-network-adjusted-0/code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/run_guarded_product_census_v4.py)
- [../../../experiments/arithmetic-kernels/expanded-union-layer-5-products-32-network-adjusted-0/code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/run_guarded_product_census_v5.py](../../../experiments/arithmetic-kernels/expanded-union-layer-5-products-32-network-adjusted-0/code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/run_guarded_product_census_v5.py)
- [../../../experiments/arithmetic-kernels/expanded-union-layer-5-products-32-network-adjusted-0/code/opus_reexam_20261001_v1/lead/runs/astra_oct18_lead_v1/run_current_controls25_v2.py](../../../experiments/arithmetic-kernels/expanded-union-layer-5-products-32-network-adjusted-0/code/opus_reexam_20261001_v1/lead/runs/astra_oct18_lead_v1/run_current_controls25_v2.py)
- [../../../submissions/336867/estimator.py](../../../submissions/336867/estimator.py)
- [../../../experiments/covariance-structure/pair-mean-rank-old-tier-tail-dead-cut-response/code/grade_store_v1.py](../../../experiments/covariance-structure/pair-mean-rank-old-tier-tail-dead-cut-response/code/grade_store_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5942](../../TEAM_LOG_2026-10-01_to_10-22.md#L5942)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-19-scoring-incomplete-native-run-creates-merge-directory-prevents-final) · [Copy provenance](../../../PROVENANCE.json)

