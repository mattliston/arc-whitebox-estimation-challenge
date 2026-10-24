# Loaded native evaluation trips inherited residual routing; release gates need at least 55% free memory

2026-10-20T00:17:30Z · incident · informational · measured

The standard-cap release-form judge asserts zero fast-rescue calls and died at its first part on a four-lane box because the card's inherited fast rescue trips under memory load; release forms are therefore judged as science copies (release form plus the science rescue cap, identical physics) and the standard judge is kept for a quiet box. The release gate itself runs under a hard 50%-free rule: a gate attempt died with lanes paused (pausing does not release memory) and passed only after the eval process and two lanes were killed; a stale queue driver also launched a fifth lane, and probes beside four lanes tripped the 1 GiB swap-growth rule.

Recorded measurement:

    release gate: free < 50% with 4 lanes + eval; after killing eval + 2 lanes 63% free, setup5 3.34 s; earlier setup5 5.026 s FAIL vs 4.82 s PASS with eval paused

## Programs

- [candidate_rel_k4cv2_sci_v1.py](candidate_rel_k4cv2_sci_v1.py)
- [../../../instruments/native-evaluation-uses-7-9-gb-caps-15-free/code/astra_oct18_small_stack_qualification_v1/run_current32_fp79_fable_v1.py](../../../instruments/native-evaluation-uses-7-9-gb-caps-15-free/code/astra_oct18_small_stack_qualification_v1/run_current32_fp79_fable_v1.py)
- [../../../experiments/birth-terms/variance-weighted-same-count-birth-ranking-8-store-networks/code/lead/runs/astra_crunch_physics_v1/guard_memory50_v1.py](../../../experiments/birth-terms/variance-weighted-same-count-birth-ranking-8-store-networks/code/lead/runs/astra_crunch_physics_v1/guard_memory50_v1.py)
- [../../../experiments/arithmetic-kernels/expanded-union-layer-5-products-32-network-adjusted-0/code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/run_shared32_continuation_v1.py](../../../experiments/arithmetic-kernels/expanded-union-layer-5-products-32-network-adjusted-0/code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/run_shared32_continuation_v1.py)
- [../../../experiments/arithmetic-kernels/expanded-union-layer-5-products-32-network-adjusted-0/code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/run_guarded_product_census_v4.py](../../../experiments/arithmetic-kernels/expanded-union-layer-5-products-32-network-adjusted-0/code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/run_guarded_product_census_v4.py)
- [../../../experiments/arithmetic-kernels/expanded-union-layer-5-products-32-network-adjusted-0/code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/run_guarded_product_census_v5.py](../../../experiments/arithmetic-kernels/expanded-union-layer-5-products-32-network-adjusted-0/code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/run_guarded_product_census_v5.py)
- [../../../experiments/arithmetic-kernels/expanded-union-layer-5-products-32-network-adjusted-0/code/opus_reexam_20261001_v1/lead/runs/astra_oct18_lead_v1/run_current_controls25_v2.py](../../../experiments/arithmetic-kernels/expanded-union-layer-5-products-32-network-adjusted-0/code/opus_reexam_20261001_v1/lead/runs/astra_oct18_lead_v1/run_current_controls25_v2.py)
- [../../../submissions/336867/estimator.py](../../../submissions/336867/estimator.py)
- [../../../experiments/covariance-structure/pair-mean-rank-old-tier-tail-dead-cut-response/code/grade_store_v1.py](../../../experiments/covariance-structure/pair-mean-rank-old-tier-tail-dead-cut-response/code/grade_store_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6021](../../TEAM_LOG_2026-10-01_to_10-22.md#L6021)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6023](../../TEAM_LOG_2026-10-01_to_10-22.md#L6023)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6051](../../TEAM_LOG_2026-10-01_to_10-22.md#L6051)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6077](../../TEAM_LOG_2026-10-01_to_10-22.md#L6077)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6097](../../TEAM_LOG_2026-10-01_to_10-22.md#L6097)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-loaded-native-evaluation-trips-inherited-residual-routing-release-gates) · [Copy provenance](../../../PROVENANCE.json)

