# Six fresh networks established a held-out gate for frozen correction tables and refitted candidates

2026-09-04 · experiment · informational · measured

Fresh nets 3016500-3016505 with weights, 4M MPS truth and the deployed card's baseline (host+fitted transport-energy residual correction (F2J) mean squared error (MSE) 1.89-4.10e-8, mean 2.71e-8 ~ the board's 2.94e-8; lineage check BASE == estimator_cap.py to 1e-5). Harness run_virgin_gate_p2.py <cand.py> <seeds> (venv; BudgetContext 2^41; paired cut vs deployed, bill). Fire pre-authorized by Matt (2026-09-02) after: 8-net panel under independent truths, virgin gate, adjusted-score check, meaningful bar or the ~1.5e-8 bank line; ledger entry; 10/day. 2026-09-03: gate ready; no candidate beats the board (round-1 candidate valid but worse); nothing fired. 2026-09-03 04:00: no candidate reached the gate; nothing fired; board 2.94e-8 unchanged.

Recorded measurement:

    Fresh nets 3016500-3016505 with weights, 4M MPS truth and the deployed card's baseline (host+F2J MSE 1.89-4.10e-8, mean 2.71e-8 ~ the board's 2.94e-8; lineage check BASE == estimator_cap.py to 1e-5). Harness run_virgin_gate_p2.py <cand.py> <seeds> (venv; BudgetContext 2^41; paired cut vs deployed, bill). Fire pre-authorized by Matt (2026-09-02) after: 8-net panel under independent truths, virgin gate, adjusted-score check, meaningful bar or the ~1.5e-8 bank line; ledger entry; 10/day. 2026-09-03: gate ready; no candidate beats the board (round-1 candidate valid but worse); nothing fired. 2026-09-03 04:00: no candidate reached the gate; nothing fired; board 2.94e-8 unchanged.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_d11_util_halving_v1/estimator_cap.py](code/public_d11_util_halving_v1/estimator_cap.py)
- [code/public_g19_attribution_v1/mc_truth_mps.py](code/public_g19_attribution_v1/mc_truth_mps.py)
- [code/public_g19_attribution_v1/run_attribution.py](code/public_g19_attribution_v1/run_attribution.py)
- [code/public_g19_attribution_v1/run_fresh_deployed.py](code/public_g19_attribution_v1/run_fresh_deployed.py)
- [code/public_g19_attribution_v1/run_virgin_gate_p2.py](code/public_g19_attribution_v1/run_virgin_gate_p2.py)
- [code/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/public_q24_native_transport_energy_final_workspace_v4/estimator.py)

## Evidence

- [LANE_g3_tracker.md#L622](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L622)
- [fresh_3016501_truth.log#L1](fresh_3016501_truth.log#L1)
- [fresh_3016501_deployed_v2.log#L1](fresh_3016501_deployed_v2.log#L1)
- [fresh_3016500_deployed.log#L1](fresh_3016500_deployed.log#L1)
- [fresh_3016503_deployed.json#L1](fresh_3016503_deployed.json#L1)
- [fresh_3016504_instance.log#L1](fresh_3016504_instance.log#L1)
- [fresh_3016503_instance.log#L1](fresh_3016503_instance.log#L1)
- [fresh_3016500_deployed_v2.log#L1](fresh_3016500_deployed_v2.log#L1)
- [fresh_3016500_truth.log#L1](fresh_3016500_truth.log#L1)
- [fresh_3016500_deployed.json#L1](fresh_3016500_deployed.json#L1)
- [fresh_3016504_deployed.json#L1](fresh_3016504_deployed.json#L1)
- [fresh_3016502_instance.log#L1](fresh_3016502_instance.log#L1)
- [fresh_3016505_instance.log#L1](fresh_3016505_instance.log#L1)
- [fresh_3016505_truth.log#L1](fresh_3016505_truth.log#L1)
- [fresh_3016505_deployed.json#L1](fresh_3016505_deployed.json#L1)
- [fresh_3016500_cap.log#L1](fresh_3016500_cap.log#L1)
- [fresh_3016501_deployed.json#L1](fresh_3016501_deployed.json#L1)
- [fresh_3016502_deployed_v2.log#L1](fresh_3016502_deployed_v2.log#L1)
- [fresh_3016500_instance.log#L1](fresh_3016500_instance.log#L1)
- [fresh_3016503_truth.log#L1](fresh_3016503_truth.log#L1)
- [fresh_3016505_deployed_v2.log#L1](fresh_3016505_deployed_v2.log#L1)
- [fresh_3016500_cand_estimator_cap.json#L1](fresh_3016500_cand_estimator_cap.json#L1)
- [fresh_3016502_deployed.json#L1](fresh_3016502_deployed.json#L1)
- [fresh_3016501_instance.log#L1](fresh_3016501_instance.log#L1)
- [fresh_3016504_truth.log#L1](fresh_3016504_truth.log#L1)
- [fresh_3016504_deployed_v2.log#L1](fresh_3016504_deployed_v2.log#L1)
- [fresh_3016502_truth.log#L1](fresh_3016502_truth.log#L1)
- [fresh_3016503_deployed_v2.log#L1](fresh_3016503_deployed_v2.log#L1)

<details>
<summary>All result files and records (28)</summary>

- [LANE_g3_tracker.md](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record
- [fresh_3016501_truth.log](fresh_3016501_truth.log) · log
- [fresh_3016501_deployed_v2.log](fresh_3016501_deployed_v2.log) · log
- [fresh_3016500_deployed.log](fresh_3016500_deployed.log) · log
- [fresh_3016503_deployed.json](fresh_3016503_deployed.json) · result
- [fresh_3016504_instance.log](fresh_3016504_instance.log) · log
- [fresh_3016503_instance.log](fresh_3016503_instance.log) · log
- [fresh_3016500_deployed_v2.log](fresh_3016500_deployed_v2.log) · log
- [fresh_3016500_truth.log](fresh_3016500_truth.log) · log
- [fresh_3016500_deployed.json](fresh_3016500_deployed.json) · result
- [fresh_3016504_deployed.json](fresh_3016504_deployed.json) · result
- [fresh_3016502_instance.log](fresh_3016502_instance.log) · log
- [fresh_3016505_instance.log](fresh_3016505_instance.log) · log
- [fresh_3016505_truth.log](fresh_3016505_truth.log) · log
- [fresh_3016505_deployed.json](fresh_3016505_deployed.json) · result
- [fresh_3016500_cap.log](fresh_3016500_cap.log) · log
- [fresh_3016501_deployed.json](fresh_3016501_deployed.json) · result
- [fresh_3016502_deployed_v2.log](fresh_3016502_deployed_v2.log) · log
- [fresh_3016500_instance.log](fresh_3016500_instance.log) · log
- [fresh_3016503_truth.log](fresh_3016503_truth.log) · log
- [fresh_3016505_deployed_v2.log](fresh_3016505_deployed_v2.log) · log
- [fresh_3016500_cand_estimator_cap.json](fresh_3016500_cand_estimator_cap.json) · result
- [fresh_3016502_deployed.json](fresh_3016502_deployed.json) · result
- [fresh_3016501_instance.log](fresh_3016501_instance.log) · log
- [fresh_3016504_truth.log](fresh_3016504_truth.log) · log
- [fresh_3016504_deployed_v2.log](fresh_3016504_deployed_v2.log) · log
- [fresh_3016502_truth.log](fresh_3016502_truth.log) · log
- [fresh_3016503_deployed_v2.log](fresh_3016503_deployed_v2.log) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-six-fresh-networks-established-held-out-gate-frozen-correction) · [Copy provenance](../../../PROVENANCE.json)

