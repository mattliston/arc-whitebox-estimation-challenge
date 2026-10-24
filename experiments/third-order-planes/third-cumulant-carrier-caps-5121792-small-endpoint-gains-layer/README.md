# Third-cumulant carrier caps 512–1792: small endpoint gains, layer-2 error saturated at cap 1024

2026-08-23T18:06Z · experiment · adopted · measured

Four preregistered estimators identical except for the symmetric CP3 factor-bank cap (512, 1024, 1536, 1792; float32 symmetric-CP K3 readout on the float64 Hermite-10 covariance host) were metered target-blind under the common driver and then scored once on mini100. The K3 readout lowers the layer-2 error by a factor of about seven relative to the Hermite host and the layer-2 error stops changing above cap 1024, but the final-layer error moves by only a few percent across the whole ladder, so the third-order marginal readout is not where the layer-16 error lives.

Recorded measurement:

    metered FLOPs per MLP: 136,462,170,114 / 168,812,139,522 / 199,018,976,258 / 214,122,341,378 for caps 512/1024/1536/1792; final raw MSE 3.799028043642153e-6 / 3.762168087950475e-6 / 3.7339069924135637e-6 / 3.722402415224689e-6; all-layer 2.0302987318411863e-6 / 1.9742425590632073e-6 / 1.936026483441147e-6 / 1.920413376561686e-6; layer 2 7.850484537840704e-8 (cap 512) and 2.192582806672553e-8 (caps 1024, 1536, 1792); Hermite order 4 reference 3.849300358689406e-6 final, 1.5536452260972563e-7 layer 2

Program source is in [`code/`](code/), preserving the relative layout below `.`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/cp3_cap512_k3_readout/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/cp3_cap512_k3_readout/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/cp3_cap1792_k3_readout/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/cp3_cap1792_k3_readout/estimator.py)
- [code/estimator.py](code/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/cp3_cap1024_k3_readout/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/cp3_cap1024_k3_readout/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/cp3_cap1536_k3_readout/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/cp3_cap1536_k3_readout/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/cp3_cap512_k3_readout/verify_source.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/cp3_cap512_k3_readout/verify_source.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/cp3_cap_ladder_preregistered/derive_static_ladder.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/cp3_cap_ladder_preregistered/derive_static_ladder.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/meter_protocol/run_synthetic_meter.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/meter_protocol/run_synthetic_meter.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/meter_protocol/verify_meter_run.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/meter_protocol/verify_meter_run.py)

## Evidence

- [BUDGET_AUDIT.json#L1](../../../instruments/target-custody-tool-one-use-sealed-scoring-target-blind/BUDGET_AUDIT.json#L1)
- [STATIC_LADDER.json#L1](STATIC_LADDER.json#L1)

<details>
<summary>All result files and records (5)</summary>

- [PREREGISTRATION.md](PREREGISTRATION.md) · record
- [STATIC_LADDER.json](STATIC_LADDER.json) · result
- [CP_LADDER_METER_ADMISSION.json](CP_LADDER_METER_ADMISSION.json) · result
- [BUDGET_AUDIT.json](../../../instruments/target-custody-tool-one-use-sealed-scoring-target-blind/BUDGET_AUDIT.json) · result
- [METER_LADDER_SUMMARY.json](../../../instruments/literal-meter-harness-subprocess-evaluation-run-records-verification-resource/METER_LADDER_SUMMARY.json) · result

</details>

## Related work

- ← measured by: [Scalar third-cumulant readout rescaling: layer-16 gain fraction below 0.004 versus 0.066 at layer 2](../../../records/research/2026-08-23-scalar-third-cumulant-readout-rescaling-layer-16-gain-fraction/README.md)
- supersedes: [Hermite pair-series order sweep: order 4 reached 3.849300358689406e-6 final raw error; orders 4–10 tied](../../series-expansions/hermite-pair-series-order-sweep-order-4-reached-3/README.md)
- ← supersedes: [Full-birth third-cumulant carrier with recursive mean feedback: final raw error 9.340196551564291e-7](../full-birth-third-cumulant-carrier-recursive-mean-feedback-final/README.md)

[Research graph](../../../graph/index.html#node=2026-08-23-third-cumulant-carrier-caps-5121792-small-endpoint-gains-layer) · [Copy provenance](../../../PROVENANCE.json)

