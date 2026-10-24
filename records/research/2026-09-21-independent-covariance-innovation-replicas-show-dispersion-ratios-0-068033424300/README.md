# Independent covariance-innovation replicas show dispersion ratios 0.068033424300 and 0.108483609808

2026-09-21 · finding · informational · measured

Two independent 8M-sample streams measured covariance and response-label dispersion on the current estimator without fitting or new sampling.

Recorded measurement:

    This measures the exact current331684 Gaussian-gain innovation target used by
    checkpoint70/81, on rank35 layers8..14, using two existing independent8M
    |covariance|0.068033424300|0.931967412570|
    |diagonal|0.107753806874|0.892278973283|
    |off_diagonal|0.067965363221|0.932035458304|
    |response|0.108483609808|0.891539903521|
    |output_only|0.982074148581|0.979831306455|0.996037144330|0.995199153108|
    |nonlinear|0.966052377454|0.963730027171|0.990035893007|0.988933765979|
    The original final16-output and528-parameter models are unchanged. Their
    this study and match independent FP64 formulas. The bank reconstruction is
    The nonlinear model's broader native panel remains failed at1.009582713004.
    The fixed decision threshold was at least20% pooled dispersion fraction in
    ceiling. Both streams share the MPS/FP32 numerical implementation and its
    unmeasured common bias. Rank35 is a TRAIN network; the parent and fitted
    Both reference affine checks pass1e-4. Every original FP32 U byte and wide
    independent linear-transport identity;16 fixed response columns per layer
    labels, both model losses, pooling and gain-gap identities within2e-10.
    Four guarded groups retired; controller28123 terminal. Peak sampled RSS1231568896 B,self-report1256816640 B,zero added swap. Artifacts 176.307 MB before closeout; no new sampling,fit,native call,download or GPU. All five bank pins and checkpoint82 studies verified.
    Official331684 remains adjusted3.5844144212297072e-9,raw1.949715441540434e-8,100/100 complete,zero failures.4e-9 banked;3/2/~1e-9 unmet. The full goal remains active; qualified uploads remain authorized.

## Programs

- [audit.py](audit.py)
- [common.py](common.py)
- [finalize.py](finalize.py)
- [measure.py](measure.py)
- [metrics.py](metrics.py)
- [model.py](model.py)
- [predict.py](predict.py)
- [preflight.py](preflight.py)
- [prepare.py](prepare.py)
- [run.py](run.py)
- [tangent.py](tangent.py)
- [unit.py](unit.py)

<details>
<summary>Additional programs and dependencies (6)</summary>

- [../../../experiments/error-attribution/covariance-innovation-independent-replica-measurement-audit-complete/code/astra_phase2_reexamination_v1/run_covariance_replica_mc_v1.py](../../../experiments/error-attribution/covariance-innovation-independent-replica-measurement-audit-complete/code/astra_phase2_reexamination_v1/run_covariance_replica_mc_v1.py)
- [common-7c4a76fa.py](common-7c4a76fa.py)
- [model-46ffa91b.py](model-46ffa91b.py)
- [../../../submissions/331684/estimator.py](../../../submissions/331684/estimator.py)
- [../../../experiments/learned-correction/learned-backward-adjoint-residual-observer-rejected-full-width-cross/code/codex_resume_20260919_v1/run_guarded.py](../../../experiments/learned-correction/learned-backward-adjoint-residual-observer-rejected-full-width-cross/code/codex_resume_20260919_v1/run_guarded.py)
- [../../../experiments/flop-accounting/honest-learned-closure-variance-heads-noise-tests-separated-information/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g19_attribution_v1/mc_truth_mps_indep.py](../../../experiments/flop-accounting/honest-learned-closure-variance-heads-noise-tests-separated-information/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g19_attribution_v1/mc_truth_mps_indep.py)

</details>

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L4](RESULT.md#L4)
- [RESULT.md#L10](RESULT.md#L10)
- [RESULT.md#L11](RESULT.md#L11)
- [REEXAMINATION_INDEX_20260919.md#L616](../../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L616)
- [REEXAMINATION_INDEX_20260919.md#L619](../../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L619)

<details>
<summary>All result files and records (6)</summary>

- [RESULT.md](RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result
- [result.json](result.json) · result
- [summary.json](summary.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-21-independent-covariance-innovation-replicas-show-dispersion-ratios-0-068033424300) · [Copy provenance](../../../PROVENANCE.json)

