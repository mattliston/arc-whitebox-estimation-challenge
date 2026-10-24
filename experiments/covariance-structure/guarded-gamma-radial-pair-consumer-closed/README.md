# Guarded Gamma radial pair consumer: closed

2026-09-06T06:47Z · experiment · rejected · measured

The universal8-node certificate remains FAILED at synthetic r=.30. The separate numerical-domain-guarded version passes: parent exactly outside 0<=r<=.10, unchanged integration within it. Actual information result: pooled covariance error ratio1.0322070915, pooled next-mean-response error ratio1.0057929469, only2/7 value wins.

Recorded measurement:

    The universal8-node certificate remains FAILED at synthetic r=.30. The separate numerical-domain-guarded version passes: parent exactly outside 0<=r<=.10, unchanged integration within it. All current cells are inside and passed exact input/host/target replay and the original8/32 tolerances. Actual information result: pooled covariance error ratio1.0322070915, pooled next-mean-response error ratio1.0057929469, only2/7 value wins. Fails the .95/.90/5-win gates. No independent-net expansion, full-state propagation, coefficient/domain/node sweep, fit, candidate or submission. This is not a final-MSE or adjusted-score measurement.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/check_gamma_radial_guarded_value_v3.py](code/astra_phase2_reexamination_v1/check_gamma_radial_guarded_value_v3.py)
- [code/astra_phase2_reexamination_v1/gamma_radial_guarded_v3.py](code/astra_phase2_reexamination_v1/gamma_radial_guarded_v3.py)
- [code/astra_phase2_reexamination_v1/gamma_radial_pair_v1.py](code/astra_phase2_reexamination_v1/gamma_radial_pair_v1.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)
- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)

## Evidence

- [GAMMA_RADIAL_GUARDED_V3_CLOSED.md#L1](GAMMA_RADIAL_GUARDED_V3_CLOSED.md#L1)

<details>
<summary>All result files and records (3)</summary>

- [GAMMA_RADIAL_GUARDED_V3_CLOSED.md](GAMMA_RADIAL_GUARDED_V3_CLOSED.md) · record
- [gamma_radial_guarded_value_r35_v3.json](gamma_radial_guarded_value_r35_v3.json) · result
- [GAMMA_RADIAL_GUARDED_V3_DESIGN.md](GAMMA_RADIAL_GUARDED_V3_DESIGN.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-guarded-gamma-radial-pair-consumer-closed) · [Copy provenance](../../../PROVENANCE.json)

