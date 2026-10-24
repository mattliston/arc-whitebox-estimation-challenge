# Four value-only changes on 49 public networks: MSE +0.318%, bill -0.541%, adjusted -0.223%

2026-10-16T20:37:46Z · finding · informational · measured

The completed public networks of #336570, paired by network against #336412, measured the four value-only parts (JORD + DC35 + L15RC30 + DCV2, plus the dm2/lazybind bill) on the board population: a small accuracy loss and a bill cut, worth about -0.22% adjusted, exactly the store verdict (seeds had said MSE -0.38%). With store-aligned numbers no available build beat oliver, so the five single-part measurement fires prepared for the next day (dead cut, row cut, floor, rank 320, old-tier tail; all RELEASE_CONFIRMED) were moot. The per-network response to value-only changes was noise-like (sd 0.90%, no correlation with base bill, base MSE or bill change), so no class to gate on exists and even an oracle per-net switch would give only -0.22%.

Recorded measurement:

    19 nets: MSE +0.255% (SE .16, 6/19 lower), bill -0.571%, score -0.316% (14/19). 49 nets: MSE +0.318% (SE .13, 13/49 lower), bill -0.541%, score -0.223% (31/49), aggregate -0.213% -> ~1.4727e-9. Per-net dMSE sd 0.90%; corr with base bill +0.08, base MSE +0.02, dBill -0.13. Measurement arms: mr7rank320 bill -1.025%, ml15rc30 -0.514%, mdcv2 -0.159%, mottp30 +0.491%.

## Programs

- [../../../submissions/336570/public_pair_v1.py](../../../submissions/336570/public_pair_v1.py)
- [cand_mr7rank320.py](cand_mr7rank320.py)
- [cand_m_r7rank320.py](cand_m_r7rank320.py)
- [cand_ml15rc30.py](cand_ml15rc30.py)
- [cand_m_l15rc30.py](cand_m_l15rc30.py)
- [cand_mdcv2.py](cand_mdcv2.py)
- [cand_m_dcv2.py](cand_m_dcv2.py)
- [cand_mottp30.py](cand_mottp30.py)
- [cand_m_ottp30.py](cand_m_ottp30.py)
- [cand_mdc35.py](cand_mdc35.py)
- [cand_m_dc35.py](cand_m_dc35.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4477](../../TEAM_LOG_2026-10-01_to_10-22.md#L4477)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4485](../../TEAM_LOG_2026-10-01_to_10-22.md#L4485)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4489](../../TEAM_LOG_2026-10-01_to_10-22.md#L4489)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4491](../../TEAM_LOG_2026-10-01_to_10-22.md#L4491)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4506](../../TEAM_LOG_2026-10-01_to_10-22.md#L4506)

<details>
<summary>All result files and records (7)</summary>

- [eval_336570.json](../../../submissions/336570/eval_336570.json) · result
- [STACK_mr7rank320.json](STACK_mr7rank320.json) · record
- [STACK_ml15rc30.json](STACK_ml15rc30.json) · record
- [STACK_mdcv2.json](STACK_mdcv2.json) · record
- [STACK_mottp30.json](STACK_mottp30.json) · record
- [STACK_mdc35.json](STACK_mdc35.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- measured by: [Store-32 validation: every value-only part picked on generated seeds reads about zero or worse on 32 store networks](../2026-10-16-store-32-validation-every-value-only-part-picked-generated/README.md)
- ← measured by: [Submission #336570: join ordering and three cutoff changes; a public network failed at .430 s residual](../../../submissions/336570/README.md)

[Research graph](../../../graph/index.html#node=2026-10-16-four-value-only-changes-49-public-networks-mse-0) · [Copy provenance](../../../PROVENANCE.json)

