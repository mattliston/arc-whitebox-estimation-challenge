# Selective flat-upload restoration removes measured runtime regression at a small FLOP debit

2026-10-20T22:51:09Z · experiment · adopted · measured

Extra changing-data uploads added 73/146 calls on networks 35/200 and raised quiet backend time +1.30/+2.38%. Selective restoration preserved parent prediction bytes on all 32 networks and passed original release gates.

Recorded measurement:

    Measuredsource regression: additional changing-data flat uploads add73/146 putcalls on ranks35/200, quietbackend+1.30/+2.38%; no proofallpublic7.5%slowdownisserverjitter. Selective HZD/ownedB restore retains otherCOPY_ALL cuts; 24official tiny cases/fullNumPy-freeimports/independentsourceauditPASS. SCI D/candidate_eh14_flatload_extra_revert_sci_v1.py9af5617cf1a8be8e8a6d0734c5779d277a0febc1f0c6a73d777bd204d2318e44, RELc7b39747c2dcaaa3dba6575bab963075416ea7a9cd0aef435ef05b25bbd39a80; tail67ec4711. Billcost+.02485/+.04364%on35/200 versusbank, expectedonly.24/.49srecoverycomponent, notwhole-timeoutsolution. Actualnativeevidencepending.
    Qualification completed23:01:37Z, D/rebank_flatload_revert_v1/QUALIFICATION_v1.json. OriginalMac full32 measured-.2472268139%SE.0998522,23/32lower; held25-.2563591641%SE.1224276,18/25lower; raw-.0212975%,bill-.2259753%. All32 predictions byte-identical to actualEH14+FG parent. Quiet bank+candidate35/150/200 both pass original7.5GB/free50/125s guards and production/science bytes+tariff parity. Original release_min setup5max3.5087864s, StageA PASS, prefire approvedgc-only/0new signatures/0otherfailure. Runtime guarantee explicitly absent; selectiveHZD/ownedB restoration addresses measured upload regression, not proof of eliminating public timeout. Localfive research dispatchers andqueue resumed23:01:37Z.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_oct20_bill_v1/candidate_eh14_flatload_extra_revert_rel_v1.py](code/astra_oct20_bill_v1/candidate_eh14_flatload_extra_revert_rel_v1.py)
- [code/astra_oct20_bill_v1/qualify_rebank_flatload_revert_v1.py](code/astra_oct20_bill_v1/qualify_rebank_flatload_revert_v1.py)
- [code/astra_crunch_physics_v1/guard_memory50_v1.py](code/astra_crunch_physics_v1/guard_memory50_v1.py)
- [code/astra_oct18_accuracy_v1/run_shared32_continuation_v1.py](code/astra_oct18_accuracy_v1/run_shared32_continuation_v1.py)
- [code/astra_oct18_bill_v1/run_guarded_product_census_v4.py](code/astra_oct18_bill_v1/run_guarded_product_census_v4.py)
- [code/astra_oct18_bill_v1/run_guarded_product_census_v5.py](code/astra_oct18_bill_v1/run_guarded_product_census_v5.py)
- [code/astra_oct18_lead_v1/run_current_controls25_v2.py](code/astra_oct18_lead_v1/run_current_controls25_v2.py)
- [code/astra_oct18_small_stack_qualification_v1/run_current32_m15c_fable_v1.py](code/astra_oct18_small_stack_qualification_v1/run_current32_m15c_fable_v1.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/run_current32_parallel5_v2.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/run_current32_parallel5_v2.py)
- [code/fable_oct19_kc_v1/compare_best_v1.py](code/fable_oct19_kc_v1/compare_best_v1.py)
- [code/fable_oct19_kc_v1/compare_relative_v1.py](code/fable_oct19_kc_v1/compare_relative_v1.py)
- [code/opus_fix336687_v1/candidate_v61j.py](code/opus_fix336687_v1/candidate_v61j.py)

<details>
<summary>Additional programs and dependencies (1)</summary>

- [code/opus_fix336687_v1/grade_store_v1.py](code/opus_fix336687_v1/grade_store_v1.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6363](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6363)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6372](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6372)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [QUALIFICATION_v1.json](QUALIFICATION_v1.json) · result

</details>

## Related work

- shipped in: [Submission #337421: Layer-14 edge width reduction with selective flat-upload restoration](../../../submissions/337421/README.md)

[Research graph](../../../graph/index.html#node=2026-10-20-selective-flat-upload-restoration-removes-measured-runtime-regression-small) · [Copy provenance](../../../PROVENANCE.json)

