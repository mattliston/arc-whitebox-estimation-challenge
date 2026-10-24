# Stack gate v1-v3: base plus appended fragments through release pipeline, count gates, exact bill and pre-registration

2026-10-16T15:20:52Z · tool · adopted · measured

stack_gate builds a candidate as a base file plus fragments appended last, runs release_pipeline_v5 (boundary normal and slow, official 120 s gate, stage A, captured-setup parity, RELEASE_CONFIRMED, prefire), then the count gates, writes STACK_<name>.json with the exact bill against the base and the estimator submitted as #336412 (S2tr) and a pre-registered projection, and prints the fire command only if everything passes (one native, 5-10 minutes). v2 moved to count_gates_v4; v3 uses release_pipeline_v5b, whose STEP6 replay helper (captured_setup_replay_generic_v2) also reproduces a setup that bound the whole prebind list and then fused keys, after three builds (iffy-noB, S1b, S5+ms) stopped at STEP6 on the v1 assertion keys_bound < keys_total or on a 1-flop in-process parity difference.

Recorded measurement:

    First clean stack confirmed in 5.3 min. STEP6 parity trap: fnp.asarray(fnp.float32(0)) bills 1 flop in-process and 0 in production (fix: fnp.zeros((), float32)). iffy-noB2 setup binds all 649 keys in 16,556 requests and fuses 48; released builds bind 588-589 of 649-652 and fuse 0.

## Programs

- [code/stack_gate_v1.py](code/stack_gate_v1.py)
- [code/stack_gate_v2.py](code/stack_gate_v2.py)
- [code/stack_gate_v3.py](code/stack_gate_v3.py)
- [code/stack_v1/cand_s3.py](code/stack_v1/cand_s3.py)
- [code/s3_v1/cand_s3base_v1.py](code/s3_v1/cand_s3base_v1.py)
- [code/stack_v1/cand_iffynob2.py](code/stack_v1/cand_iffynob2.py)
- [code/repair_v1/cand_repair_VQm_v1.py](code/repair_v1/cand_repair_VQm_v1.py)
- [code/count_gates_v3.py](code/count_gates_v3.py)
- [code/count_gates_v4.py](code/count_gates_v4.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4278](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4278)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4281](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4281)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4286](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4286)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4314](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4314)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4315](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4315)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4479](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4479)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4485](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4485)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4495](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4495)

<details>
<summary>All result files and records (3)</summary>

- [STACK_s3.json](STACK_s3.json) · record
- [STACK_iffynob2.json](STACK_iffynob2.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-10-16-stack-gate-v1-v3-base-plus-appended-fragments-through) · [Copy provenance](../../PROVENANCE.json)

