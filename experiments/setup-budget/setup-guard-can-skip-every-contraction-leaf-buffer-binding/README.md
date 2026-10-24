# Setup guard can skip every contraction leaf buffer; binding those buffers first is staged

2026-09-25T17:06Z · experiment · adopted · measured

Under load, frame fusion exhausted the setup guard before leaf-buffer allocation. Setup-only variants prioritize the buffers or trim the key list while preserving the predictor.

Recorded measurement:

    2026-09-25 10:06 porter: S27r SETUP GUARD issue (lead 10:0x): S27r's re-derived list = 388 keys (S27: 372); under load (9-10) setup hit the 3.5 s guard during fusion (lvfuse 383 by 3.504 s), 0/22 k-split leaf buffers bound. Built two setup-only variants from s27r_pre.py (+ treuse last; einsum/validate PASS): stack_S27r_ksfirst_mem.py (_pb_bind wrapper: v2b arena verbatim -> 22 leaf buffers -> lvfuse2 two-phase bind; build_setup_variants.py) and stack_S27r_trim372_mem.py (list cut to the first 372 of 388 in rule order; drops 16 C-side keys). Running setup_variants_census.sh: in-process bit-identity + concurrent census pairs vs S27 (setup wall, per-phase log, bill, requests). New S28 smoke gate: all intended keys bound (list + batched + leaf buffers; S27 394/394 PASS, S27r 388/410 FAIL).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_porter_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/stack/s27r_pre.py](code/stack/s27r_pre.py)
- [code/stack/stack_S27r_ksfirst_mem.py](code/stack/stack_S27r_ksfirst_mem.py)
- [code/build_setup_variants.py](code/build_setup_variants.py)
- [code/stack/stack_S27r_trim372_mem.py](code/stack/stack_S27r_trim372_mem.py)
- [code/setup_variants_census.sh](code/setup_variants_census.sh)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L1078](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1078)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

## Related work

- shipped in: [Submission #332361: middle join rank 320 with refitted learned correction and matched dials](../../../submissions/332361/README.md)

[Research graph](../../../graph/index.html#node=2026-09-25-setup-guard-can-skip-every-contraction-leaf-buffer-binding) · [Copy provenance](../../../PROVENANCE.json)

