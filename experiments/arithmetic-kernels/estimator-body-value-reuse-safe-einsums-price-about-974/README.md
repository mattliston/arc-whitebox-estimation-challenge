# Estimator-body value reuse and safe einsums price about 974 fewer requests per network

2026-09-25T08:44Z · experiment · informational · measured

The census separated exact reuse from operation forms that changed billing, rounding or client behavior. A limited prototype preserved all layer outputs and billing.

Recorded measurement:

    - 2026-09-25 01:44 [engineer2] E-71 GROUP 5 (physics-body request diet): PRICED, and the einsum part PROTOTYPED and verified (lead GO paper; fable_engineer2_v1/diet5/). Census: per-request trace of S23 net 100, the first net after setup (32,316 requests, 11,455 in the body), with each request's bill and full call stack (body_census.py v2).
      (1) Value reuse, bit-identical: repeated compute -421, re-derived views -89, per-process constants -32, no-op cast -1 = -543. Bill -49 M (-0.016%).
      (2) Einsum-ize, counted after (1): -431. A multiply chain and its broadcast/transpose views run as ONE einsum(optimize=False). Only class (a) forms (distinct operand objects, the crash law), no out=, and compute-safe forms: vector chains, one multiply plus its views, and float64 patterns measured faster. Value-identical.
      Total: about -974 requests (-3.0%), about -6.3 ms residual/net locally.
      EXCLUDED:
      - repeated-operand einsum -173 (crash law);
      - einsum with out= -204 (flopscope bills it at 2x the multiply);
      - float32 or slower multi-multiply matrix einsums -120 (1.2-2.0x server time);
      - rounding-level multiply->sum contractions -62 (85 more repeat an operand);
      - stacked cross-function fusion, estimated -1,188 (needs persistent stacked buffers; the engineer's strided-op compute penalty applies).
      PROTOTYPE diet5/proto/S23_g5e_mem.py (f288a88a): the einsum part in derivatives, _k31_row_scale, _gn_dcov and _bd_update, through guarded helpers that check operand identity at run time, with 0-d constants bound in setup.
      - Einsum gate PASS: 0 new repeated-operand sites, 0 new symmetric tags.
      - Parity, setup first, flopscope 0.12, nets 35/100: all 16 layers BITWISE identical, 0 signed-zero flips, flops identical.
      - Census, net 100: -264 requests (= the predicted -264), -42 free trips, same bill, setup 1.34 s.
      Every form used is certified bitwise and same-bill (diet5/proto/einsum_forms_check.py). Nothing else is built; waiting for the lead's GO.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_engineer2_v1/diet5`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/body_census.py](code/body_census.py)
- [code/proto/S23_g5e_mem.py](code/proto/S23_g5e_mem.py)
- [code/proto/einsum_forms_check.py](code/proto/einsum_forms_check.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L899](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L899)
- [TEAM_LOG_2026-09-22_to_09-26.md#L901](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L901)
- [TEAM_LOG_2026-09-22_to_09-26.md#L902](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L902)
- [TEAM_LOG_2026-09-22_to_09-26.md#L903](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L903)
- [TEAM_LOG_2026-09-22_to_09-26.md#L904](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L904)
- [TEAM_LOG_2026-09-22_to_09-26.md#L905](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L905)
- [TEAM_LOG_2026-09-22_to_09-26.md#L906](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L906)
- [TEAM_LOG_2026-09-22_to_09-26.md#L907](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L907)
- [TEAM_LOG_2026-09-22_to_09-26.md#L908](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L908)
- [TEAM_LOG_2026-09-22_to_09-26.md#L909](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L909)
- [TEAM_LOG_2026-09-22_to_09-26.md#L910](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L910)
- [TEAM_LOG_2026-09-22_to_09-26.md#L911](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L911)
- [TEAM_LOG_2026-09-22_to_09-26.md#L912](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L912)
- [TEAM_LOG_2026-09-22_to_09-26.md#L913](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L913)
- [TEAM_LOG_2026-09-22_to_09-26.md#L914](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L914)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-estimator-body-value-reuse-safe-einsums-price-about-974) · [Copy provenance](../../../PROVENANCE.json)

