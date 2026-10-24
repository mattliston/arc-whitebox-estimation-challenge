# Additional eligible depth-six products save 0.54% work for 3.95% more user CPU

2026-09-25T02:48Z · experiment · adopted · measured

An exact shape-aware arithmetic rule adds depth only where the saving is large enough. The resulting candidate passed paired residual and wall gates.

Recorded measurement:

    2026-09-24 19:48 [engineer2] d6x LAYER ON S18 (lead: remaining depth moves + W5): fable_engineer2_v1/d6x/S18_d6x_v2_mem.py d3884efb = S18 ab8c8ba8 + d6x_fragment_v2.py (wrappers only). (1) depth-5 _packed_product_cached products -> depth 6 when the exact alt-basis bill model bill(6)-bill(5) = -0.12822 mkn + 12.56(mk+kn) + 12.75 mn predicts >= 15 Mflop/call saved (model matches in-situ S18 bills to 3 digits; reads stay depth 5); (2) W5-granted products at depth 6 with their own cached depth-6 A tree (arena-sharing with the colpack push NOT used: fragile across calls, worth <= -0.05% more); (3) prebind list re-derived for d6x by the engineer's key-census procedure (359 keys, held-out coverage 63-76% vs 55-64% for S18's list on d6x). S18 in-situ price table: all moves -0.64..-0.72%/net for +3.7..+4.5 s (1 local user-s = 1.78 platform s on S18). PARITY vs S18 (35/100/37/39): bill -0.562% (cold -0.74%, warm -0.49..-0.52%), MSE -0.012%, adjusted -0.574%, maxrel 2e-6. validate PASS, einsum gate PASS, no new imports. User-CPU pair + paired residual gate (setup client) running.
    2026-09-24 20:01 [engineer2] d6x (S18_d6x_v2 d3884efb) GATES vs S18: user-CPU pair (6 nets) user +3.95% (se 0.33%), bill -0.54%, op calls +770/net; paired residual gate (setup-calling client, 3 concurrent reps, 35 cold/100/37) cold delta +0.005 s (rule <= +0.010) PASS (warm +0.008 / -0.002 s; requests +630..+1,040/net; bill -0.50..-0.74% with prebinding); project_wall_v3 base #332226: expected worst 70.2 / 69.7 / 69.2 s (S18 67.6 s; lead's gate <= 85 s) PASS; parity adj -0.574%; validate/einsum PASS. Engineer review: fine; pending the engineer's 12-net setup-calling run_mem (max handles vs S18, bar ~32k) + setup wall (guard 3.0 s), queued on the lock behind the porter's S20 chain. Then handoff to the porter (d6x_fragment_v2.py stacks after S19/S20).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_engineer2_v1/d6x`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/S18_d6x_v2_mem.py](code/S18_d6x_v2_mem.py)
- [code/d6x_fragment_v2.py](code/d6x_fragment_v2.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L588](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L588)
- [TEAM_LOG_2026-09-22_to_09-26.md#L593](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L593)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

## Related work

- shipped in: [Submission #332230: refitted physics scales and additional six-level products](../../../submissions/332230/README.md)

[Research graph](../../../graph/index.html#node=2026-09-25-additional-eligible-depth-six-products-save-0-54-work) · [Copy provenance](../../../PROVENANCE.json)

