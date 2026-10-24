# Depth seven on fitting product classes saves only 0.55% work for 22% more local backend time

2026-09-25T08:33Z · experiment · rejected · measured

Additional multiplication depth doubles product rounding and adds requests while larger packets exceed the arena. The current fitting-class package was not recommended.

Recorded measurement:

    - 2026-09-25 01:33 [engineer] E-72 DEPTH 7 PRICED, NOT RECOMMENDED (RESULT.md E-72; e72_depth7_bench.py, S23 tree kernel in-process): per product 1024^3 bill -4.9%, backend +90%, ops +15, error x2.2; 1024x1024x2048 -6.1%; 512-scale -0.5..-2.6%; 4096-wide colpack does not fit the arena at depth 7. All eligible classes (63 products/net): -0.55% of the bill for +3.5 s local backend (+22%), +945 requests, error x2.2 on those products. Model bill(7)-bill(6) = -0.112 mkn + 22.0(mk+kn) + 22.3 mn (within 3% of measured). The big depth-5 read/push products cannot reach depth 6 by padding (A tree 224M floats vs a 90M region). Far below the -2% bar.
    - 2026-09-25 01:34 lead: DEPTH-7 CLOSED (engineer E-72, measured per product at depth 6 -> 7 on S23's tree kernel): 1024^3 bill -4.9% but backend +90%, rounding x2.2. All fitting S23 classes (63 products/net): -0.55% of bill for +3.5 s local backend (+22%), +945 requests, 2.2x rounding. The 1024x1024x4096 colpack packets don't fit the arena at depth 7. The depth-5 bank read/push products (inner dims 6-8k) can't reach depth 6 by padding (A tree ~224M floats vs a 90M region). Model: bill(7)-bill(6) = -0.112 mkn + 22.0(mk+kn) + 22.3 mn (within 3%). The kernel depth lever is SPENT on our product mix; bill room is in the card's product count and shapes. The leaders' higher ns/kflop is not deeper Winograd on our shapes.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_engineer_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/e72_depth7_bench.py](code/e72_depth7_bench.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L877](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L877)
- [TEAM_LOG_2026-09-22_to_09-26.md#L878](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L878)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-depth-seven-fitting-product-classes-saves-only-0-55) · [Copy provenance](../../../PROVENANCE.json)

