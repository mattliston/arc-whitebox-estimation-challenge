# After response-row reduction, 4e-9 requires bill factor below .82743; 1e-9 requires raw below 1e-8

2026-09-07T03:20Z · finding · informational · derived

Planning arithmetic on the retained component's panel (mean NC raw 2.2454337457623213e-8, mean complete FLOPs 472413324017.2143, adjusted 4.83424098592912e-9) showed that uniform bill scaling would need factors below .8274308235031477 for 4e-9 and .6205731176273608 for 3e-9 (17.257% and 37.943% more savings), that at the .1 multiplier floor the unchanged raw still averages 2.2454337457623216e-9 so cost reduction alone cannot cross 2e-9, and that below 1e-9 needs mean raw below 1e-8 (a 55.465% reduction) or a 79.314% raw cut at unchanged bills. It fixed the evaluation discipline as the mean of per-net raw x max(.1, F/2^41) products.

Recorded measurement:

    Bill factor <.8274308235031477 (4e-9), <.6205731176273608 (3e-9); floor-limited adjusted 2.2454337457623216e-9; raw reduction >55.465% at the floor or >79.314% at unchanged bills; break-even for +19.327B of added cost about 3.93% raw, for +30B about 5.97%.

## Evidence

- [OBJECTIVE_FRONTIER_AFTER_RESPONSE_ROWS_V1.md#L18](OBJECTIVE_FRONTIER_AFTER_RESPONSE_ROWS_V1.md#L18)
- [OBJECTIVE_FRONTIER_AFTER_RESPONSE_ROWS_V1.md#L26](OBJECTIVE_FRONTIER_AFTER_RESPONSE_ROWS_V1.md#L26)
- [OBJECTIVE_FRONTIER_AFTER_RESPONSE_ROWS_V1.md#L66](OBJECTIVE_FRONTIER_AFTER_RESPONSE_ROWS_V1.md#L66)

<details>
<summary>All result files and records (1)</summary>

- [OBJECTIVE_FRONTIER_AFTER_RESPONSE_ROWS_V1.md](OBJECTIVE_FRONTIER_AFTER_RESPONSE_ROWS_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-07-after-response-row-reduction-4e-9-requires-bill-factor) · [Copy provenance](../../../PROVENANCE.json)

