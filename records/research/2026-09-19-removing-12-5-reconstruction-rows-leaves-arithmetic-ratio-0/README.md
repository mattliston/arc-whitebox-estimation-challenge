# Removing 12.5% of reconstruction rows leaves arithmetic ratio 0.9998682568 after a kernel-depth change

2026-09-19 · finding · rejected · measured

The current six-level push kernel requires 1024 output rows; reducing rows can force a shallower product that erases the expected cost saving.

Recorded measurement:

    Phase 2 only. This source check follows the completed orthogonal cubic
    the unlearned older frame, raw worsened 3.23645% and 0.39963%; the first
    missed its <=3% gate. Its 14.17%/14.07% output-row push reduction was a
    _rule_depth admits depth6 for wide push only at exactly 1024 output rows
    (and dimensions divisible by64). _packed_product_4 has the matching
    1024-row contract. Other row counts use at most depth4. The immutable
    weight-cache grants also require a 1024-row left operand at their four
    products. The existing live masks are rounded to32, but that alone does
    For a 1024x384 times384x8128 old-factor reconstruction, the current rule
    selects depth6 and estimates 3,420,852,050 arithmetic operations. Reducing
    the output to896 rows selects depth4 and estimates 3,420,401,376: ratio
    0.9998682568, despite removing12.5% of rows. At928 rows the ratio is
    1.034940768; at960 it is1.070013279. At832/864 rows there are modeled
    savings of about7.03%/3.52% for this product. Shapes2048/6144/16384 show
    Other source alternatives inspected are already measured: CP-only q4
    one-high matrix recursion worsened older raw means30.57x/35.44x; combined
    CP+star q4 static projection had scalar relative error0.26798 versus0.05;
    remains Lazy #331599 at adjusted3.8164492657795035e-9. The approximately
    1e-9 goal stays active and unachieved, with qualified submissions preauthorized.

## Programs

- [audit.py](audit.py)
- [../../../submissions/331599/estimator.py](../../../submissions/331599/estimator.py)

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L9](RESULT.md#L9)
- [RESULT.md#L10](RESULT.md#L10)
- [RESULT.md#L15](RESULT.md#L15)
- [REEXAMINATION_INDEX_20260919.md#L1110](../../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L1110)
- [REEXAMINATION_INDEX_20260919.md#L1116](../../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L1116)

<details>
<summary>All result files and records (4)</summary>

- [RESULT.md](RESULT.md) · record
- [closeout.json](closeout.json) · result
- [result.json](result.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-19-removing-12-5-reconstruction-rows-leaves-arithmetic-ratio-0) · [Copy provenance](../../../PROVENANCE.json)

