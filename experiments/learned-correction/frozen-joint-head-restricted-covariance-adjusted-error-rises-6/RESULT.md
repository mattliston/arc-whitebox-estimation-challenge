# Frozen joint head, covariance-only endpoint: negative

The separately declared covariance-only transfer fails the native TRAIN35/100
screen: pooled raw error +3.47638%, full billed arithmetic +3.40112%, and
adjusted error +6.99703%, with zero of two adjusted wins. The fixed gate was
two wins and at least 1% pooled adjusted improvement. No expansion, refit,
amplitude/band/plane selection, validation or submission follows.

The 9184-parameter historical joint head is unchanged. All three predicted
planes affect only the current pair map's covariance output, including its
existing non-kink GN response. The parent same-input moment outputs, radial
source and ordinary feature calculations are preserved. Later covariance
recurrence sees the changed covariance. This is a separate consumer test;
the previous full moment transfer and its original information gate remain
failed.

| TRAIN rank | Raw MSE | Full FLOPs | Adjusted | Adjusted ratio |
|---|---:|---:|---:|---:|
| 35 | 1.4888389107301385e-8 | 291429383117 | 1.9731096713925147e-9 | 1.07858281123 |
| 100 | 1.569295578919443e-8 | 290643306694 | 2.0741265699935955e-9 | 1.06190390954 |

The disabled rank35 control matches all 16 saved S31 prediction rows byte
for byte and its complete 281856471787 FLOP bill. Enabled predictions retain
the first eight rows exactly and all outputs are finite. Independent scalar
score recomputation agrees. Source/helper inverses and learned bytes pass.
Signed/asymmetric native fixtures agree with independent float64 algebra
within 4.46e-10; full virtual pair-plus-GN differences are within 3.57e-8,
inside the declared float32 rounding bound. Inputs and coefficient buffers
are unchanged, zero corrections are exact, and diagonals remain zero.

Use `unit_v3_result.json` as the successful unit receipt. The original unit
harness omitted scratch initialization; v2 fixed that and passed arithmetic
but failed JSON serialization of a NumPy scalar. v3 changes only reporting.
Original source, logs, failed guards and the incomplete `unit.json` remain
preserved, alongside HARNESS_FIX.md and RECEIPT_FIX.md.

Seven guarded groups retired: five completed and two failed in harness or
reporting. Peak sampled RSS was 5,069,422,592 bytes; added swap was zero.
No worker, approval or grading is pending. Candidate SHA256:
7cdb2aeeb0427a809cdf1c1deff5fa60710aaf88e3f9b3a60c015ba195066709.

S31 submission 332517 remains the bank at 2.1736465056908115e-9 adjusted.
No fresh official poll was made. The 2e-9 and approximately 1e-9 goals remain
unmet, and submission authorization remains in force.
