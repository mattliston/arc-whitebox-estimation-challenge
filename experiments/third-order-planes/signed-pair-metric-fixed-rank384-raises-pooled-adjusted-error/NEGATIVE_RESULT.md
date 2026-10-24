# Fixed rank384 signed-pair metric: development failure

The entry-metric-only variant failed its prospective actual-objective gate.
Rank384, age4, trained heads, physical factor coordinates, Lazy schedule,
all original consumers and kernels were held fixed. This supplies the
previously absent same-rank comparison; the original PAIR256 joint
rank-and-metric failure remains unchanged.

| Development rank | Lazy raw MSE | Pair384 raw MSE | Raw ratio | Own-cold adjusted ratio |
| --- | ---: | ---: | ---: | ---: |
| 35 | 1.870602849007794e-08 | 1.867859647500963e-08 | 0.998533520085 | 0.999268168499 |
| 100 | 1.842453417527493e-08 | 1.88229765452842e-08 | 1.02162564145 | 1.02237056117 |

Pooled adjusted ratio: **1.010772448328348**.
Rank35 improves slightly; rank100 worsens. Both raw and adjusted
improvement on both networks and pooled adjusted<=.99 were required.
These are two reused ordinary1B development networks, not an
independent panel, official result, or universal metric-capacity test.

Rank35 complete cold FLOPs: 398080917697 -> 398373796713 (+292879016, 0.073573%).
Rank100 complete cold FLOPs: 400893063472 -> 401185375192 (+292311720, 0.072915%).

Independent explicit small third tensors certify the Gram and range
action with maximum relative error1.5600663146273522e-7. Signed,
near-coalesced, coalesced and zero-coefficient cases, original physical
projection bytes, exact repeat bills and four input rejections pass.
Full outputs are finite/nonnegative; first5 parent layers, own repeat,
warm/cold100 output bytes and cache retirement pass. Changed later
outputs use their own actual complete bills. All predictions and bills
were sealed before the two original truth files were opened.

Four full calls total 1566584423454 billed prediction FLOPs.
Peak RSS including self-reports: 6501187584 bytes. Zero added swap.
All four supervised stages exit0 and retire their process groups.
No NumPy-free client, larger panel, fit, metric refresh, scaling/blending,
rank/age/precision/QR sweep, package or submission follows this failure.

The entry-frozen per-pair Gram omits cross-pair tensor terms and is not
refreshed after transport. The result closes this exact fixed variant,
not every cancellation-aware or endpoint-aware representation.

Official bank remains Lazy #331599 at 3.816449265779503e-09 adjusted.
The4e-9 milestone is confirmed. Approximately1e-9 remains active and unmet.
