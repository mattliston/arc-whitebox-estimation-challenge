# Fixed covariance symmetry: closed negative at fixed development gate

One insertion symmetrizes full Cpre/Bpre in FP64 before their consumers.
The directed K21 plane, final diagonal path, rank384/age4, all learned
parameters, Lazy schedule and kernels are unchanged. No PSD repair.
This tests numerical rounding, not a new physical source.

| Rank | Lazy raw MSE | Symmetry raw MSE | Raw ratio | Own-cold adjusted ratio |
| --- | ---: | ---: | ---: | ---: |
| 35 | 1.870602849007794e-08 | 1.875453353257169e-08 | 1.00259301661 | 1.0028993619 |
| 100 | 1.842453417527493e-08 | 1.85786822762268e-08 | 1.00836645852 | 1.00867240662 |

Pooled adjusted ratio: **1.005774160317667**.
The fixed gate requires both raw/adjusted wins and pooledadjusted<=.99.
These are two reused ordinary1B development networks, not independent
generalization evidence or an official result. Own complete cold bills
include all symmetry arithmetic and any changed selector geometry.

Rank35 cold FLOPs: 398080917697 -> 398202552513 (+121634816).
Rank100 cold FLOPs: 400893063472 -> 401014698288 (+121634816).

All seven explicit entrywise certificates, symmetry/idempotence,
diagonal and quadratic-form checks pass; maximum scaled quadratic error
1.06581410364015e-14.
Three malformed-input rejections, exact unit arrays/bills, complete
network repeat/warm-cold100 bytes, first-layer parent mean equality,
finite nonnegative outputs and cache retirement pass. The six sealed
source/protocol files remain unchanged. Predictions and bills were
sealed before opening the two pinned truth files.

Four predictions total 1565900581246 FLOPs; peak RSS 6073057280 bytes.
All four guarded stages exit0 and retire; zero added swap.

No wider panel, fitted rescue, strength/location/precision/kernel sweep,
production client, package or upload follows this failed variant.

Official bank remains Lazy #331599 adjusted 3.816449265779503e-09.
The4e-9 milestone is confirmed;3e-9,2e-9 and approximately1e-9 remain unmet.
Future qualified Phase2 board uploads have the user’s standing preauthorization.
