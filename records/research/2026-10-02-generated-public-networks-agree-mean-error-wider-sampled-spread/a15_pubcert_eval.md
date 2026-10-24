## A15 certificate: the production truth path on 6 public networks (base torch card, final layer unless a layer is named; 1e-9)

| public network | random seeds of the run | error against the organizers' truth | error against the pipeline's 64M truth, noise subtracted | difference | difference in units of the modelled noise sd | (pipeline truth - organizers' truth)^2 over the predicted noise, layers 0 / 7 / 15 | the same along the mean direction (z^2), layers 3 / 7 / 11 / 15 | two halves over prediction, layer 15 |
|---|---|---|---|---|---|---|---|---|
| 537 | 72005371, 72005372 | 12.53 | 13.17 | +0.64 | +1.1 | 0.940 / 0.856 / 0.983 | 0.18 / 0.22 / 0.19 / 0.33 | 0.77 |
| 423 | 72004231, 72004232 | 13.08 | 12.80 | -0.28 | -0.5 | 0.972 / 1.119 / 1.120 | 0.02 / 0.37 / 0.06 / 0.00 | 1.28 |
| 35 | 9000351, 9000352 | 14.04 | 13.28 | -0.76 | -1.2 | 1.064 / 0.932 / 1.055 | 0.73 / 1.10 / 0.79 / 1.46 | 1.66 |
| 546 | 72005461, 72005462 | 17.00 | 17.82 | +0.82 | +1.3 | 1.012 / 0.856 / 0.988 | 0.16 / 0.34 / 0.07 / 0.48 | 1.11 |
| 455 | 72004551, 72004552 | 22.06 | 20.54 | -1.52 | -2.5 | 1.105 / 0.787 / 0.834 | 1.75 / 0.78 / 0.26 / 0.76 | 1.04 |
| 513 | 72005131, 72005132 | 22.35 | 23.00 | +0.65 | +1.1 | 1.042 / 1.172 / 1.075 | 0.91 / 0.04 / 0.10 / 0.02 | 0.93 |

Across the 6 networks, final layer: mean difference -0.07, root mean square 0.86, against a modelled noise sd of 0.61 (a pipeline that added the generated networks' extra spread would give about 2.5). Correlation between the error against the organizers' truth and against the pipeline's truth: 0.977.

| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| root mean square of the difference | 0.66 | 0.46 | 0.53 | 0.45 | 0.48 | 0.60 | 0.81 | 0.63 | 0.65 | 0.77 | 0.85 | 0.76 | 0.78 | 0.96 | 0.72 | 0.86 |
| mean difference | +0.21 | +0.12 | +0.17 | -0.15 | -0.24 | -0.26 | -0.43 | -0.21 | -0.25 | -0.24 | -0.38 | -0.27 | -0.16 | -0.14 | -0.05 | -0.07 |
| modelled noise sd (generated networks, two halves) | 0.64 | 0.57 | 0.63 | 0.49 | 0.45 | 0.40 | 0.40 | 0.39 | 0.52 | 0.51 | 0.51 | 0.55 | 0.59 | 0.58 | 0.71 | 0.61 |
| (pipeline - organizers)^2 over predicted noise, mean | 1.022 | 0.996 | 1.005 | 0.972 | 0.966 | 0.971 | 0.958 | 0.954 | 0.963 | 0.965 | 0.952 | 0.950 | 0.953 | 0.936 | 0.953 | 1.009 |
