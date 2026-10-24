# Late-layer exactness

Exact evaluation and row cuts at the last layers (layers 14 and 15).

| Date | Experiment | Programs | Outcome |
|---|---|---:|---|
| 2026-09-23 | [Layer-15 third-cumulant response cut \|z\| < 3.5: native adjusted error improves 0.66% on 40 networks](layer-15-third-cumulant-response-cut-z-3-5/README.md) | 3 | adopted |
| 2026-09-23 | [Local final-layer error prediction is null: held-out R^2 -0.0001](local-final-layer-error-prediction-null-held-out-r/README.md) | 30 | rejected |
| 2026-09-30 | [Final Jensen lower-bound projection raises 100 of 106 outputs and increases raw error 0.318748%](final-jensen-lower-bound-projection-raises-100-106-outputs/README.md) | 2 | rejected |
| 2026-10-10 | [Additional layer-15 inputs rejected: opening all ten masks adds 7.5% MSE](additional-layer-15-inputs-rejected-opening-all-ten-masks/README.md) | 3 | rejected |
| 2026-10-11 | [Exact final-layer row and push dead cuts rejected: MSE changes 0.00%/-0.01% and +0.02%](exact-final-layer-row-push-dead-cuts-rejected-mse/README.md) | 6 | rejected |
| 2026-10-16 | [Removing final-layer row cut or common-read split: no accuracy gain for +1.354/+0.372% bill](removing-final-layer-row-cut-or-common-read-split/README.md) | 11 | rejected |
| 2026-10-16 | [Layer-15 row cut z lowered 3.5 to 3.0: 24 seeds net -0.315% at bill -0.360%; 2.75 and 2.5 are a cliff](layer-15-row-cut-z-lowered-3-5-3/README.md) | 3 | adopted |
| 2026-10-17 | [Gaussian early termination reduced bill but increased final error by 1,613% to 123,169%](gaussian-early-termination-reduced-bill-but-increased-final-error/README.md) | 3 | rejected |
| 2026-10-19 | [Final-layer third-cumulant row threshold reduced from 3.5 to 3.0: -0.177% adjusted on 32 networks](final-layer-third-cumulant-row-threshold-reduced-3-5/README.md) | 3 | adopted |
| 2026-10-19 | [Final-layer old-pair truncation keeping 70% of weight: +5.56% raw error for -0.318% FLOPs](final-layer-old-pair-truncation-keeping-70-weight-5/README.md) | 4 | rejected |
| 2026-10-19 | [Layer-14 exactness (old tail z 3.0 + young crop z 3.0): raw -0.352%, bill +0.270%, adjusted -0.079% on 28 ranks](layer-14-exactness-old-tail-z-3-0-young/README.md) | 9 | adopted |
| 2026-10-19 | [Projection-row dead cut z 3.5 restored on the row-cut lineage: +0.159% adjusted (22 ranks); at layer 14 only, null](projection-row-dead-cut-z-3-5-restored-row/README.md) | 3 | rejected |
| 2026-10-19 | [Layer-13 dead-row crop at threshold 3.0: -0.095% adjusted; extending to layer 12 gives +0.013%](layer-13-dead-row-crop-threshold-3-0-0/README.md) | 5 | adopted |
| 2026-10-19 | [Late-layer restores: covariance floor 0.0005, old-tail negative side, column-cut restore, star rank 8 all null or worse](late-layer-restores-covariance-floor-0-0005-old-tail/README.md) | 5 | rejected |
| 2026-10-20 | [Sensitivity-ranked terminal third-cumulant row cuts cost +0.7..+3.7% MSE for -0.1% FLOPs](sensitivity-ranked-terminal-third-cumulant-row-cuts-cost-0/README.md) | 3 | rejected |
| 2026-10-21 | [Terminal row cutoff 2.5 or 3.5: +3.147% or +0.190% adjusted on the deployed first-128 source](terminal-row-cutoff-2-5-or-3-5-3/README.md) | 2 | rejected |
