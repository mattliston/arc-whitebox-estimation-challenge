# Combined v1: boundary feasibility failed

Frozen candidate `09dd5166b1edae161ef0b877d66df99b1891689ed2465280cf646d45dadacbfd` remains unqualified. The original [boundary assessment](boundary_assessment.json) and the single fixed [contemporary-pair assessment](paired_boundary_assessment.json) both pass receipt checks and fail only the unchanged residual projection limit of 0.35 seconds.

| Comparison | Residual deltas, ranks 35 / 100 | Projected residual | Projected wall |
|---|---:|---:|---:|
| Historical S32 control | +0.0518 / +0.0489 s | 0.492650751 s | 97.326862801 s |
| Fixed fresh S32 control | +0.0455 / +0.0445 s | 0.467450751 s | 97.907555153 s |

The fresh pair used the prospectively frozen order S32 then unchanged candidate, each on ranks 35 and 100. Candidate request increases remain 15,701 / 16,659. Candidate setup was 3.364 seconds with 673 total and 649 fused bindings; CPU disagreement was 4.2171 percentage points, below the 5-point limit. Both runs and the assessor retired successfully with zero added swap; maximum pair RSS was 5,823,660,032 bytes. Full boundary/native bills matched. No boundary output-byte claim is made.

The pair was an explicitly outcome-informed resolution, not a retroactive pass of the historical comparison. Both failed receipts remain unchanged. Their SHA256 pins are `f8ab9902712e8d2f903713c73f0e2d3349300e82bb298d88cac54e1524903e22` and `9091ca7be8089e8e170540309dd558433a8116d76648b6b854d068af526afbce`.

The prior TRAIN8 full-candidate S32 score screen passed, while its incremental component CI gate failed and remains false. Neither result overrides this resource failure. No VALID40, worker20, full serving qualification or submission is admitted for unchanged v1. No further timing reruns of v1 are planned. The separately versioned batched-v2 repair must establish its own exact native parity and resource results. S32 #332738 remains the bank.

This closeout checked source/JSON pins and retired guards only; no arrays, models or numerical jobs were executed by the reviewer.
