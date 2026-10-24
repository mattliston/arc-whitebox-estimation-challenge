# Fixed NODE/P1 independent-panel verdict

Recorded 2026-09-09, after the native and independent scoring supervisors completed.
This is Phase 2 local evidence, not an official score or production qualification.

Both predeclared strict group gates pass against the exact archived P1 baseline.
The baseline's own cold/warm bills and score were recomputed exactly. All candidate
predictions, bills, numerical domains and selectors were verified before truth
access. There was no refit; the old NODE-only pilot still failed its own gate.

| Independent 4M truth group | P1 noise-corrected adjusted | NODE/P1 noise-corrected adjusted | Candidate / P1 |
| --- | ---: | ---: | ---: |
| Public 8 | 4.896376273904718e-9 | 4.77415874629813e-9 | 0.9750391880097231 |
| Virgin 6 | 3.839813677062407e-9 | 3.7345329733435845e-9 | 0.9725818196992919 |
| Combined 14 | 4.4435637324008696e-9 | 4.328604843603325e-9 | 0.9741291234422259 |

Combined mean noise-corrected raw MSE is 2.0305056229588224e-8 versus
2.0847972531261027e-8 for P1. Complete mean candidate FLOPs are
467697788399.9286 versus 467682934475.3571 for P1. Each reported adjusted
mean is the mean of per-net raw times its own max(0.1, FLOPs/2^41), not a
product of the two population means. The small cost increase means that this
panel establishes a beneficial tradeoff, not exact Pareto dominance. Independent
truths are not all untouched model-selection nets; the existing panel overlap
disclosures remain applicable.

Scoring integrity PASS; retention PASS. No milestone achieved, no submission
authorized by this receipt alone, and production qualification remains pending.
The native40 fixed-1e9-truth result remains 4.451951498340901e-9 adjusted,
2.092035131674627e-8 raw, 468313646101.175 mean complete FLOPs.

Independent score supervisor: 116.26108766533434 seconds, maximum sampled
combined RSS 154615808 bytes, zero swap growth, maximum completed sampling
interval 0.3187950002029538 seconds. All experiment and observational groups
retired; input/source hashes unchanged. Score child: 77.66815949976444 seconds,
106037248 bytes peak RSS.

Seals:

- panel_v2/score/result.json: `379f7fa393f81dfe9af0cd30cb9d8ad5a3a08e71f34242c58bc30f540569023c`
- panel_v2/score/completion.json: `c539b369bd5e8ca15ee6213c6acb9e4e208e3fcaeda52fa3d146dbc1d0d3733f`
- panel_v2/score_watch/result.json: `99832e1036940913306ad1a2580796d24553a40875988f50c56156eaa771df46`
- panel_v2/score_watch/completion.json: `a87fa8b7d85d7c969971c2807f79c01d1df5df8ba8cbd6aa1892da21707a7c8e`

Decision: retain the fixed combination as a local component. Continue the
already-built partial-D6/P1 complete native objective test independently; its
result is not assumed to compose with NODE/P1. Preserve P1 and all failed arms.
