# Four-bin angular-energy information: inconclusive gate, no advance

Phase 2, rank 35, preactivation 13 only. Judge completed September 6, 2026
at 20:26:34.872122 UTC (13:26:34 PDT). This is an oracle local-mean diagnostic,
not a final-layer raw/adjusted score or deployable estimator.

## Faithful acquisition and preserved technical failure

The initial v1 A run was correctly disqualified because an agent changed its
source provenance tuple after launch. Its failure/progress remain preserved.
See `CROSS_DEPTH_ENERGY_TECHNICAL_REPLAY_V2.md` for the incident and additive
replay wrappers. No accuracy arrays were produced or inspected from that run.

Identical fixed A was replayed under v2, followed by the already-prespecified B.
Each used 8,000,000 samples, unchanged 65,536 batches/seeds/float32 MPS forward
operations, and float64 CPU statistics. Both streams passed ALL source/input
hash, legacy byte-parity, third-diagonal, block-count, empirical algebra,
bin-support, finite/nonnegative variance and resource checks. Across replicas,
membership and conditional-variance stability also passed.

A: 156.185 s, peak RSS 1,665,548,288 bytes. B: 158.033 s, 1,586,200,576 bytes.
Each was the only MPS job, nice 10, three CPU threads, fallback disabled.

## Prespecified result

G is the weighted conditional Gaussian ReLU mean in the four fixed hidden-energy
bins. The bank comparator is the original NumPy bank post-13 reference after
the served maximum-zero/float32 transform, not a native-D4 byte-parity claim.
Errors are against each stream's same-sample empirical true ReLU mean.

| Local mean-error quantity | Bank | Four-bin oracle G |
| --- | ---: | ---: |
| A/B cross energy | 1.42635238844e-8 | 1.36818703285e-7 |
| Ordinary stream A MSE | 2.60648956896e-8 | 1.37318438182e-7 |
| Ordinary stream B MSE | 2.91880755922e-8 | 1.36606531371e-7 |
| Half squared replica difference | 1.33629617565e-8 | 1.43781491530e-10 |

Observed G/bank ratios: cross **9.59220908**, A **5.26832871**, B **4.68021713**.
All three accuracy inequalities fail their fixed .50/.65/.65 limits.
However, the bank noise/cross ratio is **.93686258**, above the required .10.
The mixture ratio is .01008036 and passes. The unchanged primary verdict is
therefore **INCONCLUSIVE_FIXED_ACQUISITION**, not a resolved information-gate
failure or a theorem excluding all conditioning. Do not relabel it as a pass
or silently loosen its noise criterion.

Neither material nor boundary-sensitive estimated plug-in-bias warning fired.
Eight-block leave-one-block cross ratios ranged 8.9316–10.8075; the approximate
jackknife standard error was 1.7348. These are diagnostics, not rigorous
confidence bounds or removal of common float32/finite-sample bias.

The single unconditional Gaussian oracle G0 has cross error 2.38859760023e-7,
1.74581219 times the four-bin error. Thus the observed conditioning does reduce
Gaussianization error, but the literal four-bin model remains far from the bank.
The physical omitted-third ratio 74.61795 is diagnostic only and does not decide
the primary gate.

## Decision and artifact identities

No expansion of this fixed acquisition, bins, thresholds, source, fit, carrier,
candidate or submission. The data do not justify that work. Keep its formal
inconclusive classification and move to genuinely distinct leads. No claim of
an affordable conditional covariance source was established.

- A NPZ SHA-256: `66092fde816e2b23bd7d0ada7f6afde780a3e10cb90719e8e398fd4eafc2e69c`
- B NPZ SHA-256: `8d744b6a63fe935a91abaf9b326187e6a29d95311767c159f94999e2738f81c2`
- Judge NPZ SHA-256: `2dbc23fada9f2b283bd771c1a9724f633c539d870196549123c441257fdc7f21`
- Judge JSON SHA-256: `85ee439631ed38bd108aed395f95ca21784842603ed5a9bf8f14a84be06a810c`
- Acquisition wrapper SHA-256: `4e5d82e1fa6fb7c88f7d074bcfe35b89e6acab508dec3e4edf6e616ffe88a76f`
- Judge wrapper SHA-256: `ebe7c6844873c1871cfa5882769dd29aae3485bbdc011ac24cfbe9a606305f91`
