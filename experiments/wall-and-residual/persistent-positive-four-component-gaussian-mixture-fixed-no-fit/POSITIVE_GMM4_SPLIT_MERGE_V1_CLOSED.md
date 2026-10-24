# Persistent positive GMM4: fixed no-fit raw test failed

Phase 2 only, September 6, 2026. Root executed the frozen, independently
reviewed source at about15:03 PDT. All four model runs completed, source and
weight digests remained unchanged, and the numerical/resource checks passed.
The scientific accuracy gates failed on both networks. No fit, component
expansion, native port or submission follows this fixed proposal.

The complete model was an endogenous positive mixture with four retained
full conditional covariances, fixed Hermite12 Gaussian maps, four power steps
for one curvature-selected half-variance split per nonfinal layer, and the
trace-aware isotropic scalar-third merge criterion. It had no CP carrier,
teacher, star, radial state, learned P/Q head or current-target feature.
The control was the same scalar/map/precision/factor policy with one Gaussian
and no splitting or merging. Both arms on both networks were frozen before
the stored1e9-sample research reference means were decoded.

## Final-layer served-float32 raw MSE against existing1e9 truths

| Network rank | Single-Gaussian control | GMM4 | GMM4 / control |
| --- | ---: | ---: | ---: |
| 35 | 2.89617838895e-6 | 2.61823343950e-6 | .9040304456 |
| 100 | 4.63004606124e-6 | 4.25983219808e-6 | .9200409978 |

The fixed requirements were BOTH raw<=6e-8 AND ratio<=.25 on EACH network.
Neither is close. The observed8–10% improvement over an untrained Gaussian
control is not competitive with the existing learned full-pair bank. These
are raw MSEs, not adjusted scores, panel means or official submission scores.
Do not multiply by the provisional cost envelope and advertise a validated
adjusted candidate. No new Monte Carlo or training was performed.

## Qualification and complete cost distinction

- Rank35 single/GMM wall: .7533/4.0350s; rank100: .8307/3.9105s. Whole smoke
  including scoring took9.6163s. These are offline NumPy timings, not grader
  persistent wall or residual-time gates.
- Maximum process RSS1,203,224,576 bytes, below2GiB. One nice10 NumPy-only
  worker; requested two BLAS threads via environment and threadpool_limits.
  Empty backend instrumentation does not independently prove actual count.
- Each GMM used115 full FP32 products,59 FP64 Choleskys,86 pair maps,
  335 merge-pair comparisons and280 direction actions, exactly the design.
- All Choleskys succeeded on first attempt. No jitter, eigensolver or retry.
  Mean/variance roundoff lifts were zero; largest diagonal-tail roundoff lift
  8.3267e-17. Largest factor-rounding relative error2.6066e-8; no positive
  variance coordinate was zeroed by factor underflow.
- Retained L32 factors own PSD mixture state. Rounded affine/child matrices
  remain approximate numerical inputs, not separately PSD-certified objects.
- Full provisional charge324,194,464,597.33 FLOPs, utilization .147426574,
  includes a35B auxiliary reserve. This remains an unmetered cost envelope.
  The tiny native Cholesky operation check does not qualify a whole estimator.

## Provenance and closure scope

- Prototype SHA16cc38b25bcb6ff0796b5630eb538bb60ccaba166c97be944994ce905d317a58.
- Design SHAba16f81c2e33d1eb63aaa71082493018e7362c64ec5ae4b07fb3e6e887a7a565.
- Result `positive_gmm4_split_merge_smoke_v1/result.json`,
  SHA0444451ea51165d3b305c5b32270c5c1718d9e042b50c7553714548e7dfa387c.
- Prediction-freeze receipt SHA4cc2af694342abe5c128f62b8d656a5eb778a639ff28e17c21ad6866da920cd8.
- Weights: rank35 SHA14563b956b4e326aee081263077c0dd1dcb86037a0b7870dc58a5e33865f47f2;
  rank100 SHAc4923171c8ddef28396aabd70335dad29141361efc13865ccb660a4a57421ee5.
- Existing truths: rank35 SHAec47c040198070cfe936126d18d24d02a94a6fef1da14795173c23d199e05f61;
  rank100 SHA637a63d46ac5480be50cca3f0021a1574eef80fa580ce98f1a8cc69b0b86450c.

The frozen no-fit component/source/direction/power/map/merge round is closed.
This is not a general impossibility theorem for learned mixtures or models
with a non-Gaussian transverse state. Such a state would need genuinely new
source information and full cost accounting, not free inherited CP trajectories.
Official bank #329926 remains4.982231460183304e-9 adjusted (not freshly polled);
the <1e-9 goal remains active and unmet. No threshold bank was earned here.
