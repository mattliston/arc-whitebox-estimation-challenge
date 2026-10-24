# One-scalar collective-skewness tracker matched the transported non-Gaussian component

2026-08-30 · experiment · informational · measured

THE COLLECTIVE-SKEWNESS TRACKER DERIVED AND VALIDATED (19:45). Split measurement: 97% of collective kappa3 is TRANSPORTED input non-Gaussianity (Gaussian-input part = 0.304 of 9.696) - the ridge fails for generation but is EXACT for transport: one-step law kappa3(s') = (G/sigma)^3 kappa3(s) + fresh predicts 96.4%. Full scalar recursion with ONE universal constant (fresh_true ~ 2.9 x fresh_ridge, measured band 2.2-3.6): free-run from ZERO tracks truth within +-2% from L8 on (r35: ratios 1.00/1.01/1.02/1.00/0.99 at L8/11/12/13/14) and HELD-OUT on r100 at 1.001/0.987 (L8/L12) with everything frozen.

Recorded measurement:

    THE COLLECTIVE-SKEWNESS TRACKER DERIVED AND VALIDATED (19:45). Split measurement: 97% of collective kappa3 is TRANSPORTED input non-Gaussianity (Gaussian-input part = 0.304 of 9.696) - the ridge fails for generation but is EXACT for transport: one-step law kappa3(s') = (G/sigma)^3 kappa3(s) + fresh predicts 96.4%. Full scalar recursion with ONE universal constant (fresh_true ~ 2.9 x fresh_ridge, measured band 2.2-3.6): free-run from ZERO tracks truth within +-2% from L8 on (r35: ratios 1.00/1.01/1.02/1.00/0.99 at L8/11/12/13/14) and HELD-OUT on r100 at 1.001/0.987 (L8/L12) with everything frozen. Cost ~1e5 flops/layer - free. ALSO EXPLAINS v4's stable 2/3: its collective transport gain is ~2.5%/layer low ((1.074/1.05)^12 = 1.31 = the deficit). DEPLOYMENT: K3IMPL hook (implied collective kappa3 per layer from the tensor), run_collective_patch.py (pre_l += (K3S - implied) * w^2 x w), full chain running.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g19_attribution_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/run_collective_patch.py](code/run_collective_patch.py)

## Evidence

- [run_collective_patch.py#L1](code/run_collective_patch.py#L1)

[Research graph](../../../graph/index.html#node=2026-08-30-one-scalar-collective-skewness-tracker-matched-transported-non-gaussian) · [Copy provenance](../../../PROVENANCE.json)

