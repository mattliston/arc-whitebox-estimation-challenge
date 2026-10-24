# Read-aware hub compression reduced supply correlation from about 0.62 to 0.08–0.17

2026-08-30 · experiment · rejected · measured

HUB v2 (read-aware OMP subset compression, KRHO=16, RANK=96): FAILS - q collapses to 0.08-0.17 (v0 cap-4: 0.62). ROOT TENSION now explicit: after one t<=1 spectral transport the content spreads over ~4700 columns; SUBSET selection (dynamically exact - each kept column is a genuine triad with correct future transport) covers ~1-2% of columns and loses the content; MIXTURES (rank-R recombination) cover it but break triad transport geometry (U-hadamard-X coupling does not commute with mixing; an SVD-of-read factorization has wrong dynamics). v4 escapes by carrying the FULL tensor (no factorization). Candidate resolutions for the sweep: (a) two-tier carrier (full recent + compressed old where dynamics fidelity matters less); (b) route t1-transport content into plane-class accumulators (exact plane transport, wrong distinct transport - vs dropping it); (c) lineage-structured mixing (mix only columns sharing sigma_r ancestry); (d) different carrier parametrization closed under both push and Mehler transport (e.g.

Recorded measurement:

    HUB v2 (read-aware OMP subset compression, KRHO=16, RANK=96): FAILS - q collapses to 0.08-0.17 (v0 cap-4: 0.62). ROOT TENSION now explicit: after one t<=1 spectral transport the content spreads over ~4700 columns; SUBSET selection (dynamically exact - each kept column is a genuine triad with correct future transport) covers ~1-2% of columns and loses the content; MIXTURES (rank-R recombination) cover it but break triad transport geometry (U-hadamard-X coupling does not commute with mixing; an SVD-of-read factorization has wrong dynamics). v4 escapes by carrying the FULL tensor (no factorization). Candidate resolutions for the sweep: (a) two-tier carrier (full recent + compressed old where dynamics fidelity matters less); (b) route t1-transport content into plane-class accumulators (exact plane transport, wrong distinct transport - vs dropping it); (c) lineage-structured mixing (mix only columns sharing sigma_r ancestry); (d) different carrier parametrization closed under both push and Mehler transport (e.g. Tucker with shared factors?). Handed to SOL as a design sweep (codex INBOX).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/p2_width1024_depth16/public_g19_attribution_v1/hub_carrier_v2.py](code/p2_width1024_depth16/public_g19_attribution_v1/hub_carrier_v2.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/hub_carrier_v0.py](code/p2_width1024_depth16/public_g19_attribution_v1/hub_carrier_v0.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_attribution.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_attribution.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v2b.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v2b.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py)
- [code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py)
- [code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py](code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py)

## Evidence

- [hub_carrier_v2.py#L1](code/p2_width1024_depth16/public_g19_attribution_v1/hub_carrier_v2.py#L1)

[Research graph](../../../graph/index.html#node=2026-08-30-read-aware-hub-compression-reduced-supply-correlation-about-0) · [Copy provenance](../../../PROVENANCE.json)

