# Cubic-free decoder-only estimator gives adjusted ratio 43.94708516 versus the matched reference

2026-09-08 · experiment · rejected · measured

The frozen decoder and response payload were tested end to end with their own complete bills on two development networks.

Recorded measurement:

    VERDICT_2026_09_08_PDT.md:
    source/weight, resource, first-Gaussian/P layer0 byte-parity and repeat gate.
    Context wall7.82964 /7.59921 /7.70142s; peakRSS1823178752B<4e9.
    observed effective thread count. Native residual.577–.618s is NOT a remote
    | Development net | D-only raw | D-only adjusted | Matched409f raw | Matched409f adjusted |
    |35|1.553026197829566e-6|1.553026197829566e-7|2.110483474536467e-8|4.5139473645742165e-9|
    |100|2.2944735474608648e-6|2.294473547460865e-7|1.9974337632158568e-8|4.240898903605451e-9|
    |mean|1.9237498726452155e-6|1.9237498726452154e-7|2.053958618876162e-8|4.377423134089834e-9|
    Mean adjusted ratio43.94708516; per-net ratios34.40506/54.10347. Both are
    decoder-training nets, not holdouts. This is observed MSE against1e9 truth,
    raw error is approximately93.66 times the same-frame bank raw. Additional
    Errors begin immediately at layer1 (~2.45–2.49e-8 versus bank~5.65–5.81e-10)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cp_free_donly_objective_check_v1/check.py](code/cp_free_donly_objective_check_v1/check.py)
- [code/build_response_rows_cost_v1.py](code/build_response_rows_cost_v1.py)
- [code/build_response_rows_cost_v2.py](code/build_response_rows_cost_v2.py)
- [code/build_response_rows_current_runtime_v1.py](code/build_response_rows_current_runtime_v1.py)
- [code/candidate_fullpair_runtime_d5_t8192_v1.py](code/candidate_fullpair_runtime_d5_t8192_v1.py)
- [code/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py](code/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py)
- [code/candidate_response_rows_cost_v1.py](code/candidate_response_rows_cost_v1.py)
- [code/candidate_response_rows_cost_v2.py](code/candidate_response_rows_cost_v2.py)
- [code/candidate_response_rows_current_runtime_v1.py](code/candidate_response_rows_current_runtime_v1.py)
- [code/candidate_response_rows_parent_cost_v1.py](code/candidate_response_rows_parent_cost_v1.py)
- [code/check_response_rows_adjusted_v1.py](code/check_response_rows_adjusted_v1.py)
- [code/cp_free_donly_objective_v1/estimator.py](code/cp_free_donly_objective_v1/estimator.py)

<details>
<summary>Additional programs and dependencies (7)</summary>

- [code/k3_radial_numpy.py](code/k3_radial_numpy.py)
- [code/radial_binary_compact.py](code/radial_binary_compact.py)
- [code/radial_compact_feed_fullpair_v1.py](code/radial_compact_feed_fullpair_v1.py)
- [code/radial_compact_radialfeed_v1.py](code/radial_compact_radialfeed_v1.py)
- [code/radial_cubic.py](code/radial_cubic.py)
- [code/radial_mehler.py](code/radial_mehler.py)
- [code/radial_response_rows_v1.py](code/radial_response_rows_v1.py)

</details>

## Evidence

- [VERDICT_2026_09_08_PDT.md#L3](VERDICT_2026_09_08_PDT.md#L3)
- [VERDICT_2026_09_08_PDT.md#L15](VERDICT_2026_09_08_PDT.md#L15)
- [VERDICT_2026_09_08_PDT.md#L19](VERDICT_2026_09_08_PDT.md#L19)
- [VERDICT_2026_09_08_PDT.md#L22](VERDICT_2026_09_08_PDT.md#L22)
- [VERDICT_2026_09_08_PDT.md#L28](VERDICT_2026_09_08_PDT.md#L28)

<details>
<summary>All result files and records (2)</summary>

- [VERDICT_2026_09_08_PDT.md](VERDICT_2026_09_08_PDT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config

</details>

[Research graph](../../../graph/index.html#node=2026-09-08-cubic-free-decoder-only-estimator-gives-adjusted-ratio-43) · [Copy provenance](../../../PROVENANCE.json)

