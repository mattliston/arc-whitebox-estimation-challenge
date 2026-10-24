# Input-gradient Gram observer fails all three gates: pooled raw ratio 1.00093491602506 versus the reference

2026-09-07T01:39:49Z · experiment · rejected · measured

An approximate input Dirichlet Gram K_l = Q_l * (W.T K_(l-1) W) (Q the degree-0..5 Hermite Gaussian gate kernel, K0 = Cpre0 * (1/4 + arcsin(rho0)/(2 pi))) was propagated along the bank's captured tapes and eight final features of D = (Kpre15 - C15)/(sigma sigma.T) were added to a 134->128->128->1 final head, fitted once (seed 930080, 40 epochs AdamW) on 20 TRAIN networks against a same-size retrained control. On held-out ranks 37/39/42/48 the sources were 0.09349% worse than the parent bank and 0.31501% better than the control with 2 of 4 parent wins, failing the fixed .65/.70/3-win advancement gates; the observer, features, order, window and fit were closed. Leading cost of the extra state would have been 62 n^3 = 66.572B (utilization 0.0302734375).

Recorded measurement:

    Pooled raw: parent 2.120159171214818e-8, control 2.1288475293110075e-8, sources 2.1221413419996646e-8; sources/parent 1.00093491602506, sources/control 0.99684985081411; per-rank sources/parent 1.00642209895 (37), 0.99501685779 (39), 0.99892281548 (42), 1.00252421102 (48). NumPy/MPS delta <= 2.9103830456733704e-11; execution 11.768510459 s; certificate max finite-difference error 5.459e-12.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/train_dirichlet_gram_probe_v1.py](code/train_dirichlet_gram_probe_v1.py)
- [code/dirichlet_gram_observer_v1.py](code/dirichlet_gram_observer_v1.py)
- [code/certify_dirichlet_gram_observer_v1.py](code/certify_dirichlet_gram_observer_v1.py)
- [code/compact_final_head_v1.py](code/compact_final_head_v1.py)
- [code/gaussian_shadow_disagreement_v1.py](code/gaussian_shadow_disagreement_v1.py)
- [code/generate_dirichlet_gram_row_v1.py](code/generate_dirichlet_gram_row_v1.py)
- [code/k3_radial_numpy.py](code/k3_radial_numpy.py)
- [code/prepare_dirichlet_gram_pilot_v1.py](code/prepare_dirichlet_gram_pilot_v1.py)
- [code/radial_mehler.py](code/radial_mehler.py)

## Evidence

- [DIRICHLET_GRAM_INFORMATION_PILOT_V1_CLOSED.md#L14](DIRICHLET_GRAM_INFORMATION_PILOT_V1_CLOSED.md#L14)
- [DIRICHLET_GRAM_INFORMATION_PILOT_V1_CLOSED.md#L21](DIRICHLET_GRAM_INFORMATION_PILOT_V1_CLOSED.md#L21)
- [DIRICHLET_GRAM_INFORMATION_DESIGN_V1.md#L144](DIRICHLET_GRAM_INFORMATION_DESIGN_V1.md#L144)
- [CONTINUATION_2026_09_06_1852_PDT.md#L49](CONTINUATION_2026_09_06_1852_PDT.md#L49)

<details>
<summary>All result files and records (7)</summary>

- [DIRICHLET_GRAM_INFORMATION_PILOT_V1_CLOSED.md](DIRICHLET_GRAM_INFORMATION_PILOT_V1_CLOSED.md) · result
- [DIRICHLET_GRAM_INFORMATION_DESIGN_V1.md](DIRICHLET_GRAM_INFORMATION_DESIGN_V1.md) · record
- [DIRICHLET_GRAM_ROOT_ACCEPTANCE_V1.md](DIRICHLET_GRAM_ROOT_ACCEPTANCE_V1.md) · record
- [DIRICHLET_GRAM_FIXED_FIT_ACCEPTANCE_V1.md](DIRICHLET_GRAM_FIXED_FIT_ACCEPTANCE_V1.md) · record
- [dirichlet_gram_small_certificate_v1.json](dirichlet_gram_small_certificate_v1.json) · result
- [result.json](result.json) · result
- [CONTINUATION_2026_09_06_1852_PDT.md](CONTINUATION_2026_09_06_1852_PDT.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-07-input-gradient-gram-observer-fails-all-three-gates-pooled) · [Copy provenance](../../../PROVENANCE.json)

