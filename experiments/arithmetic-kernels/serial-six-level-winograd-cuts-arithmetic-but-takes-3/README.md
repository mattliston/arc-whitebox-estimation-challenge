# Serial six-level Winograd cuts arithmetic but takes 3.1969129249046593x the four-level wall

2026-09-09 · experiment · rejected · measured

Real NumPy-free clients tested one cold and two warm calls per kernel; lower workspace did not compensate for excessive request-driven wall time.

Recorded measurement:

    RESULT.md:
    per role, real NumPy-free FlopScope-client.12 and the bank's OpenBLAS server.
    | Role | Warm productF | Maximum warm context wall | FP64-reference RMS error | Sampled combined peakRSS |
    | CurrentD4 |10409197568|0.295003708s|1.9912512466005153e-5|1440022528B|
    | BreadthD6 |8833219584|0.735729750s|7.023140158240704e-5|3565387776B|
    | Serial-topD6 |8833219584|0.943101167s|7.023140158240704e-5|987529216B|
    Serial maximum warm wall divided byD4 is3.1969129249046593, versus the
    prospective ceiling1.35. Its maximum residual0.004388582s is below0.20s.
    remains330390 at4.53886349950395e-9 adjusted. Strict official<1e-9 is unmet.
    the client.12 constructor has no wall-limit keyword, and role limits needed a

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/current_d50_serial_d6_kernel_v1/client.py](code/current_d50_serial_d6_kernel_v1/client.py)
- [code/current_d50_serial_d6_kernel_v1/kernel.py](code/current_d50_serial_d6_kernel_v1/kernel.py)
- [code/current_d50_serial_d6_kernel_v1/run.py](code/current_d50_serial_d6_kernel_v1/run.py)
- [code/birth_direction_fit_v1/build_terminal_v1/estimator.py](code/birth_direction_fit_v1/build_terminal_v1/estimator.py)
- [code/packed_arena_generic.py](code/packed_arena_generic.py)

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L5](RESULT.md#L5)
- [RESULT.md#L8](RESULT.md#L8)
- [RESULT.md#L10](RESULT.md#L10)
- [RESULT.md#L11](RESULT.md#L11)

<details>
<summary>All result files and records (2)</summary>

- [RESULT.md](RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config

</details>

[Research graph](../../../graph/index.html#node=2026-09-09-serial-six-level-winograd-cuts-arithmetic-but-takes-3) · [Copy provenance](../../../PROVENANCE.json)

