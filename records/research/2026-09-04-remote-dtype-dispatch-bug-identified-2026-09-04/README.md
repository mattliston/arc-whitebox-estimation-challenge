# Remote dtype dispatch bug: identified 2026-09-04

2026-09-04T23:40Z · incident · failed · measured

Our new _mm helper compared a.dtype != fnp.float32. Native NumPy dtype objects compare equal to the scalar type, but flopscope-client0.12 returns a string from RemoteArray.dtype, and its fnp.float32 is a _DtypeLabel which does not equal a string. Future candidates must be tested at the RemoteArray boundary, not just with the native library's official version.

Recorded measurement:

    Our new _mm helper compared a.dtype != fnp.float32. Native NumPy dtype objects compare equal to the scalar type, but flopscope-client0.12 returns a string from RemoteArray.dtype, and its fnp.float32 is a _DtypeLabel which does not equal a string. Every remote product therefore took ordinary @. This was our implementation bug, not a scientific failure of the card. - Submission329798 raw3.17094401225404e-8, adjusted1.747652393096604e-8; impliedutil0.5511458. Local ordinary-matmul controlutil0.5519164 cold. - Client source _remote_array.py dtype property returns str; _dtypes.py _DtypeLabel.__eq__ does not equate strings. Native-only version checks did not exercise this interface. - Fix normalizes with str(a.dtype) == "float32"; numerical operations and learned payload remain unchanged. File: candidate_radial_gnn_remote_dtype_v2.py. - Actual numpy-free client/server test radial_remote_dtype_v2b.json: old comparison false, new comparison true;util0.3408311088142, residual0.1644413s,wall30.728s. This reproduces the expected Winograd bill. - Initial local server launch used its generic100MiB array default and rejected the arena before evaluation. It was rerun with the existing official4GiB per-array setting; the failed log is preserved.

## Programs

- [../../../submissions/329798/candidate_radial_gnn_remote_dtype_v2.py](../../../submissions/329798/candidate_radial_gnn_remote_dtype_v2.py)
- [build_remote_dtype_fix.py](build_remote_dtype_fix.py)
- [../../../submissions/329798/estimator.py](../../../submissions/329798/estimator.py)

## Evidence

- [REMOTE_DTYPE_BUG_2026_09_04.md#L1](../../../submissions/329798/REMOTE_DTYPE_BUG_2026_09_04.md#L1)

<details>
<summary>All result files and records (2)</summary>

- [REMOTE_DTYPE_BUG_2026_09_04.md](../../../submissions/329798/REMOTE_DTYPE_BUG_2026_09_04.md) · record
- [radial_remote_dtype_v2b.json](radial_remote_dtype_v2b.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-remote-dtype-dispatch-bug-identified-2026-09-04) · [Copy provenance](../../../PROVENANCE.json)

