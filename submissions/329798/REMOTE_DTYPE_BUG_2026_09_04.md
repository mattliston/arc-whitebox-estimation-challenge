# Remote dtype dispatch bug — identified 2026-09-04

Our new `_mm` helper compared `a.dtype != fnp.float32`. Native NumPy dtype
objects compare equal to the scalar type, but flopscope-client0.12 returns a
string from RemoteArray.dtype, and its fnp.float32 is a `_DtypeLabel` which
does not equal a string. Every remote product therefore took ordinary @.
This was our implementation bug, not a scientific failure of the card.

Evidence:

- Submission329798 raw3.17094401225404e-8, adjusted1.747652393096604e-8;
  impliedutil0.5511458. Local ordinary-matmul controlutil0.5519164 cold.
- Client source `_remote_array.py` dtype property returns str; `_dtypes.py`
  `_DtypeLabel.__eq__` does not equate strings. Native-only version checks
  did not exercise this interface.
- Fix normalizes with `str(a.dtype) == "float32"`; numerical operations and
  learned payload remain unchanged. File: candidate_radial_gnn_remote_dtype_v2.py.
- Actual numpy-free client/server test `radial_remote_dtype_v2b.json`:
  old comparison false, new comparison true;util0.3408311088142,
  residual0.1644413s,wall30.728s. This reproduces the expected Winograd bill.
- Initial local server launch used its generic100MiB array default and
  rejected the arena before evaluation. It was rerun with the existing
  official4GiB per-array setting; the failed log is preserved.

Future candidates must be tested at the RemoteArray boundary, not just with
the native library's official version. No claim that329798 itself improved
the board. Best actual submission remains329790 until a new official grade.
