# CLOSED — causal matrix-error observer

September 6, 2026, 02:20 PDT. Phase 2 only. No submission or forum post.

The fixed read-only Gaussian-tangent error memory passed its prerequisites:
synthetic tangent errors <=7.43e-10; exact linearity <=3.66e-16;
small MPS/NumPy output <=2.37e-7 and gradient <=3.03e-7 relative.
Current rank35/100 zero observer preserved the actual scalar-final P/Q bank
byte-exactly. Fixed nonzero wide MPS/NumPy relative errors <=9.53e-7.
Twenty-four final rows were captured without targets in80.31s using two
nice10 CPU workers. All15 prefix feature/sigma fields and existing final
prediction/host/sigma/delta/h matched byte-for-byte. The actual scalar-final
x and sigma replace the old full-final tape entries; other tape fields stay
unchanged and rounded as originally stored. Final C is not consumed by
the final observer response.

Both locked100-epoch fits completed,87.61s total on one MPS worker,
peak host RSS3,564,699,648 bytes. Existing1e9 final means only,20train and
37/39/42/48 held out. Frozen bank and main P/Q never modified.

| Holdout | Bank raw MSE | Matrix memory / bank | Local control / bank |
|---|---:|---:|---:|
|37|2.1281180954e-8|.9502870722|.9924554326|
|39|1.7861628910e-8|1.0088789229|1.0139762581|
|42|2.3224433217e-8|.9811457271|1.0058288623|
|48|2.2439381374e-8|.9465989321|.9968337969|

Pooled memory raw2.0567774985e-8, bank2.1201656114e-8,
local2.1240007568e-8. Memory/bank=.9701022823;
memory/local=.9683506430;3/4 wins. Required<=.75,<=.85,>=3 wins.
FAIL. This3% raw gain does not establish an adjusted gain and is too small
for the extra full-matrix recurrence. No200-net fit, optimizer/feature/
source/window/scale sweep, state write, native port or bank follows.
This is a result for the fixed fit, not an exact linear-representation
ceiling or an impossibility claim about every matrix observer.

Receipt: matrix_error_observer_training_verdict_v1.json.
Reference SHA19d798bb285bc871805396cef65f23c89d28c696e88d6ac0164ae8d185ebd852.
Torch SHA0f50cfa14baab23bd407e9091a37cc6e948c0e8efcd661f02dce377f83e4da7d.
All arrays, models, prerequisites and failed verdict retained.
