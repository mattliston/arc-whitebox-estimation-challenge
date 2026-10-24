# Existing recurrent-encoder direction: no expansion

September 5, 2026,21:23 PDT. Phase2 only. No candidate or upload.

A single train-only relative-gradient direction used44,144 existing
message/encoder parameters; all mean heads, gains, normalizers, moment
physics and finalQ stayed frozen. Training ranks35/36/38/40; fixed new-fit
holdouts37/39/42/48. The scalar least-squares step .02141471511 came only
from training tangents and did not reach the .10 relative-group cap.

Full-carrier reverse/forward directional derivatives agree to2.05e-7
relative. The independent dynamic NumPy finite-step results are:

| Rank | Original raw | New raw | Ratio |
| --- | --- | --- | --- |
|37|2.12811809538e-8|2.16057574007e-8|1.015252|
|39|1.78616289101e-8|1.78167840369e-8|.997489|
|42|2.32244332166e-8|2.36819632407e-8|1.019700|
|48|2.24393813739e-8|2.09415472550e-8|.933250|

Pooled ratio .9910316850;2/4 wins. This fails .90 and3-wins. Do not
expand this direction to optimization, a different scalar/group/radius
or larger data. This is not a universal limit on all learned encoders.
Torch/NumPy actual trajectories differ by at most2.19e-7. Probe118.83s,
peakRSS9.37GB, one explicitly authorized nice10 CPU worker,3threads;
swap stayed5926.50MiB. CPU finite replay13.84s; NumPy judge24.20s.

Payload SHA2560de209b33a5f457010309ee3ccb3ca3b7f43ed5a1aeefcf6f3fa2a37fd2d616b.
Receipts: encoder_end_to_end_direction_v1.json,
encoder_end_to_end_actual_torch_v1.json,
encoder_end_to_end_actual_verdict_v1.json. No official score change.
