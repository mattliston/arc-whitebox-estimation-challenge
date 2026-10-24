# Native response origin: covariance arithmetic identified locally

The first changed output is post9 for all four checkpoint48 direction/network
cells. Ten passive full-network traces for direction0 reproduce all16 saved
prediction bytes and complete operation bills. No reference final means are
read. C8, star8, covariance/star transport, CP state, scalar physical mean and
learned mean contribution are captured without changing the trajectory.

The imposed C8 response is linear to below1e-12. The native Cpre9 response
has81.41/85.11% secant inconsistency on MINI57/68; its diagonal has61.77/67.86%.
The star response is much smaller and more stable: full secant gaps4.73/5.38%.
Layer8 scalar/hidden states and layer9 incoming mean are byte-identical across
probes. Retained star columns and all selection sets are identical. Some
live/response index orders change. The physical mean carries the main first
output instability; the learned mean delta is about400 times smaller in RMS.

Captured star factors reproduce the actual star transport exactly in the
same native NumPy/OpenBLAS runtime. Ordinary FP32 QR/SVD replay also reproduces
it exactly. FP64 factors and transport reduce its diagonal secant gaps to
0.0231/0.0317%, but this is a numerical reference, not an estimator change.
Initial sketch condition numbers are20.89/17.90; this first star factor is
not severely ill-conditioned. The post8 star-change source identity passes
the declared FP32 rounding bounds.

Ordinary covariance multiplication, preserving the parent's block symmetry
construction, isolates the larger local arithmetic effect:

| MINI | Native full gap | Classical32 full gap | Classical32 diagonal gap | Classical64 full gap |
|---:|---:|---:|---:|---:|
| 57 | 0.814134 | 0.007004 | 0.022505 | 9.43e-12 |
| 68 | 0.851101 | 0.008129 | 0.024795 | 1.12e-11 |

The FP64 central response agrees with independently transported input
differences to below5e-12. Native central-response errors relative to that
linear reference are52.58/57.52% for the full matrix and34.61/37.59% for its
diagonal. Classical FP32 passes the prospectively fixed local gates on both
networks. This does not establish final-objective differentiability, score
improvement or a global error bound. A separately specified native covariance
intervention subsequently confirms the first-layer benefit but fails at the
final layer; see classical_covariance_response_v1/RESULT.md.

All8 guards complete and retire, zero added swap. No targets, fitted
coefficients or new submission. The checkpoint48 qualification remains failed.
