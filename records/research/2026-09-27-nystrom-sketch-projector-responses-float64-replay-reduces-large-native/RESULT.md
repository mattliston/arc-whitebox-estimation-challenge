# Current Nyström origin: arithmetic and inherited sensitivity

Ten passive layer10 traces of the checkpoint49 covariance-only estimator
reproduce all16 prediction bytes and complete bills. The five fixed direction0
cases on MINI57/68 contain the actual joining bank, stored weights, old frame
and core, selected physical neurons, sketch, QR factors and new core.
Native-runtime QR and core replay reproduce captured outputs byte for byte.
No final targets, new intervention, fitted coefficients or score are involved.

The actual selected physical column IDs and order are identical across all
five cases on both networks. The previous-row order changes for the positive
MINI68 cases, but its physical set stays fixed. All comparisons embed rows
in physical-neuron coordinates and compare projectors or represented cores,
not arbitrary QR signs or unrelated core coordinates. The fixed-zero-column
reference equals the actual-column reference here.

| MINI | Native sketch gap | FP64 sketch gap | Native projector gap | FP64 projector gap | FP64 represented-core gap |
|---:|---:|---:|---:|---:|---:|
| 57 | 1.004359 | 0.026123 | 1.118416 | 0.070052 | 0.026432 |
| 68 | 0.994501 | 0.233003 | 1.103045 | 0.498728 | 0.211728 |

Native sketches differ from same-input FP64 references by3.20e-6..3.47e-6
relative. Their condition numbers are roughly66,000–81,000. Native/reference
projector discrepancies are0.588–0.725%, despite those small sketch errors.
Native projector response RMS is1.46e-4/1.79e-4, compared with6.84e-7/4.74e-8
for the FP64 references. Native represented-core response gaps exceed105%.

The FP64 sketch/reference-core response is much smoother on MINI57, while
MINI68 retains23.30% sketch and21.17% represented-core gaps. These references
inherit the captured upstream native inputs, including row-order-dependent
calculations. This isolates a substantial local arithmetic contribution,
but does not prove that changing this merge alone yields a globally stable
response or better estimation accuracy. All inspected old-core and Nyström K
eigenvalues are positive; no spectral repair is justified by these captures.
The old ten-sketch FP64 producer and endpoint failures remain preserved.

All6 guarded groups complete and retire, peak5.098GB,zero added swap.
Capture is approximately250MB; no new reference samples or downloads.
Both live handles are reaped. S31#332517 remains saved adjusted
2.1736465056908115e-9; no fresh official poll or new submission. The2e-9/~1e-9
goal remains active and submissions remain authorized.
