Cholesky-QR source and cost check; no numerical or model execution

This is an existing closed family. The relevant failure is the Gram of the ordinary unpowered Nystrom sketch, not only the failed power-step or whitening variants.

Evidence:

- `lead/runs/opus_fix336687_v1/jsk/JOINSKETCH_v1.md`, 22:4x section: the unpowered sketch Y1 has smallest/largest singular-value ratios approximately 3e-5, 3e-6, and 1e-6 at L7/L10/L13. It explicitly records that Cholesky-QR with a float32 Gram fails because that Gram is not positive definite. Forming Y1.T @ Y1 squares those condition numbers. A second Cholesky-QR pass cannot repair a first Cholesky that already fails.
- `lead/runs/opus_hunt2_v1/RESULT.md`: the earlier bill audit priced three ordinary QR calls at 1.32 GF (0.51% of its S2tr bill) and estimated a Cholesky-QR replacement would save only about 0.5 GF, or 0.2% gross whole bill, with float32 conditioning risk. That was a historical cost estimate, not a validated current-source saving.
- `lead/runs/astra_crunch_compute_v1/nt_reuse_v1/RESULT_v1.md`: explicitly records that block QR and CholQR failed their separate numerical gates.

Cost interpretation: public dense QR is billed as 4*m*r^2 - (4/3)*r^3, i.e. 490,733,568 FLOPs at m=960,r=384. Using the public dense Gram, Cholesky, and general solve formulas documented in the JSK note gives about 622.7 million FLOPs before copies/casts, already dearer. Triangular-system structure does not automatically turn the general public solve into a triangular tariff. The historical optimistic 0.2% assumes cheaper products/solve implementation and excludes conditioning repair. A second orthonormalization adds another Gram/factor/solve sequence; it does not preserve the original optimistic price.

Decision: do not reopen from a same-span argument alone. No actual current metadata rerun was necessary because the precise numerical family is already closed. No candidate, tiny numerical test, model, or SDK run was performed for this check. A shifted or float64 version would need a distinct cost-and-accuracy argument before any experiment; the existing cost ceiling does not justify one.
