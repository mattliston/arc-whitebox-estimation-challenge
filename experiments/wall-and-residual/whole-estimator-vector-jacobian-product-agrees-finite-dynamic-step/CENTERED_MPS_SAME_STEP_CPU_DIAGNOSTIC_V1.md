# Same finite steps on CPU: diagnose, do not tune the failed gate

September6,2026. Fixed-selection MPS finite-step certificate retains FALSE:
rank35 passes(.1962%/.0163% FD-vsAD), rank100large fails5.7781%, half step
3.6287%. Both zero-point predictions are byte-identical to the dynamic
MPS baseline. All16x1024 point VJPs already match CPUf64 to<.001% relative.
Thus birth/active selection alone is not all finite-step nonlinearity.

Run the SAME archived layer13 direction, SAME projection, SAME2e-4/1e-4
amplitudes, SAME frozen selected indices/masks through the ORIGINAL CPU
f64 moment BankFullCard. This is a numerical attribution, not another
amplitude search. Compare CPU scalar finite differences with MPS finite
differences and the previously computed CPU point VJP. Require CPU/MPS
finite-difference agreement<=1% on every case as a backend consistency
diagnostic. Preserve original failed FD-vsAD flags regardless of result.
No fits, payload updates, or new target access.
