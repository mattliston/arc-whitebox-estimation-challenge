# Aligned CPU35 forward: fidelity negative

Recorded 2026-09-13 UTC, from the completed and reaped `cpu_aligned_forward_v1`
and `cpu_aligned_forward_watch_v1` run. This is an offline, target-free fidelity
test, not raw-error or adjusted-score evidence.

The prospective maximum absolute error limit remains 1e-5 over all 16 served
layers versus the exact current-c2 native rank35 reference. Aligning the pair
precision, covariance association and cubic-star diagonal precision produced
maximum error **3.358721733093262e-5**, final-layer RMS discrepancy
**6.901781156176013e-6**. The earlier unaligned forward was respectively
3.2573938369750977e-5 and 6.862793798152959e-6. Both fail. No fitting, tolerance
change, native-gradient claim or score claim follows.

Actual model time 6.637899999506772 seconds; independently observed whole tree
13.997149458155036 seconds, peak 4700782592 bytes, zero added swap, maximum
completed resource-observation interval 0.22834708262234926 seconds. Numerical
owners and worker group retired. P/A/cstar bytes unchanged. Session 73830 reaped.

Worker result/completion/finalization SHA256:

- daf67e9ba5155b56e436ad0bef6ea2a64fc43e8e9a63178045fc1b46172cf067
- 061e8d35c8eeb33b7ec966d3718e71f4d7e3b48ee34407295c987bf8c8ec0565
- 2cb18ef7c6046595563ef53421b33e9bb4eb5200f5aea41a206d21f1d461a84a

Independent watcher result/completion/finalization:

- 23f61de442cf8cb9f131096ec0cae5cc948f8f9563cf20e7e301053142e85be7
- 89c2f732918676c3d89decc6237006cc89efbcb3a6e3573d2068a525180a50c8
- b317805a0b1ba2889e689cca6173dc03041caf6c4996776b8f4b717054cf2404

The next proposed diagnostic is a fixed saved-entry8 old-state comparison, not
another full forward or a new trace-correction efficacy experiment. It compares
the native reader, ordinary Torch old-state propagation and a Torch cut given
the exact native rangefinder input. Existing negative receipts remain frozen.
