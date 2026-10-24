# Complete prefix/final coadaptation infrastructure: PASS

September 6, 2026, 11:20 PDT. Phase 2 only; bank 329926 unchanged.
No training targets or accuracy claims in these certificates.

All 78,097 trainable coefficients in the existing prefix messages, encoder,
mean heads and gains, plus the actual final Q, were differentiated through
the complete dynamic CP/star/trace carrier. Original CPU f64 versus repaired
native MPS f32 pooled parameter-gradient relative error was 8.3697402e-6 on
rank 35 and 1.0061080e-5 on rank 100. The worst non-negligible group was
1.9422288e-5; all unchanged gates passed. MPS backward took 3.159 / 2.447 s.
This establishes point-VJP agreement, not global smoothness.

An independent target-free finite test perturbed ALL these groups together
along one fixed direction, at relative group-RMS amplitudes +/-0.05 and
+/-0.025. Actual float32 parameter bytes were frozen by the NumPy stage and
reused by MPS. The original NumPy zero point was byte-exact. All finite
trajectories agreed within 1.495e-6 maximum mean error. CPU/MPS central
final-response discrepancies were 1.791% / 3.439% on rank 35 and
1.475% / 2.954% on rank 100, passing the fixed 5% implementation gate.
Old finite-difference-versus-local-derivative failures remain failures.

Receipts: joint_prefix_final_gradients_v1.json, and
joint_prefix_final_finite_numpy_v1.json / joint_prefix_final_finite_mps_v1.json.
Frozen finite payload SHA256:
2914b59543398dc7cc23e0559dee903d6b2c9ba515bd8abb084a6a9f2adb47c1.

The tests authorize a separately fixed bounded coadaptation experiment, not
an optimizer sweep, training success, submission, or a new bank score.
