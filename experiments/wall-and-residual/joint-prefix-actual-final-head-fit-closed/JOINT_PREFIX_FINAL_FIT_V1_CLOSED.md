# Joint prefix and actual final-head fit: closed

Written 2026-09-06 12:16 PDT. Phase 2 only.

## Terminal independent result

The fixed terminal payload was judged on all 40 declared public-development
validation networks, using existing independent 1e9-sample post-ReLU means.
These are not the 14-net grader panel, a pristine holdout, a metered bill, or an
official board score.

- Original supplied P/Q NumPy raw final-layer MSE: 2.247525992904174e-8.
- Jointly fitted P/Q NumPy raw final-layer MSE: 2.244005465955291e-8.
- Pooled ratio: 0.9984335990061971 (0.1566401% improvement).
- Wins: 22/40.
- NumPy/MPS fidelity gate: passed on all 40 networks.
- Declared improvement gates: pooled ratio <=0.90 and at least 24/40 wins.
- Verdict: FAIL. No candidate port, panel run, submission, earlier-checkpoint
  selection, amplitude sweep, optimizer sweep, or epoch extension.

The experiment jointly trained the existing prefix messages, encoder, heads,
and gains together with the actual separate final Q, through the dynamic CP,
star, and trace state. It was not a frozen-Q/readout-only retry. Its negative
result closes this fixed fit, not every possible learned state architecture.

## Execution and provenance

Training completed all 240 updates, six epochs, and all 40 target-free terminal
validation predictions. After writing these and the completed receipt, its
final logger raised a duplicate `wall_seconds` keyword TypeError. The process
therefore exited nonzero. Both the scientific artifacts and the failure receipt
are retained unchanged. No retraining or checkpoint substitution occurred.

Additive integrity and independent provenance audits passed: all 200 weight
hashes matched the authenticated manifest; all 24 transitive local code
dependencies predated training and remained unchanged; terminal payload arrays
equal epoch-six arrays; all validation predictions were complete and finite.
78,097 coefficients were allocated, of which 2,052 were structurally inactive.

Terminal payload SHA256:
`57bfc5bd91eaea3fc07e7f6ee9cf3ecc6fc4fb6c500f6643e3d48cc5758a4621`.

Authoritative receipts:

- `joint_prefix_final_fit_v1/receipt.json` and `failure.json`
- `joint_prefix_final_terminal_integrity_v1.json`
- `joint_prefix_final_provenance_v1/receipt.json`
- `joint_prefix_final_numpy_baselines_v1/receipt.json`
- `joint_prefix_final_judge_cached_v2/receipt.json`

Independent judge completed at 2026-09-06 19:13:29.076114 UTC, exit 0, in
252.600 seconds with peak process RSS 6,413,893,632 bytes. All workers released.
Official bank remains #329926, adjusted 4.982231460183304e-9.
