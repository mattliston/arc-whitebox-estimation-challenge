# Tiny V1: accounting assertion failure, no algebra qualification

Actual run90867, September13 UTC. Frozen worker434bc0f3 and all original
protocols/results remain unchanged. No second numerical attempt has run.

The sole RPC context stopped at `Exact actual section tariff: L8_lazy_join`.
The preceding eager-join tariff assertion passed; this is not a qualified
lazy-join bill, full algebra result, raw result or runtime improvement.
Worker elapsed0.4723086255s, host peak215236608B. Watch elapsed1.7475220831s,
combined peak329023488B, maximum completed observation interval0.2851002496s,
no added swap. All numerical and observational process groups retired.

Both worker and watch are FAILED. Zero completed-call receipts were published.
The failing harness asserted before appending the failed section's actual
counter delta and did not archive the closed failed context's summary. Therefore
the precise paid F is not preserved and cannot honestly be reported as measured.
The hard paid context cap was10BF; no budget overrun was reported. A source-price
reconstruction, if made, must remain distinct from a recovered actual bill.
Future failed-context accounting must retain the completed summary where the
backend supplies one, without treating it as a positive call.

Decision: source-only exact tariff/API diagnosis. No unchanged retry, numerical
or scientific gate relaxation, full-model expansion, fit or upload. A concrete
harness defect may justify a separately versioned correction and fresh root
review; this negative receipt must never be overwritten or presented as PASS.

Actual result/completion/finalization SHA256:

Worker:
11cf52d993c2ba419f2d66469693408f6f0ebccde32a1a0e5508f21f856032f7
0dca96050e2fa8a798adfa2b9a693073940cd1cc2b2a357d2e1a06596bc6e20a
552731115dc3b58b8c6cb6fe8782c70397e7dcd6b5ca4990317d13ca7209eb92

Watch:
6676b2cfe18c766b5bcd2ab78ae292ccde05ad8b52a6188a9180fe2b60095abd
8cf0f70a3cd35743f3cdeb384b3fb67bd853cba071de7ce280a1128702324e03
adbe7d08cf8474cca5490cc3ec30c63f5a821f8eefec81e89243761994bb681a

Official bank remains#330390, adjusted4.53886349950395e-9. The source-only
395.728GF conditional model price is not converted into a measured score.
