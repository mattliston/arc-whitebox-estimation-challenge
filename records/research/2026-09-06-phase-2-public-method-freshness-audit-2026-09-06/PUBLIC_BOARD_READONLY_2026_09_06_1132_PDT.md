# Read-only public refresh: September 6, 2026, 11:32 PDT

Scope Phase 2 only. Read public pages; no discussion post, message, upload,
submission or other external write. Values below are the board's displayed
rounded columns, not reconstructed higher-precision scores.

Source: https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/leaderboards

| Entry | Displayed adjusted | Displayed raw | Utilization | Failed column |
| --- | ---: | ---: | ---: | ---: |
| Puffi, rank1 | 3.0e-9 | 2.02e-8 | .1478860181 | 0 |
| suliman_tadros, rank1 | 3.0e-9 | 1.97e-8 | .1504215170 | 0 |
| J2W, rank3 | 3.3e-9 | 2.01e-8 | .1664958591 | 0 |
| marius_binner, rank4 | 3.4e-9 | 1.84e-8 | .1871960857 | 0 |
| mliston, rank7 | 5.0e-9 | 2.24e-8 | .2220768611 | 0 |

Separately, the earlier10:43PDT safe API poll of our329926 returned adjusted
4.982231460183304e-9 and raw2.2448352261505987e-8. The public leaderboard's
zero-failure column is now observed; it is not a newly obtained per-MLP
100/100 detailed grading transcript. Our candidate/package are unchanged.

Inference: the two leaders' utilization is roughly one-third below ours,
while their raw error is only about10-12% lower. There remains a concrete
cost gap as well as an accuracy gap. This says nothing about their private
implementation; no leading Phase2 algorithm was identified from this page.

The new September5 public write-up18214 describes exact bivariate Gaussian
ReLU integration and per-layer recorded-output calibration, at1.32e-7
adjusted (author-reported), not the3e-9 leading method:
https://discourse.aicrowd.com/t/phase-2-write-up-exact-bivariate-relu-covariance-plus-per-layer-calibration-3x-over-the-kit-baseline-with-negative-results-submission-329951/18214

Its limitation is consistent with our already executed correlation-tail
test (Gaussian6->12 changes final means only~4e-8 RMS, less than0.1% raw).
Do not reopen that local integration-order family from the new write-up.

Also read the public organizer-talk thread18206 and primary Gaussian-
mixture literature, including arXiv2412.00343v1. No new cheap, qualified
Phase2 estimator follows just from generic split/merge convergence claims.
