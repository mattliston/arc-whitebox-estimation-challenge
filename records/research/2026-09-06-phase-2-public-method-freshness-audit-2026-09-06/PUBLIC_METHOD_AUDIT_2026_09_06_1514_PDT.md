# Phase 2 public-method freshness audit — 2026-09-06 15:14 PDT

Read-only audit by `fresh_efficiency`, checked approximately 22:09–22:14 UTC
(15:09–15:14 PDT). No forum posts, messages, submissions, logins, numerical
experiments, target-array reads, tuning, or changes to existing sources.

## Verdict

No new public method or code release explaining the leaders' approximately
0.15 utilization / 2e-8 raw score was found in the bounded checks below.
This is a qualified negative search result, not proof that no release exists.
In particular, the leaderboard does not establish which algorithm its teams
use. Do not resurrect the old tracker speculation that a particular leader
must use ordinary or rank-reduced K=3 propagation.

Comparison anchors, both read before searching:

- `WGMPROP_READ_ONLY_AUDIT_V1.md`, filesystem mtime 2026-09-06 02:39:30 PDT.
- `PUBLIC_BOARD_READONLY_2026_09_06_1132_PDT.md`, mtime 11:44:36 PDT.
- `../public_g19_math_acceleration_v1/LANE_g3_tracker.md`, T-128–130 and T-128d:
  organizer code already evaluated; ordinary K=3 factorized f64 at rank35
  reached 3.447e-8 raw but about 1.586 budgets by its own counter. AUGMENT
  and K=4 attempts have preserved resource failures. No repeat run justified.

## Actual public delta: score, not method

The returned [Phase 2 leaderboard](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/leaderboards)
shows the following rounded scores. Utilization is the displayed per-MLP
mean; multiplying separately averaged raw and utilization is not an exact
reconstruction of the adjusted mean.

| Team | Adjusted | Raw | Utilization | Displayed failures |
| --- | ---: | ---: | ---: | ---: |
| Puffi | 3.0e-9 | 2.02e-8 | 0.1478860181 | 0 |
| suliman_tadros | 3.0e-9 | 1.97e-8 | 0.1504215170 | 0 |
| marius_binner | 3.2e-9 | 1.84e-8 | 0.1729243469 | 0 |
| J2W | 3.3e-9 | 2.01e-8 | 0.1664958591 | 0 |
| mliston | 5.0e-9 | 2.24e-8 | 0.2220768611 | 0 |

The earlier 11:32 PDT note showed marius_binner at 3.4e-9 and 0.1871960857
utilization. The present row lists a September 6, 21:18 submission time
(page display; timezone not established here). This is evidence of an
improved board entry, not evidence of how the savings were obtained.

## Recent forum and linked code

The [latest-topic index](https://discourse.aicrowd.com/latest) still lists the
September 5 writeup as the newest challenge topic. Targeted searches for
Puffi/Phase 2/GitHub, marius_binner/white-box/method, and
suliman_tadros/estimation/method did not locate a new leader release.

The [September 5 venture_agent_ai writeup](https://discourse.aicrowd.com/t/phase-2-write-up-exact-bivariate-relu-covariance-plus-per-layer-calibration-3x-over-the-kit-baseline-with-negative-results-submission-329951/18214)
is already in the earlier audit. Its [public method document](https://github.com/agentaiventure-dot/arc-whest-estimator/blob/main/docs/ALGORITHMIC_CONTRIBUTION.md)
describes Gaussian bivariate ReLU covariance via fixed quadrature and a
per-layer ridge correction to recorded outputs, not propagated state.
Its author reports adjusted 1.32e-7 for submission 329951; the exact
covariance correction alone gives only 7.4% raw improvement over that
author's Gaussian baseline. It does not explain the leaders' accuracy.
Our Gaussian6-to-12 receipt already makes a new exact-Gaussian-map arm
unmotivated; this audit creates no exception to that closure.

The [organizer-talk thread](https://discourse.aicrowd.com/t/interesting-talks-by-the-organizers/18206)
was posted September 1 and last replied to September 4. Its visible replies
contain no new algorithm or code. The linked talks were already recorded in
the prior audit; they were not reinterpreted as a new implementation release.

The [Puffi archived writeup thread](https://discourse.aicrowd.com/t/phase-1-write-ups-team-puffi/18175)
contains its August 16 PDFs and replies through August 17, with no new Phase 2
method attachment visible. This check was only for freshness; no Phase 1
estimator development or numerical work was undertaken.

## Organizer code ownership and two important qualifications

The [paper record](https://arxiv.org/abs/2605.05179) still shows latest revision
v2 dated May 14, 2026. The [public repository history](https://github.com/alignment-research-center/mlp_cumulant_propagation/commits/main/)
shows latest commit `93d091a4c26c042bfffa28f2e76a81bc0aba94bb`, August 4.
Some GitHub responses were cached; no claim of exhaustive real-time branch
or release coverage is made.

Read-only `git log` and source inspection of our existing organizer checkout
confirm it is already at **that exact HEAD**, author date
`2026-08-04T18:38:34Z`, and contains both `metric_needed` and
`factored_keeps_term`. Checkout:

`/private/tmp/claude-501/-Users-matt-arc-white-box-estimation/8d3a8d39-721c-4038-9498-e971e578bcaa/scratchpad/mlp_cumulant_propagation`

Two explicit upstream details are worth retaining, but are not new code for us:

- [August 4 metric-bookkeeping patch](https://github.com/alignment-research-center/mlp_cumulant_propagation/commit/93d091a4c26c042bfffa28f2e76a81bc0aba94bb):
  skips metric transport only when no radial pieces are tracked, e.g. SIMPLE
  with even K. It does not eliminate a necessary K=3 radial carrier. The
  author reports identical outputs for the affected modes; no new local
  verification was done here.
- [August 1 factored-AUGMENT patch](https://github.com/alignment-research-center/mlp_cumulant_propagation/commit/56f736a7e74c60d9714fa06bd5cb6bf48217a6d8):
  explicitly excludes non-hypertree top-slice diagrams and certain
  top-cumulant/all-distinct cross-products. The excluded terms have the same
  asymptotic squared order as the retained augmentation. Thus factored and
  unfactored AUGMENT are not equivalent accuracy ceilings. This warning
  already exists in our checkout and does not reopen its resource failure.

## Decision

No new numerical arm or transplant follows from this audit. The public
record confirms the competitive cost/accuracy target and contains a useful
ownership check, but does not reveal the leaders' lower-cost implementation.
Continue genuinely new source-derived research under separately frozen
tests; do not spend compute on the already-covered Gaussian quadrature,
ordinary organizer propagation, or old Puffi transfers merely because they
reappear in search results.
