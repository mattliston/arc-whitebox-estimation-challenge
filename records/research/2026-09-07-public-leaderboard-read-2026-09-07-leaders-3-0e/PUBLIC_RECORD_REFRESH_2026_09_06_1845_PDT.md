# Phase2 public-record refresh, September6,2026 approximately18:45 PDT

Read-only browser-tool access. No authentication, forum post, comment,
email, upload or external mutation. No Phase1 estimator work. Earlier
method audit: `PUBLIC_METHOD_AUDIT_2026_09_06_1514_PDT.md`.

The returned [official leaderboard](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/leaderboards)
still shows Puffi and suliman_tadros rounded at3.0e-9, raw2.02e-8/1.97e-8,
utilization0.1478860181/0.1504215170, zero displayed failures. mliston is
rank7, rounded5.0e-9, raw2.24e-8, utilization0.2220768611, zero displayed
failures. This corroborates the unchanged bank, not a new improvement.
Separately averaged raw and utilization do not exactly reconstruct the
adjusted mean, and these rows do not reveal the leaders' algorithms.

The [challenge topic index](https://discourse.aicrowd.com/c/white-box-estimation-challenge-2026/2991)
now exposes a [September6 post by yujigon](https://discourse.aicrowd.com/t/phase-2-write-up-zero-fitted-constants-measured-what-the-64-calibration-numbers-in-a-relu-mlp-mean-propagator-are-actually-absorbing-submission-330001/18216)
about submission330001. The author reports1.066e-7 adjusted and a
rank4608 inherited-third-cumulant channel without fitted constants, at
9.89% compute. Its diagnostic distinguishes propagated drift from
teacher-forced one-step skew. These are the author's measurements, not
independently reproduced here. The reported performance does not improve
our bank; CP inheritance and propagated-vs-local-error distinction are
already represented in our research. No new numerical arm follows. The
linked PDF was not opened; only the public post was read. The prior audit
did not expose this topic, but its displayed posting time does not establish
that it was created after that audit; cached/index freshness can differ.

The September5 exact-Gaussian-covariance post and organizer-talk thread
still provide no new leader implementation in this bounded check. The
Gaussian6-to12 closure already prevents treating an exact-Gaussian map as
a new raw-error experiment merely because that post reappears.

The [official Phase2 announcement](https://discourse.aicrowd.com/t/phase-2-of-the-arc-white-box-estimation-challenge-is-live/18197)
confirms1024x16,2^41FLOPs,120s wall cap,0.4s residual cap,8GB process
limit and flopscope-only inference. It specifies October17,2026 at23:59UTC
as submission close. The challenge landing page's older September19
schedule is stale relative to this specific Phase2 announcement. No
environment upgrade or submission was performed on the basis of browsing.
