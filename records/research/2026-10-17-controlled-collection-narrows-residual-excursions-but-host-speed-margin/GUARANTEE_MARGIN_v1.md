# Residual-cap margin: what "guaranteed not to trip" means for our builds (2026-10-17, lead)

Data: every network of the 15 graded Phase-2 runs since #336342 (1,497 network-runs), residual excursion = residual / run median,
with the server kernel ratio of the same network vs #336342 (host speed proxy). Source: platform eval telemetry; script compare_runs_v1.py /
the excursion table in TEAM_LOG 17:2xZ.

| GC design class | network-runs | excursion p99 | p99.9 | max | > +10% | > +15% | > +20% |
|---|---|---|---|---|---|---|---|
| default Python GC (2b108, b709, S2t, S3/S4 stacks, S2tr, VAnoB, v21a, v22b) | 1,097 | 1.179 | 1.259 | 1.365 | 26 | 15 | 4 |
| v25f end-of-predict full collect, no per-predict freeze (#336741) | 100 | 1.144 | 1.144 | 1.144 | 3 | 0 | 0 |
| controlled: per-predict freeze (#336759 v30, #336760 v31t) and GC-quiet (#336754) | 300 | 1.057 | 1.127 | 1.127 | 1 | 0 | 0 |

- With the GC design fixed (freeze after every predict), the residual excursion of a network tracks the host's speed on that network: the three
  worst controlled-class excursions (1.127, 1.059, 1.057 in #336754) sat on networks whose server kernel time ran +12%, +11%, +9%; in #336759 /
  #336760 the worst excursions (1.047 / 1.052) sat on networks at kernel +7% / +6%. No controlled-class excursion exceeded its host slowdown by
  more than ~2 points.
- The worst host episode in the record is b709's worker 4: kernel +22% on its first three networks (residual +36% there because that build also
  carried default-GC scans). The worst controlled-class host event: +12%.
- Therefore the design rule: residual median x (1 + worst host episode) <= 0.400 s. For the worst observed host episode (+22%, +2 points of
  residual on top = +25%): median <= 0.320 s. For the worst controlled-class event observed (+12%): median <= 0.357 s.

| build | median (platform) | covers host slowdown up to | status |
|---|---|---|---|
| #336741 v25f | 0.361 (+growth late) | +11% minus the growth | 2 failures |
| #336759 v30 (freeze + dtype) | 0.349 | +14.6% | 100/100, 1.473e-9 |
| v34y = v30 + coarse young GC | ~0.338 (projected: -3.3% local) | +18% | qualified, fires 10-18 |
| #336760 v31t SAFE (two-strip) | 0.327 | +22% | 100/100, 1.481e-9 |
| v35y = v31t + coarse young GC | ~0.316 (projected) | +26.6% (beyond the worst host episode ever observed) | qualified, fires 10-18 |

Conclusion: the SAFE line (v31t now, v35y next) is guaranteed against every host event in the 1,497-network record; the primary line keeps the
1.473 score and is guaranteed against every CONTROLLED-class event observed (+12%) but not against a b709-class host episode (+22%), which
occurred once in 15 runs. Nominate the pair.
