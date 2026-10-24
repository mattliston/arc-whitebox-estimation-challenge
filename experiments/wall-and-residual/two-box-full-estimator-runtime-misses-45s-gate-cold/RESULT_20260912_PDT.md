# Two-box full-card runtime result — 2026-09-12 PDT

Actual run completed at 20:31 PDT (September 13 UTC); file modification times
and the live completion were checked. Official #330390 remains 4.53886349950395e-9.
This is NOT an eligible bank or a repaired official timeout.

Candidate 378092f76102b83ddcff445adaf27309ae8136d1df6844599dad59b69cd75dfa
retains c2's numerical model and changes only two D6 PUSH execution schedules.
One new OpenBLAS RPC process, cold rank37 and own repeat37, completed:

| Context | Total s | Backend s | Metering overhead s | Residual s | Complete FLOPs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cold37 | 51.229001750 | 39.199473715 | 11.807964589 | 0.221563446 | 407741701711 |
| Repeat37 | 50.602393416 | 38.701598909 | 11.679800114 | 0.220994393 | 404454713141 |

All16 FP32 output bytes exactly equal c2 and each other; all16 maximum absolute
drifts are zero. Full initialization is 3286988570F and both bills match the
prospective own-cost calculation exactly. This adds 1342018272F of cold arena
bindings to c2; there is no replacement-binding credit.

The saved, failed c2 cold37 control took62.916841667s. New/saved ratio is
0.8142335246441607, an18.58% historical runtime reduction, NOT a fresh paired
timing experiment. That old control remains failed. The new fixed<=45s AND
ratio<=.80 advancement gate FAILED. No gate is relaxed retrospectively.

Preserve the exact-byte scheduling improvement as a possible component.
Do not upload or claim full native/RPC/domain/selector or official runtime
qualification from these two contexts. No truth was read or adjusted score
measured in this test.

Actual watch:104.856734750s elapsed;6339084288B maximum combined RSS;
0.407031375s maximum completed sample interval;0 added swap. Both calls and
all owned/observational groups retired, no errors or cleanup failures.

Actual result/completion/finalization SHA256:

- 9db5c5f9fd0afbe1b8bd00fc4b8bb5e146501996d8bd008b129be95944d9bcfa
- 880f49684a5e71031387c596c5a847e6b322e4d486c7cb39fbe1e94684a2c88c
- 2827bff6c5ab20dce11501172b86b5bc4d7dae18ccbc835064fdf98718d9111d
