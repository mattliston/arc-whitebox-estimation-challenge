# Current-COMP cstar qualification: failed, no information run

2026-09-10 UTC. Phase 2 only. Additive report; every original source and
failed receipt is preserved. External session60501 exited1 and its owned
worker30748 was retired. No fit, target-reading information stage, candidate
or submission followed.

The fixed original CPU/MPS scalar finite-step gate failed on rank35,
h=.025. CPU secant -1.2771553660195778e-5 versus MPS
-8.940696716308594e-6 gives relative disagreement .2999523038318002,
exceeding the prospective .05 limit. This is NOT a qualified gradient.
All17 CPU calls and five MPS calls had accepted retirement markers; the
next MPS minus025 forward produced its artifacts but failed the comparison
before its marker. Thus23 actual attempts,22 accepted markers. The failed
record's counters have not been edited or retrospectively promoted.

Worker273.9591413746s; external274.7831861246s, maximum combined host
RSS9038643200bytes, sampled MPS driver4331405312bytes, maximum completed
sample interval.3304269593s,1083 samples and zero added system swap.
Resources were not the cause. The worker and external result remain FALSE.

## Small-JSON diagnosis, not an alternative numerical acceptance test

Rank35 scalar values (CPU / MPS):

- Null: .13903103101620415 / .13903070986270905.
- Plus025: .13903071219985974 / .13903070986270905.
- Minus025: .13903135077754275 / .13903115689754486.

The local FP32 ULP is1.4901161193847656e-8. CPU/MPS secant numerators
correspond to42.8542/30 ULP, a12.8542-ULP difference; the5% bound permits
only2.1427 ULP. Final scalar rounding alone is insufficient: rounding the
same exact two endpoints once could change the numerator by at most1 ULP.
Accumulated products/reduction/scaling or physical-trajectory differences
remain possible causes; existing scalars do not identify which. Saved
served-vector backend discrepancies reached1.1920928955078125e-6.
No array was decoded or rescored for this diagnosis.

The CPU vector FAD/RAD internal checks that had completed are partial
observations, not a replacement PASS. Their agreement does not establish
native finite-response derivatives. The drafted information.py and its
protocol are parked: their required passed qualification is absent.

## Immutable failed seals

Worker result d2c01e7013633c23b0fe97ce0a2444065232527ab8730951ee7eb432df36a77b
completion141253a47adab3bd7b87798241d49468b379305c959334f27fcfd12a6815651c
finalization566e3ebf96ddd9fe90ec01379fbf98998d1eb38762ab337bd73f6be9261dfb1b.

External result adb4a133dc06d21164d766a4e9e9a5a93867a7a6bd1456e0b8e833e9591fd555.
Completion888e107fc153b696926a060351854f7c7e2b06975a8ba743b6d0aef5f8472ec9;
finalizationa7401eb8c2f326b766079069cccc00455e4140a9bcab83292fb16222f9c3dd87.

Neither a remaining-gain claim nor a proof of stationarity follows. The
official bank remains #330390, adjusted4.53886349950395e-9 (archived).
