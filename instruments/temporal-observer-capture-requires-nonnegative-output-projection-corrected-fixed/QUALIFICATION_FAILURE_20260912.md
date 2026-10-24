# Original v29 operational qualification: FAIL

Actual September12,2026 around20:20UTC. Candidate-adapter source cab844cd,
worker551644da, watchf7783cd0 unchanged. One cold37 prediction was attempted
and returned the required FP32[16,1024] array; zero calls were qualified or
retired by the original loop. No repeat was attempted, no output/capture was
accepted, no truth/fit/score/submission occurred.

Failure is the original paid-loop line61:

    assert np.isfinite(values).all() and (values>=0).all()

It occurred after exact official materialization and byte-equality checks,
before saving the output and full trial. This is a genuine output-domain
failure, not resource exhaustion. The existing combined predicate does not
itself record whether nonfiniteness, negative values, or both caused it;
their size/location is not measured. No tolerance is relaxed and no output
is silently clipped. The frozen adapter's feature-host finite scans completed,
but no claim of captured-host/served parity passed the later acquisition gate.

Actual constructor+setup0.02115954179316759s/2261285F/33operations, seed0,
inside cold participant context, original warm-up catch policy unchanged.
Worker25.457867791876197s, peak6863536128B. External watch26.34757800027728s,
combined6874742784B, max completed sample.3351257909089327s, zero added swap.
All numerical/observation groups retired. This is not a complete-cost or
accuracy result; the attempted participant bill was not accepted/saved.

Actual negative worker result/completion/final:
ea17d5505acdcbb4e2bd2f258a28530b2cf988551f4e71e43e019e7f4ddaed49;
933f69c3425c3487ffd84e2561db42682edb7141e858bde6d45889a61e1047b7;
f0ce9cc9d37f0e7a5faac90e96ed2f40dde44d7b1fa85697590a968a1cab00d4.
Actual negative watch result/completion/final:
6073323d59c56bd48f910acd9c640156ef0a81e036de2014e57d7f83167ac781;
a920c31d29a9238e0559ebab6aa8384cd6ff2fa03616975c7804dd3a025cc53e;
429a6c39e3e51495e2684fc328de9268d53e5e5e8174b318e4c9fff500ab1dfd.

The root read the four expanded-capture sources/protocols and model/design,
but those were not executed and cannot admit these negative prerequisites.
TRAIN20, CHECK5, observer qualification and training remain unrun. The
raw-route question is still open scientifically, but this unmodified
acquisition is closed operationally. Any further run needs an explicit
new diagnostic/repair decision and versioned source, not a resume or retry.
The immediate bank priority remains unchanged SHARED/OpenBLAS diagnosis.
