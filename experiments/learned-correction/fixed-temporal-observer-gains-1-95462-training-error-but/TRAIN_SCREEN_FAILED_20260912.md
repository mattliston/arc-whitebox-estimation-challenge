# Fixed projected observer: operational PASS, usefulness screen FAILED

September 12, 2026. Receipt/source review only. No CHECK truth, new array
decoding, model call, fit, deployment build or qualification was performed
for this note.

The sole frozen 30-epoch / 600-backward / 150-update proposal completed all
624 neural rollouts and all execution/resource gates. Its original-precision
TRAIN20 final MSE was 2.165408115243205e-8 versus the actual projected
head-free host's 2.2085774194796532e-8: ratio 0.9804537962510641, a 1.95462%
reduction. This FAILS the prospective ratio <= 0.90 usefulness screen in
`TRAIN_DESIGN.md`. `training_v1/result.json` correctly distinguishes
`passed: true` from `TRAIN_screen_passed: false`, with
`fused_native_authorized: false` and `validation_targets_accessed: false`.

All 20 TRAIN rows improved, but neither that fact nor their fitted aggregate
establishes held-network benefit, native raw-times-own-cost improvement, or
an adequate improvement at v29's higher complete inference price. The saved
unprojected-host mean MSE is 2.2085774194893835e-8; the negligible difference
from the projected host does not explain the failed screen.

The saved changing-iterate epoch mean divided losses decreased from
0.27608334049582484 (epoch 0) to 0.27073064222931864 (epoch 29), with
0.27133634462952616 at epoch 20. These are the same 20-network population
but evaluated along changing updates and orders, not a fixed-endpoint
stationarity certificate. The last small decrease is not authority for a
longer run. The result does not prove that every observer, feature set, or
optimizer is incapable of improving this backbone; it does retire THIS
prospectively fixed proposal without a refit, seed/epoch/capacity/normalizer
change, final head polish, cached deployment qualification, full physical
qualification, CHECK scoring, or submission.

## Actual immutable endpoint

The sole terminal payload is `training_v1/terminal_payload.npz`, SHA256
ae50ca024758054352c05bc2eaf358e6d0d143a12546f4bfe892b252aa8646b6.

Worker result/completion/finalization:

- 98b6d2b83a0640fd4ef239b46e720636ea0fbec4adce95da4467a2f1e74d2a44
- 3475b7c41345bdc58dc99cdbb661898acda4578c71c1a337bbd109bf6481c83d
- 33c03446f1689cd406d551ec24eeb4862776e0f0d6708ca36c836a03f3ccdf15

Watch result/completion/finalization:

- 7ef200dde3e75de940f9e3fd0a036152b7b8832e111dce8bf1beca73f9bf0512
- 92cc4643692af970a8d818a09122f382855715f7e2d06b1f4c29b44daa0479ca
- 5bc61f8697b6b40ffdaf5058a10d7408b7f596c9a867c2c1af2f41eb7fe3b064

## Inert deployment drafts preserved, not completed or authorized

The conditional work stopped when this screen failed. These two already
written drafts are preserved unchanged; neither was executed, preflighted,
fully qualified or used to create a candidate:

- `fused.py`: 2ce2a9a92ab7966a7641626e06a6715ba66227f08e4f9ebdd4cca0febfefd0d8
- `build.py`: 8a8e99ac742ca81647d84ad94631b9feeab9eb01f5c0635a5a7ac0ef94c25ab6

No deployment qualification worker, builder watch, candidate literal, or
build stage was created. These files confer no eligibility and must not be
completed or launched as a rescue of this failed fixed screen. The earlier
capture/qualification failures and their negative seals also remain intact.
