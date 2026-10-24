# Whole-bank VJP agrees; finite dynamic step crosses memberships

September6,2026,07:14PDT. No training or parameter update.

All16x1024 control-gradient VJPs, fixed independent random final projection,
MPSf32 versus original certified CPUf64 BankFullCard:
relative2.65834453e-6 /5.39764156e-6 on35/100. Maximum per-layer relative
error9.701e-6. All finite; both full forward2e-6 and baseline membership
gates pass. CPU reverse11-13s/net; MPS2.4-3.0s/net in this mixed worker.

The composite certificate nevertheless FAILS its dynamic finite-step
membership gate and remains marked false. At layer13, eps2e-4/1e-4:
rank35 minus2e-4 crosses a membership; rank100 plus BOTH amplitudes crosses.
Rank35 eps1e-4 remains in cell, scalarFD vsreverse relative.00016805.
Other discrepancies are.00502066 rank35large, .0500772/.0487778 rank100.
Do not call this global smoothness, change amplitudes/direction or hide
the failed composite flag. The wrong-cell comparison is not evidence of
a wrong local AD implementation; it is the already known discrete-selector
issue, now explicitly observed for this late-direction probe too.

A separate zero-point BRANCH certificate may freeze both memberships
AND ordered indices to the previously recorded MPSv3 baseline, using the
SAME eps2e-4/1e-4 and same control/projection. Require zero-point prediction
byte parity with the dynamic baseline and scalarFD agreement<=5% plus
half-step agreement<=5%. This explicitly certifies only the derivative
autograd computes with detached selectors. No training or new authority
is inferred until this test completes; old global flag stays false.

Time31.898s; max processRSS7,866,630,144B; largest observed GPU driver
allocation6,504,054,784B (do not add these as independent peak totals).
Swap5910.50MiB unchanged before/after. Output SHA
e5848f4bb3a8c674347909a7682f79ff5cb2b5b083d83556f9a66b8cca9b018e.

Retrospective audit so far: original CPUf64 readout and encoder-direction
negative fits are NOT invalidated by this MPS bug. Their derivatives and
independent NumPy finite results were already certified; those particular
fit families stay closed. The old MPS frozen-K3 transport-shadow derivative
may have been affected, but it also approximates K3 dynamics and is not
automatically reopened or declared accurate. No blanket invalidation.
