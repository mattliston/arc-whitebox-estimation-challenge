# Historical capacity and gate evidence

This is a source/receipt audit, not a new estimator experiment. Existing
result JSONs and the small shape descriptors in saved PyTorch archives were
read with a restricted standard-library metadata decoder. No PyTorch global,
tensor storage, truth array, candidate inference, or training was executed.

The current bank already has three learned W/W²/W³ message groups and three
W²/W⁴/W⁶ energy groups, with state64, message16, hidden128 and30 features.
Message width is derived from the payload shape. A width change is therefore
not ruled out by a hardcoded16, but dimensions alone establish no accuracy,
native compatibility, cost, or resource benefit.

Actual wider fits exist. The old51-feature full-host state64/message16/
hidden128 fit has saved-row valid MSE7.35158e-9 after40epochs; state128 with
the same message16/hidden128 has7.57955e-9 after80epochs despite lower train
error. This is not an isolated message-width comparison. On a different
Gaussian host, state128/message48/hidden256 gives9.49439e-6, and state128/
message128/hidden512 gives9.37415e-6, both60epochs. Those fits change hidden
width as well as messages. The Gaussian-PSD message48 fit gives2.47649e-7.
These saved-row results do not measure the current endogenous estimator or
its own complete FLOP-adjusted score. They cannot establish a universal
capacity limit or forecast a current-bank gain. Serialized shapes, rather
than filename tags or today's trainer defaults, establish the dimensions.

The physical activation-probability gate replacement was actually refitted
on the frozen200 binary-host rows for40epochs. The training receipt binds
the exact trainer and its one-line source change. Saved-row held-out MSE
5.93605e-9 was slightly better than the documented6.01057e-9 control, but
the matched closed-loop rank35/100 screen gave2.25119507e-8/2.15027866e-8,
worse than the documented2.21231e-8/2.11385e-8 control. This spelling is an
executed negative, not an untested idea. Those historical screens are not
new metered native scores and do not test retaining additional raw-gate
channels alongside the existing standardized-gate messages.

The nonlinear-memory capacity directory also contains executed fits and
consumer receipts. It tested fixed16→8→8→2 plane decoders on an older teacher,
with six training networks and held-out400/500. Memory/control plane ratios
1.01105/1.01093 and mean-consumer ratios1.03505/1.03475 failed its original
gate. These are local teacher-forced plane/consumer discrepancies, not a
wider GNN test or an endogenous adjusted-score bound.

No new width, gate, feature, seed, learning schedule, candidate, or performance
threshold is selected by this audit. The active fixed fit and native endpoint
remain unchanged; their eventual native result determines their next gate.
