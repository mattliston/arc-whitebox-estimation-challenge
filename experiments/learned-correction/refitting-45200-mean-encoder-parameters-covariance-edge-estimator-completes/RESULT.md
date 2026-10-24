# Current P already uses squared and cubic recurrent messages

The effective current-bank gnn_step projects the64-dimensional recurrent state
into48 channels, split into three16-channel groups. It transports those groups
through W, W squared and W cubed. It also transports their squared values through
W squared, W to the fourth and W to the sixth, then applies nonnegative log1p
compression. The three signed messages are multiplied by the existing feature g.
All six groups are clipped and concatenated with30 scalar features, giving126
encoder inputs. The encoder has two128-wide layers and a64-dimensional output.
The existing11 P keys contain45,200 allocated parameters.

This directly refutes an absent-squared-message rationale for adding ordinary
squared-weight recurrence. A phrase search of selected Markdown documents had
returned no matches; that absence was not evidence of an absent implementation.
The effective source and its literal shapes supply the authoritative evidence.
The already-running current-P fit imports precisely this source function through
the qualified translation, so it is also already adapting these message routes.

These are raw weight powers, not an explicit orthogonal Hermite basis. The
presence of these routes does not prove that every centering, normalization,
projection-width or representation variant has been tested or is ineffective.
Any such extension needs its own distinct mechanism and evidence. No new model,
fit, score label, estimator call, FLOP measurement or performance claim is part
of this source audit. The current sealed fit and endpoint remain unchanged.
