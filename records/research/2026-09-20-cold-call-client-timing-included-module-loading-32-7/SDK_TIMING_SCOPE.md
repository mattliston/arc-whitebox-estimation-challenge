# Qualification timing scope and official inference

The installed SDK was inspected after submission331682 was accepted.
`whestbench/subprocess_worker.py:176` loads the estimator during initialization,
then calls setup at197. Its `_handle_predict` creates the BudgetContext at66
and calls predict at73. In that path, module import/instantiation is outside
prediction residual accounting. The scoring caller likewise times predict.
These are local installed sources, not a verified copy of the deployed grader.

Our conservative ordinary-client qualification included module execution in
the first prediction BudgetContext. The measured32.7% mean cold residual gain
therefore cannot be credited as an equivalent official residual improvement.
It partly measures faster module parsing/loading. Warm absolute residuals,
byte identity, complete bills, total-wall comparisons and the package checks
remain actual observations. The successful local gate is not being erased or
relabelled; its attribution is narrower than an official timeout repair.

The already accepted upload is an official trial with an unconfirmed outcome.
The prospective prediction-only probe preserves this distinction and compares
unchanged parent/packed sources with SDK-style loading. No accounting, dispatch,
GC or arithmetic changes are made. Do not claim the two official failures have
been fixed until official completion is established; report any absent or
negative prediction-only gain explicitly.
