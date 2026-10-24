# Unused final head is outside the installed prediction clock

The exact official331760 source contains an unused older final-head payload
and helpers: final_input,final_delta,_FINAL_LITERAL,_FINAL_PARAMS and
_predict_compact_final. These five top-level nodes occupy
165207 of599557 source bytes
(27.555%). The active final
Estimator calls _predict_learned_compact. All references to the five removed
names are internal to those nodes; there is no candidate namespace reflection.
The assignments contain only literal values and the removed definitions have
no decorators, defaults or annotations with execution side effects.

An in-memory deletion parses and leaves the retained source AST exact. All
remaining compiled function/class/comprehension code objects also match after
excluding source-location metadata. This is a source/bytecode observation,
not a measured native/client equivalence or runtime benefit. No transformed
candidate was written, imported or executed. No archive or active file changed.

The installed SDK loads and constructs the estimator in its separate start
command. Its predict command calls _handle_predict, whose BudgetContext wraps
predict and output materialization, not module import. The loader's
exec_module therefore runs before the prediction clock. Removing an unused
module payload does not remove an executed prediction statement or charged
array operation. Unmeasured indirect memory/cache effects are not a runtime
repair claim. This lead is not activated as a prediction-budget optimization.

The preserved current-P ordinary release probe includes cold module loading
inside its prediction context; it is a broader cold-call check. The existing
predict_only_probe_v1 places loading outside, matching this SDK distinction.
A parsing-only speedup in the broader probe would not establish a reduction
in SDK prediction residual. Any future runtime claim must use the intended
clock and independently preserve output/bill integrity. This audit does not
change or relax either existing probe or any current acceptance threshold.

The prior CDF-reuse study is a separate executed negative: prediction-only
residual ratio0.993664585825508 and3/5wins failed its fixed<=.98 and>=4 gates.
It remains closed. Neither a source-size reduction nor this audit reopens it.

Evidence is the pinned local installed SDK and saved source; no remote
implementation or runtime improvement is inferred. No estimator calls,
numeric imports, new target access, fitting or numerical experiment occurred.
The signed-message fit and both active bridges are unchanged. Their complete
native accuracy outcome remains the next decision.
