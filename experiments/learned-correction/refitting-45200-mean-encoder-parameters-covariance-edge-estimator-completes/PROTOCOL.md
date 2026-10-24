# Continue the already-running fixed fit into its sealed endpoint

This controller changes orchestration only. The existing training worker in
group19016/controller session66800 remains the sole numerical worker. Do not
restart, modify or duplicate it. The controller claims execution exclusively,
authenticates the existing fit and endpoint source seals, and waits for the
current fit's final guard and completion receipts. It decodes no arrays.

An unsuccessful fit, nonzero added swap, incomplete resource retirement or
missing completion receipt stops the controller without running the endpoint.
After success, reverify the immutable sources, final160-graph/40-update result,
terminal payload pin and actual absence of the old numerical group. Then invoke
the existing run_endpoint.py exactly once. Its original ordered build, null
qualification, parent/candidate native40 acquisition, prediction sealing and
score gates remain unchanged. Its internal guards continue to own each newly
created numerical group. No simultaneous large worker is permitted.

The waiting ceiling is14,430s from this controller's start; the original fit
itself retains its original four-hour ceiling from its own launch. Every
endpoint stage retains the existing individual wall,7.5GB RSS and swap bounds.
No restart or retry is automatic. A native40 scientific failure is preserved
as a completed negative result. Neither result launches downstream panels,
packaging, an upload, messages or any network action.

run_claim.json prevents a duplicate controller. The original controller session
still provides the fit's terminal result. This controller's separate session
provides the terminal pipeline result. Future work must inspect both handles
and receipts before deciding that a process stopped. Do not manually invoke
run_endpoint.py while this controller owns that pending invocation.
