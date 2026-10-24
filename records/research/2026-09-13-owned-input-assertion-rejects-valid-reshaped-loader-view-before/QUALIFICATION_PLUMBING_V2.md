# Owned-input plumbing correction, no numerical-method change

September13 UTC. Preserve failed qualification_v1 and its source/pins/watch.
It stopped BEFORE a model or backward call at the newly added root condition
weights.flags.owndata. The admitted NumPy2.4.6 loader explicitly executes
array=numpy.fromfile(...) followed by array=array.reshape(shape), producing
a shaped view. The .npy header confirms <f4,C-order,(16,1024,1024). This is
an overstrict root ownership assertion, not a changed-model fidelity result.

Worker failed at6.885062126s, peakself400883712B; zero model/backward/fit
calls. Watch7.075477709s, observed combined353124352B, zero added swap,
maximum observation gap .1903255424s, all processes retired; session77630
reapedexit1. Observed tree peak is sampled, not a replacement for self peak.
Worker r/c/f:31987f9b5e0601d0a37eaf960280b290260be9b0dec0c50aca0bb4a83d3851b2,
cb950efbf161d2a4ff32e8f6a2cbe67d98efac6d6b978f86e07bc9c39c15f867,
481560122632c692faafbec80dedf2157ae608a1ee08989709ca21f8636c4c92.
Watch r/c/f:d51d88de426e0fd11518606a3bcb1e1b38055a75eeab7f480fa6dfd176544abf,
57d09cf90c110503e3dceecf958dee52d091d62ca5d977da0953cd5d768233d5,
383d4b31b92c82d2d09e602ab4ccda29c35e19a01859ff381dc9f2190e19a06b.

Only v2 change: load unchanged canonical bytes, validate exact dtype/shape/
C-layout, make one explicit copy(order='K'), verify every byte matches,
retire the loader view, then retain the original owned/finite assertion.
Track both input owners to final retirement. No dtype conversion or changed
model input. Add100MF source allowance for the16,777,216-element copy and
complete byte exports/checks, taking the explicit three-call upper from
7,344,823,291,904F to7,344,923,291,904F, still<8TF. Extra transient67,108,864B
fits the unchanged8GB cap. Actual deployment FLOPs are not claimed.

The original qualifier4ed15b49ff9e6d0a58c98e8eb64fee2c64f1b19d320b3cadad2870f620a9772d
and original scientific protocol d29c711c05ba7e377aa646dd3f4da8a687cb01f2f72d456f72c2cf0a44994028
remain byte-exact. Same fixed three calls and all output, gradient, inactive,
shared, selector, variance, replay, source, process and resource gates. New
exclusive v2 directories only; no numerical repair or relaxed tolerance.
Still execution-only: no native-gradient, fitting, raw-gain or score claim.
