# Unchanged-model allocator qualification passes

The fixed plain35/checkpointed35/checkpointed100/checkpointed35 sequence completed.
All three backward passes produced finite gradients; repeat35 gradient bytes
matched exactly. Rank35 output, domain, selector and state-geometry records equal
the preserved original plain35. All parameters remained unchanged through three
zero-rate Adam steps, all cards retired, and no score label was accessed.

Original plain35 process high-water was 6,793,330,688B;
the new plain35 high-water was 2,965,045,248B.
The complete sequence peaked at 5,706,481,664B, equal to
the largest observed tree-RSS sample, below the new 7.5GB trigger. No added swap
was observed. Wide sequence wall time was 99.509694s. All
three owned groups, including the deliberately small guard test, retired.

The old model_v2 source and numerical operations were imported unchanged. The
only runtime intervention was collection and documented unused-mapping relief
at boundaries. ABI and small live-array/gradient checks passed first. The old
failed 8GB qualification is preserved and remains failed. This result qualifies
this distinct allocation schedule, not the old one.

The raw all-zone allocator counters contain 455 inconsistent
before/after snapshots, including apparent unsigned block-count wrap and live
bytes greater than reserved bytes. ABI layout agrees with the compiled C fixture;
no cause for the counter inconsistency is established. These counters cannot
quantify live ownership. Pressure-relief byte returns also do not measure RSS
change. The resource result uses independently sampled RSS and process high-water
values. Weak tensor records are only a boundary-storage lower bound.

This remains an approximate CPU proposal engine. Native rank35 final RMS output
discrepancy remains 4.0870829656109e-06; no
native derivative, score improvement, fit, candidate or official bank is claimed.
Measured backward-call durations are [29.684643416665494, 30.11640950012952, 29.870359583757818]; a separate
seal is required before the fixed 160-network/40-update training proposal. Its
native score/cost acceptance gates remain unchanged. The roughly1e-9 goal is
unmet and the existing exact-upload approval block is unchanged.
