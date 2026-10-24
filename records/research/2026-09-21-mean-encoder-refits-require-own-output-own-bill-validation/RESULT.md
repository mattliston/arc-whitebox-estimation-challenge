# Release harness applicability to the current all-P fit

This is a source-only review. No estimator/client/backend ran, no target was
opened, and no candidate, running fit, or sealed protocol changed.

The ordinary NumPy-free plain_driver.py and plain_client.py in the pending
readout composition are exact copies of the previously qualified runtime
harness. Their source and existing qualification are bound in audit.json.
They retain the measured prediction boundary and complete participant cost.

Two existing assertions belong to the readout-only model: first15 bank /
candidate output equality and equal parent / candidate FLOP counts. The
current all-P fit can change both. A future release qualifier must require
each candidate call's own all16 native output bytes and own full bill,
layer0 identity with the bank, and exact repeats under the same call history.
It must not inherit those two readout-specific assertions.

If the old13-call client design is reused (five cold pairs35/100/37/39/42,
then candidate35/100/35), four additional native reference calls are needed:
candidate35/100/35 in one fresh process, and candidate cold100 separately.
The sealed independent14 sequence starts35/100 but repeats35 after twelve
other networks. That is not the same short repeat history; a warm bill is
also not a cold100 bill. Do not substitute either without measurement.

The old local timing margins (pooled ratio<=1.075, worst<=1.10, residual
<=.25s) are properties of that preserved release study. This review does
not declare a new release study or change any scientific gate. Scientific
success, actual-client runtime/residual/memory, contract and exact package
verification remain required. No release work or upload is launched here.
