# Read5 qualified locally; exact archive approval pending

The fixed three-packet D5 read extension passed its arithmetic, development,
existing14-panel, production-client, runtime, contract and package gates.
It has NOT been submitted. The official best remains graded #331584,
adjusted4.015535762022727e-9. No official sub4e-9 milestone is claimed.
The approximately1e-9 goal remains active and unfinished.

## Paired existing-panel result

| Panel | Read5 adjusted NC | Cost75 adjusted NC | Ratio | Wins |
| --- | ---: | ---: | ---: | ---: |
|Public8|4.300302905857623e-9|4.35338633650938e-9|.9878064048|8/8|
|Fresh6|3.3511074291299474e-9|3.3966087377385946e-9|.9866039005|6/6|
|All14|3.893504844402904e-9|3.943338794179043e-9|.9873624985|14/14|

Worst individual ratio .9953509352. Mean complete cold bill407071429421.857F,
versus412654885880.857F. Raw NC means2.0997590667e-8 versus2.0977588407e-8:
the net improvement is from lower cost, with a small raw-error increase.
These are existing panel networks and existing4M targets, not a newly
untouched generalization set. All candidate outputs/cold bills and exact
parent weight joins were sealed before the new panel scoring.

Direct paired ratio to graded Cost150 is .9781167399. Applying it to the
official parent gives3.9276627483502426e-9, an estimate only. Cost75 is
ungraded, so its projection was not treated as an official score.

## Arithmetic and development

The extra level processes the D4 leaf batch axis in fixed lengths2,2,3.
Dead transform regions hold child operands/products; scratch reuses child
A after matmul. The relocated D4 leaf and unchanged backward destinations
pass explicit live-interval checks. Arena growth is1,404,928 bytes, to
1,170,305,024 bytes. This is arena growth, not a bound on total process RSS.

Three fixed geometries match a separate fullD5 circuit byte for byte.
Interleaved/repeated geometry also matches. Maximum direct-float64 relative
RMS error is3.758e-5; measured warm bills exactly match the integer cost
formula. Large-read warm bills are .906466x/.914686x D4, with more runtime.
The tiny geometry is correctly dispatched to ordinary multiplication.

Native35/100/35 plus fresh cold100 gives exact repeat and warm/cold100
predictions. Both own-cold adjusted scores improve: .985923974/.989133999,
pooled .987525188, passing the fixed <=.99 gate. The first output layer
remains exact. No coefficient, rank, age, physics, training or head changes.

## Production client and resources

All16 output layers and all FLOP bills match native exactly; repeat is
exact. The client has no NumPy installed. Timing versus saved Cost75:

| Call | Read5 seconds | Cost75 seconds | Read5 requests | Cost75 requests |
| --- | ---: | ---: | ---: | ---: |
|Cold35|40.748614|37.304136|60567|56288|
|Next100|37.582351|34.261619|51084|47608|
|Repeat35|38.361778|35.067401|45766|43750|

Pooled wall ratio1.094338260 passes the fixed <=1.10 gate; every call is
under45s. More requests and9.43% higher local runtime remain a platform
runtime risk. This is not a guarantee of platform completion.
Full context residuals are .308422/.154549/.141621 seconds. Version1 of
the timing report mislabeled operation subtotals as context totals; the
preserved v2 report corrects only those diagnostics. Candidate, full wall
times, bills, output parity and every gate decision are unchanged.

Maximum monitored process-tree RSS7,071,268,864B, zero additional swap;
all owned groups retired. All numerical jobs were serial, nice10/BLAS1,
under the8GB process-tree and256MiB additional-swap guard. No downloads or
large datasets were written. Installed whest0.14 contract validation passes
at2x4, separately from the full1024x16 production-client check.

## Pinned archive and next action

Archive `submission-read5-20260919.tar.gz`,772231 bytes, contains ONLY
`estimator.py` and `manifest.json`; source equality and validate_package pass.

- Candidate SHA256: d7723068ae269155e0369e4ba322b447222bbb31295d89394c069695ad0e824b
- Archive SHA256: a090a56fef84e31516c4552bc1c634c07f2c8ae9cf7737de25d8ba8fbb1bde72

Automatic approval review previously rejected the older Cost75 upload
because approval for another archive did not authorize that exact payload
and AIcrowd destination. No Read5 upload has been attempted to circumvent
that rejection. The exact Read5 archive approval question is now pending;
question delivery is not consent. Cost75's source/archive remain untouched.

After explicit Read5 approval, execute unchanged `submit.py`, inspect the
acceptance receipt, then poll its returned ID. Do not run it beforehand or
silently substitute either archive. A successful upload is not a graded
milestone. Preserve all qualification evidence and the previous bank.
