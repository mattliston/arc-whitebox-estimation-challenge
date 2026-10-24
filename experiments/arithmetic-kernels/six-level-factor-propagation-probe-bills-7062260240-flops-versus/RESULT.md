# Fixed occurring push probe: admission passed, not a score

Completed September 9, 2026, approximately21:03UTC. Phase2 only. All three
roles completed one cold and two warm fixed(1024,832,8192) products. The
old maximum-read test and its failed1.35 timing gate remain unchanged.

| Role | Exact F/product | Warm wall range | RMS versus synthetic FP64 |
|---|---:|---:|---:|
|D4|8,417,538,304|.212674542–.218106459s|6.7533409436236544e-6|
|breadthD6|7,062,260,240|.565592292–.583140250s|2.76109268348764e-5|
|serialD6|7,062,260,240|.704913292–.734080833s|2.76109268348764e-5|

All outputs finite, own cold/warm repeats byte-exact, and serialD6 exactly
matches breadthD6 output bytes. Neither D6 realization is byte-identical
to D4. Maximum errors versus the synthetic FP64 reference were
8.162350203644309e-5 D4 and3.2272313182524215e-4 D6. These are matrix
product errors, NOT final-neuron MSE or a network numerical certificate.

Both new serial bindings cost the predicted additional561,056,000F cold,
on top of all old setup. Complete probe bills, including setup and the
three product/verification contexts, were26,321,568,512F D4,
25,792,531,760F breadthD6 and22,816,790,320F serialD6. These microbenchmark
totals are not whole-estimator bills.

The exact prospective timing calculation was

    max(0, .101612125-.098359459)
      +6*max(0, .734080833-.212674542)
      =3.131690412 seconds <=8 seconds.

Maximum serial product residual was .003343796s, below .20s. The separately
reported first cold-frame serial-minus-D4 difference was .398567084s.
The frozen six-occurrence d50 price opportunity is still1.604% cold /
1.728% warm. P1's population shape counts and raw/complete-score change
remain unmeasured. The timing arithmetic is a partial-test admission
heuristic, not a60s production or Linux-worker bound.

Independent supervision:66.861762708s total; maximum sampled combined
watcher/worker/client/server RSS3,162,980,352B; zero added system swap;
maximum observed interval .247759291s. Owned processes retired. Swap after
completion remained752.75MiB. No MLP, target array, fit or upload ran.

Decision: admit design of ONE fixed partial-shape P1 whole-card test;
do not automatically launch it, broaden the shape/depth schedule, credit
unchanged raw MSE or submit. The next selected numerical experiment is the
separate frozen NODE/P1 compatibility test, once its source review passes.

Frozen source pins:

- PROTOCOL50dca9fe2ffab1bb51464cb98cbf1d643458abeedf750f9feb452484d1b120ca
- client8ea425a33de0fb1aca6f3ae0e5c38f2d8991b68b5f6d27eae69969715ed8328f
- run209962dc9aa51387bae6d1802662d02072df09d7f437c943344a6f768e9f2491

Sealed results:

- Supervisorbf79b160f8952db1f433093c245c5497d2d25a623db6888cafa13f0ae48aaeb2
- Completion6fafd4fa5d8e5e1ce7f6f4ca8886919404ff35eb1417f5c8d7c9238cda1b6b5f
- Measurement09ec65c86c6be60b318a3578652cd540dd2cba466af6292a54d46d3c36add0fc
- Measurement completion2b1dc29258c87a155b76702cf0cf29adae6015b0c834930858de456d68defe7e

Official bank remains #330390 adjusted4.53886349950395e-9. No milestone
or improved score was established by this primitive qualification.
