# Fixed Euclidean information pilot: FAILED, not resumed

September9,2026 PDT. All20 TRAIN reverse calls and13 served-vector JVPs
retired. The next JVP (call33,rank52) produced and saved its full output,
then failed the unchanged scalar forward/reverse identity. No nonlinear
call, sealed step, candidate or complete20-network magnitude verdict exists.

The failed rank had reverse-dot -0.26688142675273085 versus served-JVP
readout -0.26685206609253775. Absolute gap2.9360660193100685e-5,
relative1.1001387601357408e-4; original tolerance2.6688242675273088e-5.
It missed by1.100134638x. Reverse-dot cancellation ratio16.94437763 is
diagnostic, not a proof of cause. Source review found no rank, layer,
target, scalar-normalization or tangent-serving mismatch. Raw/served/scalar
primal and frozen-state bytes matched. FP64 readout versus FP32 reverse
rounding/order remains a possible contributor, not a proven explanation.

Separate saved-artifact diagnosis PASS: no models, targets, parameter
proposals or changed tolerances. The incomplete14-network FP32-target
proposal quadratic has a2.078909925052098e-8,b-2.0616049757948547e-8,
c9.775399728938367e-6, conditional line step0.00210897255658 and
ratio0.99790858263551. This is a post-hoc failed-run diagnostic, NOT actual
raw, native, adjusted or full20 evidence. Its approximately0.209% scale
does not justify rerunning merely to clear the marginal numerical gate.

Watch130.762301875s,33 retired markers,combined RSS peak4362682368 bytes,
no added swap,max completed sample interval0.363246625s,cleanup PASS.
Original execution and its finalization seals remain FAILED. The earlier
target-free qualificationV3 remains a separate PASS.

Worker result73bec2488604faebd40ecd2024990969a0983df8fe6b8c9b9004bb04333cf3f9;
completion302d0b2d5e3a2337a9133543467c11fe504b866ab79721ab15abae9945af49d2;
finalizationec6dc2ed5f96930259e2b8c7e6599adbc7c2b4ea92c729ec295049a74d006602.
Watch result5cd65c4fbcd69f0a6daf0a3f82307edfa06bdf818e31e5345b283d230139478c;
completioneda36a0ff6a238946f585c54514d078c8ecfb27f99fb3731b9f0dae2715ed7ca;
finalization0d73122f67aadf2c28a2debc6668ff9f37791950dfa6f8562aff4ff38ef0e58c.
Saved diagnosis resultd486c92e2aabc4417444cfcc0a9285a2ccff7ffeec6f6f930781d715f438c6ee;
completion058900107cff0aad926d9a6f4e25a1a2a931db18b123401b473805e1b49ce4b0.

Decision: no retry, gate relaxation, long fit or native candidate from this
fixed Euclidean direction. It is not a capacity bound on all4096 parameters.
Any different proposal rule must have a new prospective native-objective
protocol; gradient/feature conditioning alone cannot establish raw gain.
