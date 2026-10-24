# Frozen-COMP receiver-by-birth one-step information experiment

This is a new, prospective, source-only proposal. It does not change or promote any historical failure, including the NODE pilot, R10, R2, or native-gradient qualification failures. The retained parent is the literal composition `2e13d40a21e62691155918719f8ec088f581d495ae17800c5503c3b56504918a`: exact P1 P, frozen NODE A, and original cstar. No NODE/P retuning, teacher trajectory, target-derived input feature, sample generation, or new covariance map is allowed.

## One operator

The only new parameter is one shared, nonsymmetric FP32 Theta[64,64], initially zero. In birth layers 1 through 14, use the actual returned pre-mean-correction GNN state H. Recompute the original NODE standardization without changing the NODE coefficient function:

    Z = clip((H-mean(H,0))/sqrt(mean((H-mean(H,0))^2,0)+1e-6), -8, 8)
    R = tanh(((Z @ Theta) @ Z.T)/64)

First execute the unchanged original birth_action with the unchanged current NODE coefficients, obtaining its actual stored FP32 base B0 and old repeated-plane increment. Widen B0 to the card's physical dtype, B. Set V=B*R, c=sum(B*V,0)/sum(B*B,0), with denominator exactly one only for exactly zero-norm columns. Set F=V-B*c and its diagonal to zero. Store B1=FP32(B+0.1F); where B0 is exactly zero retain B0, including its sign bit. This support branch does not branch on Theta and does not suppress the new derivative at Theta=0. Use the actual stored difference D=widen(B1)-B, not an intended unrounded displacement.

Add only the new plane increment to the old increment:

    w = FP32(a) * (FP32(t*alive)-FP32(s*alive)) * births
    z = FP32(a) * (FP32(t*alive)^2-FP32(s*alive)^2) * births
    extra = (2*B*D+D*D)*w[None,:] + z[:,None]*D.T

The displayed stored scalars are widened before the original physical-dtype operations. Set extra's diagonal to zero. The original CP pair scalars, lambda values, selected birth columns, full carrier, diagonal, star restoration, radial source, and all downstream consumers remain unchanged. There is no additional birth at layers 0 or 15. In real arithmetic F is columnwise perpendicular to B, ||0.1F|| <= 0.1||B|| and the multiplicative factor lies in [0.8,1.2]. Stored FP32 orthogonality is not asserted; its defect is reported. This receiver-dependent field is not the old four per-column NODE coefficients, nor the closed 192-parameter conditional-lambda experiment.

## Complete cost, not a measured bill

At n=1024 and h=64, the two FP32 GEMMs in each of 14 layers cost
14*((2*n*h*h-n*h)+(2*n*n*h-n*n)) = 1,980,891,136 F at the installed textbook tariff. A conservative literal-source allowance is
14*(the two GEMMs +120*n*n+100*n*h+1000*n)+8192 < 4,000,000,000 F.
It covers repeated standardization, tanh, physical FP64 projection, denominator handling, stored casts/support selection, full extra-plane repair, initialization, guards and copies. Thus fixed-geometry planning is below about 472.314B total, not an actual bill. Changed state can change selection sizes and subsequent costs; a complete native owned bill remains mandatory. No raw-gain forecast is supported. The >=34% raw gate accounts for this overhead when targeting sub-3e-9, not a promise of reaching it.

## Qualification: no targets, no optimization

`qualification.py --run --source-sha256 SHA --model-sha256 SHA --protocol-sha256 SHA --watcher-sha256 SHA` is inert otherwise. It runs only under the separately reviewed supervisor, one MPS process, nice10, five BLAS environment variables=1, Torch threads=1, no MPS fallback, no threadpoolctl import. Whole qualification is 180 seconds including admission and sealing; combined host RSS <12GB, zero added swap, MPS driver <16GB configured by the allocator API, actual completed external samples <=1 second and <=120 seconds without a fully retired call. No stage auto-continues. A separately required finalization.json binds the completion digest within the same deadline.

Before numerical imports, bind the complete COMP literal build and original CO/BD/NODE source ownership. Bind only the two TRAIN fixtures and sealed COMP native references: panel_v2/terminal result `de766126360ce24263f548c27a2cb51f60e52107386c4aeb7cf15f9ea20db625`, completion `0973d38be5b37c2ff99d5b6243119a22feb5fe591c851ed4aac0aae7346a9073`, rank35 output `1df707ad91b1b75b4d50eb6d63d036e0633d1cb9e9a39c3853b0d78ccfe894ea`, rank100 output `b059807fff1e4d1a41766dc70b0b05a0f31b23f819fe33fb144370b0bc1b6721`. Their canonical weight hashes are respectively `14563b956b4e326aee081263077c0dd1dcb86037a0b7870dc58a5e33865f47f2` and `c4923171c8ddef28396aabd70335dad29141361efc13865ccb660a4a57421ee5`. No truth file is opened or hashed.

First perform fixed tiny n=8/depth=3 CPU and MPS tests using deterministic custom weights, not a truncated 1024-neuron card. Check zero-vs-bypass raw/served bytes, frozen P/A/cstar/weights, all3 domains and8 selectors; compare the new stored-pair repair with an independently written direct cubic CP contraction; check real-arithmetic projection and zero-support preservation. The tiny-only synthetic scalar uses its final row /sqrt8 plus0.05 times earlier rows /sqrt16; it is not the original16-layer scalar. Compare CPU/MPS small-card output <=1e-5 max absolute and nonzero Theta gradients within 5% relative L2. Reverse-dot/forward-AD scalar equality must meet relative 1e-4 plus absolute 1e-10. A disconnected/zero/nonfinite derivative fails; no fallback finite difference or alternative direction is selected.

Then execute exactly seven full-size MPS calls in this order: zero35, zero100, nonzero-backward35, nonzero-JVP35, nonzero-backward100, nonzero-JVP100, nonzero-backward35-repeat. The nonzero fixture is Theta[p,q]=0.05*sin(0.37*(p+1)+0.19*(q+1)), evaluated in NumPy FP64 then stored FP32; the fixed direction is cos(0.23*(p+1)-0.41*(q+1)), normalized to Frobenius norm64 in FP64 then stored FP32. No perturbation is learned. Use the original sealed CO cotangents and unchanged original scalar grouping (final row /sqrt1024, plus 0.05 times earlier rows /sqrt15360), in each backend's physical dtype.

Zero outputs must match the archived current-COMP native outputs within 1e-5 maximum absolute. Every call requires all16 unfloored variance domains, all47 original selection records, finite outputs, frozen P/A/cstar/Theta, and actual graph retirement. Reverse calls require actual checkpoint replay. Each JVP must have byte-exact primal/served output and identical ordered selections/variance bytes to its same-rank reverse call, a nontrivial tangent, and the reverse-dot identity above. The repeated35 reverse call must match all output, gradient, domain and selector bytes exactly. Save each attempt and numerical evidence before its comparison gates. Print `{ "event": "retired", "call": N }`, flushed, only after complete retirement; call0 is the finished tiny suite, then calls1..7 are the seven wide calls. Seal result/completion and rehash every input/output/source before PASS.

This qualifies the new Torch execution and AD route only. The native controls concern the zero primal, not a derivative. It does not certify a native gradient, waive old 5% native secants, establish real-number/FP32 derivative equivalence, score a candidate, or establish family capacity. Failure of an unsupported MPS forward-AD primitive is a concrete prerequisite failure, not permission to substitute a proxy.

## Later one-step information gate, not implemented or admitted by this qualifier

Only after qualification and separate source review: fixed20 TRAIN35..58 excluding37/39/42/48; P/A/cstar frozen. Obtain20 gradients at Theta=0 of equal-network final served raw MSE /2e-8, aggregate them, and freeze one direction d=-64*g/||g||F. Obtain20 qualified JVPs at the same point. With actual TRAIN residuals e, a=mean||e||^2, b=mean<e,Jd>, c=mean||Jd||^2, freeze s=clip(-b/c,0,1) once. Require predicted ratio <=0.90 before20 actual nonlinear TRAIN forwards at Theta=s*d. Two labels are fixed before any run: actual nonlinear TRAIN ratio <=0.66 is a standalone raw lead; <=0.90 is a useful raw-component lead. No optimizer, step sweep, checkpoint selection, rank change or feature change is allowed. This is one Euclidean direction, not an upper bound on the full4096-dimensional family; failure does not prove representational closure.

Only a separately reviewed continuation may then seal fixed native predictions on37/39/42/48 plus repeat before their truths. The standalone label requires mean raw ratio <=0.66, mean own-cost adjusted ratio <=0.67 and4/4 adjusted wins against COMP. The useful-component label requires mean own-cost adjusted ratio <=0.92 and4/4 adjusted wins, following the >=10% actual nonlinear TRAIN gate. This second label does not establish sub-3e-9: combining it with any prospective compression requires that separate component's actual complete costs, qualification and joint raw validation. Neither label may be chosen post hoc. Those are small-panel evidence, not universal bounds. A full native40 own-cost gate and both independent14 groups, plus all production/release requirements, remain necessary; no automatic advance or submission.
