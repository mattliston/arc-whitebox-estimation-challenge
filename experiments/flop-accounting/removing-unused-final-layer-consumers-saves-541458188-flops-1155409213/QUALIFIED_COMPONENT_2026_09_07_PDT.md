# Exact terminal component qualified and retained locally

Completed September7,2026 PDT (package receipt September8 02:36:26 UTC).
Phase2 only. No submission or milestone; official #329926 remains
4.982231460183304e-9. The strict goal remains below1e-9, with banks below
4e-9,3e-9,2e-9. This note does not change an older receipt or closeout.

## Same-configuration independent14 result

Candidate estimator.py SHA256
07b71a03e13f8daed17e0ee529c4113e88cd89c2cceb29ea0e2f148084e74d30.
Parent candidate_response_pair_polynomial_f32_v1.py SHA256
0b84ab6fa96d32804c9adec350d41766e47d09a71a936b95652893bb1e364a96.

The candidate removes only unused final-layer outgoing pair channels and
radial repair. It retains all16 current P heads, all intermediate state,
and all final head inputs. Every served layer on every panel network is
byte-identical to the parent. All14 complete bills are strictly smaller.

| Fixed panel | Noise-corrected raw, unchanged | Complete mean FLOPs | Adjusted |
| --- | ---: | ---: | ---: |
| Public8 | 2.4316999741373497e-8 | 469195523392 | 5.196969836377938e-9 |
| Historical virgin6 | 1.996803939251903e-8 | 464946370412.1667 | 4.23000801867066e-9 |
| Combined14 | 2.245315959186444e-8 | 467374457829.2143 | 4.782557628789104e-9 |

Same-frame parent combined14 adjusted4.788089831892398e-9; ratio
.9988445907872394, a .1155409213% gain. Public and virgin ratios are
.998847451516621 and .9988399045827347. Mean saving541458188 FLOPs.
Independent4M truths, unclipped per-network MC noise correction, and the
mean of per-network raw*max(.1,F/2^41) are used. Observed combined14 raw
4.074403982689888e-8; subtracted mean truth-noise variance
1.8290880235034444e-8. This is local score evidence, not an official score
or a newly untouched holdout. Initialization is counted once in14 calls.

Native panel maximum wall19.4938634s and peakRSS7091453952 bytes. Those
in-process checks do not substitute for production residual qualification.

## Persistent NumPy-free client qualification

Fixed five contexts35,100,500,3016500,35 used one persistent isolated
local OpenBLAS server and NumPy-free client. All five match their parent
remote outputs byte-for-byte and have strictly lower complete FLOPs.
Wall29.6402-30.4614s; residual .117763-.128788s; combinedRSS7178469376
bytes, below the unchanged8e9 gate. Maximum native-to-remote discrepancy
3.814697265625e-6, below the pre-existing1e-5 cross-backend tolerance.
The source/backend/resource tolerances were fixed before acquisition.
All owned runtime processes terminated. No official Linux timing guarantee
or contemporaneous parent wall-speedup is claimed.

## Local package and immutable evidence

Whest0.16 validate, package, and validate-package all passed. The archive
contains only regular estimator.py and manifest.json members; packaged
estimator bytes equal the qualified source exactly. Generic contract checks
are separate from native14 and remote5 numerical/runtime evidence.

- panel/candidate/result.json: 302d9bbaf1327e0f12aa8746406536e24ed85beca6ec23eb807c8a2a03cd7344
- panel/score/result.json: 91dfa3cd9067433c738c51ed5aec5d0fc5b807f2efd6ee468331b5f54fe076f9
- remote/result.json: b9695e839d4ba190fc82d933b5981af9409ef679c869efe8c8f470bc70d713bf
- package_result.json: a035999bfe6593f1500eb96777c944eaf50c1d7cd264224ba7a519a3dc7f7436
- terminal_dead_consumers_07b71a03_whest016.tar.gz: ce971d6ed80c0caed42b53505bba17abcdef6467f1f8e3bbf6558e4f02174839

Preserve this exact component for composition. It cannot independently
reach a milestone; no upload, forum post, refit, commit, promotion, frozen
edit or deletion is authorized by this retention. The one response-row
exception remains completed and closed.
