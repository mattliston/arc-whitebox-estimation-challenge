# Current S32 FP64 consumers — source/metadata only

The exact rank35 S32 census bills 276,549,645,399 FLOPs. Rows tagged float64 total 6,088,479,440 (2.201586%); this includes casts and mixed signatures, so it is not a universally halving-convertible amount. The older remaining_precision_consumers audit used a 402.8GF historical card with 1.096% FP64 share. Its wide-product FP32 and no-unused-covariance conclusions still apply; its numerical dtype percentage is not current.

| Consumer | FP64-tagged FLOPs | Full bill |
|---|---:|---:|
| Direct prediction-loop operations (by line, no inner function) | 1,451,794,660 | 0.524967% |
| Birth-direction update | 1,196,222,484 | 0.432553% |
| Nyström join/core | 836,581,802 | 0.302507% |
| Nonlinear pair batch | 712,958,008 | 0.257805% |
| GNN feature extraction | 633,092,096 | 0.228925% |
| Tail trace | 332,949,268 | 0.120394% |
| First Gaussian pair map | 201,711,622 | 0.072939% |

Across consumers, FP64-tagged matmul is 792,330,752F (0.286506%), einsum 563,457,536F (0.203746%), solve 476,752,554F (0.172393%), and sum/mean 648,652,296F (0.234552%). FP32 Winograd add/sub is excluded. FP16 has no cheaper tariff.

The only remaining individual named group above 0.3% is `_shared_old384_push`, candidate lines8325–8343. At L7 and L10 it computes `core=(R64 @ solve(K64,R64.T)).astype(float32)` on384² and320² cores. Solves cost301,989,888+174,762,666F; products226,197,504+130,867,200F. Together833,817,258F (0.301507%). Eliminating only the FP64 tariff premium gives a gross416,908,629F / 0.150754% ceiling before changed casts. Keeping the FP64 solve and changing just the terminal product gives178,532,352F / 0.064557% before casts. This is an actual precision boundary independent of marginal Gaussian arithmetic, but no current-native FP32-core endpoint receipt was found in the bounded targeted history search.

It is not a justified low-risk recommendation: checkpoint50/51 origin receipts show native sketches conditioned around66,000–81,000 and meaningful projector/core sensitivity. The regularization audit found positive eigenvalues and attributed the larger defect to sketch/QR, not a demonstrated safe lower-precision solve. Those studies are not an FP32-core endpoint failure, and this report does not label the untested boundary closed by measurement.

The prediction-loop0.525% is dispersed across K21/k22 construction, central-moment recentering, trace/star assembly and mean/b propagation; no single removable/reducible >0.3% contraction is hidden there. Pair batch+first Gaussian together are0.330744%, but the existing porter `pairmap_f32_s6b.py` is an old, cost-gated Torch proposal that also narrows marginal64; it is not a qualified current precision component. No new candidate is proposed. BD-only FP32 and covariance-diagonal changes remain closed by their actual endpoint evidence.

No model, arrays, targets, numerical job or candidate build was executed.

Evidence pins:
- `hill_climb_s32_bill_census_v1/census.json`: `0bac92025c957aa97b1e96a540eebb80b9fea41e8bfbaf865c2272c07f1d16e4` (5108240 bytes)
- `hill_climb_star_projection32_v1/candidate.py`: `f226a45750e6c3851e50967ba96f1742bb7bf563d43c1adfc2988fd674041aef` (930166 bytes)
- `teammate_remaining_precision_consumers_v1/RESULT.md`: `5eb7abdfaf29ce8d499ec1a925b078c643ce7d7a5d13af059fd90c15cf99dbe3` (2448 bytes)
- `current_nystrom_origin_v1/RESULT.md`: `d9f7e15c893256c324c01b0d46f0fbf187f3a913e1f0d2f3f1f0c1002e3aeafc` (2555 bytes)
- `hill_climb_birth_direction_fp32_v1/train16_assessment.json`: `693b0fb7852e942db8fa9d0b8b00114ef57181f685b2d3368bdd6654ef57d37c` (28493 bytes)

## Complete current decomposition and combined-port ceiling

`teammate_s32_fp64_charge_decomposition_v1.json` contains all consumer groups and every function/line/op aggregate, paid FLOPs/calls/layers, actual matmul/einsum operand shapes, and dtype signatures (100% of the6.088479GF). Missing pointwise shapes are explicitly absent at the charge callsite; source, not guessed metadata, identifies the1024² planes. Its SHA256 is `8435647349e2e883542e469972b86cee8840cead19a2f5d3dc86e395237d3ed2`.

| Consumer | FP64-tagged FLOPs | Cumulative FP64 coverage |
|---|---:|---:|
| `_predict_binary` | 1,451,794,660 | 23.845% |
| `_bd_update` | 1,196,222,484 | 43.492% |
| `_shared_old384_push` | 836,581,802 | 57.233% |
| `_nonlinear_pair_batch` | 712,958,008 | 68.943% |
| `features` | 633,092,096 | 79.341% |
| `_ott_trace` | 332,949,268 | 84.809% |
| `_nonlinear_first_gaussian` | 201,711,622 | 88.122% |
| `_gn_dcov` | 136,980,480 | 90.372% |
| `omitted_radial_source` | 127,456,108 | 92.466% |
| `_lr_factor` | 95,600,640 | 94.036% |
| `_g5_bcast3` | 94,371,840 | 95.586% |
| `_flms_cov_star` | 88,997,888 | 97.047% |

The residual beyond the earlier five named groups is mostly the1.451795GF directly in `_predict_binary`, then tail trace, first-Gaussian, GN correction, omitted radial, selector norm and broadcast helper work. Direct-loop largest sites:3291 K21 assembly156.270MF;3372 born125.860MF;3300 k22 construction125.829MF;3406 star assembly110.147MF;3329 covariance recenter110.116MF;3359 trace94.403MF;3331 recenter correction88.080MF. The JSON includes every smaller site.

The installed `_budget._charge_op`1840–1885 charges the already measured delta by `resolve_billing_dtype` over participants, then `rate_for`; `_dtype_billing.py` documents widest participating operand/store and weak Python scalars. All recorded signatures containingfloat64 are real-only and resolvefloat64/rate2, including float32+float64 and float64 inputs withfloat32 out. Thus the2.201586% rate2 classification is verified here; it does not tell which input/storage caused promotion or make every charge avoidable. Matrix products use captured shape metadata; vector/scalar/matrix pointwise distinctions require the pinned source. FP16/FP32 bothrate1.

A combined non-marginal plane port cannot support a1% forecast in this bounded tariff scope. Keeping closed BD and all Nyström work unchanged leaves4,055,675,154F: halving absolutely all of it is0.733263% gross. Keeping only the Nyström solve64 and allowing its terminal product32 gives0.798320%. These generous figures still include mandatory scalar means, marginal moments, radial and normalizers, plus existing materializations; they omit all newly needed casts. Even granting deletion of every remaining recorded cast/copy/gather/put/stack/concatenate charge rather than merely halving it gives only0.809214% /0.874551%, respectively. They are upper envelopes for dtype/materialization changes, not score or runtime predictions, and do not cover a new algorithmic deletion of arithmetic.

The concrete plane boundary is source947–955: FP32 Hermite series reduces into covariance/pk21/pk22 then multiplies by FP64 SVV; centralization and radial reduction follow. Carrying temporary off-diagonal planes in32 could be combined with loop K21/k22/inherited/born/trace/star assembly and feature normalized planes. Essential vector m/r2/r3/r4, diagonal replacements, all scalar reductions/normalizers, and the Nyström solve stay64; BD must remain the original64 routine. Matrix cast-back and diagonal refill costs prevent treating cheap vectorcasts as the entire implementation. In particular `_g5_message_edge_base` already stores R/S into32 but F64 inputs/normalizers still bill2; output narrowing alone earns nothing. Accuracy teammate owns the one coherent prospective cast plan; this audit establishes no .7% achievable forecast and proposes no model build.
