# Birth metric v2 critical source audit: PASS

Reviewed candidate SHA256 `afbc69ce09232157e3e845a4d0b79ebb6a47f7bd0f3d906809ccb9801a8e7361` against S2tr oracle parent `f8da47f201f0b3c1b1e4a968e225f27fe33a4599f4df1da2f03a100bb31b2947`. The exact diff changes only the active FHM birth-selection call at line 17540 and appends its helper.

The helper ranks by the prescribed `min(g,1-g)*vc`, using the current layer's clamped preactivation variance `vc` and current `alive` mask. Both are established immediately before that layer's bank push. No later assignment replaces `vc` before birth selection. The unchanged count is the original threshold count rounded up to 32. Since `min(g,1-g)>=floor` implies `g>=floor`, and both counts use the same monotone rounding rule, the birth count cannot exceed the alive count. Live scores are nonnegative, while dead scores are -1, so the kept set is live.

Alignment is preserved throughout:

- `births` is the mask of exactly `selected`; BD and born-plane calculations use that mask.
- `newborn` gathers base columns, t, and s by the same `selected` array. The birth coefficient a is gathered in the same order, then lam appends the complete +a block followed by -a.
- F4C gathers every per-unit scalar by `selected` and receives the already selected base and a. Its six coefficients therefore retain pair order.
- K4L gathers its complete feature stack by `selected`, then NP uses the resulting coefficients in the same plus/minus block order.
- NP gathers its new hub features by `hubs=selected` and appends both K-wide blocks in cohort order.
- Shared-old commit stores only cohort start and `2*len(selected)`; joins require complete paired blocks, not neuron-index ordering.

The original `select` already returns stable score order, not sorted neuron indices. No selected/hub index-sorting or contiguity requirement was found in these consumers.

AST comparison confirms the latest definitions of `_astra_fhm_core`, `select`, `_f4c_birth`, `_k4l_birth`, `_np_update`, `_shared_old384_commit`, and `_astra_bcc_newborn_impl` are identical between the S2tr parent and valid #336342. The same one-site source edit and helper can therefore be ported to #336342 without changing these contracts. Runtime qualification of a port remains separate.

This verifies implementation of the fixed ranking rule and state alignment. Variance weighting is a hypothesis for useful birth importance, not an identity for the full signed cubic tensor norm.
