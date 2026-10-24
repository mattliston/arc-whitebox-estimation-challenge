# Conditional edge bias/output — source-only design, 2026-10-10T23:28Z

Status: plausible capacity lever; no performance evidence, no native fragment or trainer patch, no experiment launched.

S46 native source is the candidate in engineer/logs/quick_s46s_fresh_a.json (63f89e8d). Its _g5_message_edge_base (7757–7788) uses one shared _EDGE_PARAMETERS dictionary on layers 8–14. e.bias and e.output are 16 CHANNEL vectors, not per-layer values. _ecut_raw adds bias at 10562 and applies output at 10582. Giving those two operations different precreated constants changes neither their arithmetic count nor their tensor shapes.

## Native insertion and cache invariant

Insert `t = _edge_band_parameters(layer, t)` AFTER `k = _e2_edge_consts(t, n)` in _g5_message_edge_base, before consuming t's fields. The helper returns a cached shallow dictionary for the chosen band; only bias/output differ. All other entries are the shared array objects. Keep the underlying global _EDGE_PARAMETERS object unchanged.

This ordering matters: _e2_edge_consts (3869–3877) caches R/S planes by n AND dictionary identity (`k['t'] is t`). Replacing the shared dictionary before that lookup would rebuild two 1024x1024 float32 planes and channel views at every band transition, adding requests and allocation churn. Selecting after the lookup preserves that cache exactly. _ecut_raw does not cache bias/output; it directly reads the provided dictionary, so no other hook is needed.

The helper should build full float32 bias/output vectors once per process from JSON-exported constants, then reuse them. No runtime base+delta operation. For B materialized bands, expected added setup is 2B asarray calls and 128B payload bytes. Three bands: 6 calls / 384 bytes; seven per-layer vectors: 14 calls / 896 bytes. These are source counts, not measured platform wall time. Ordinary prediction has zero additional fnp calls or flops, zero new einsum signatures and no growth in the response_rows x alive_columns x16 activation. Every proposal must still pass measured setup/C/slow-stage release checks.

Existing a49 BAND head lever has the same constants-only idea and measured 0 requests/layer, 6 once; it cannot be reused directly because it wraps _hc_bias, not the edge. Avoid global parameter mutation and preserve shared edge-cache identity as above.

## Minimal trainer representation and export

First screen separate bias/output gradient contrasts at current cut-sim fitted point. Root priority remains simulator certification/refit before this screen. The stale e.* result and negative jmL line-search coefficient do not prove band heterogeneity: cancellations or an overlong direction are also possible.

Prespecified compact model: bands [8,9], [10,12], [13,14], matching the late join regimes. Per band, 16 bias + 16 output values: 96 versus 32 shared, or 64 extra degrees of freedom. A seven-layer screen can diagnose signal, but band definitions and acceptance data must be fixed before measuring gains. Keep bias/output as separate arms so they do not disappear in one large e.* direction; shared e.* may still update jointly.

Torch source_environment stores the original edge dictionary in cd.env['_EDGE_PARAMETERS'] (model_bank_research.py84–99). A card-local _message_edge_source wrapper can temporarily set this environment entry to a shallow dictionary whose bias/output are base tensors plus band-specific differentiable offsets, call the original source, and restore in finally. It must use the layer argument at each checkpoint replay; never detach selected leaves or cache a value from another layer. This leaves cd.edge's context_left/right unchanged. Initial offsets all zero must reproduce the existing forward/gradient control. Keep the leaves outside the original fixed-shape e/T schema unless trainer intentionally versions it.

Export alongside candidate.py/physics_dials.json as edge_band_weights.json with:

- schema: edge_band_v1
- edges: [[8,9],[10,12],[13,14]]
- bias: B x16 full float32-rounded values (or null per band for shared)
- output: B x16 full float32-rounded values (or null)
- parent_candidate_sha256 and originating checkpoint/head/request identities

Builder support would be a new optional EDBAND@name lever/converter; absent file means off, unchanged source. Validate sorted disjoint integer ranges within 8–14, finite exact shapes, and parent identities. The head table/physics dials retain their meanings. Native fragment needs one asserted source insertion and Python helper/constant dictionaries, no changed numerical signature. Export full values, not delta arithmetic.

## Evidence gates if the screen pays

1. All bands copied from shared vectors: byte-identical native predictions (651/378), same per-layer operation counts; independently verify R/S cache identity and no repeated allocations.
2. Nonzero torch/native band perturbation: per-layer change parity, including downstream final layer; native first/warm request delta only from once-created constants.
3. Gradient directional control relative to original detached-carrier protocol; fresh sizing/independent acceptance, then one untouched judgment.
4. If improvement survives, native safe+twin parity and ten-network comparison, normal einsum and release gates. No gain is assumed in this design.
