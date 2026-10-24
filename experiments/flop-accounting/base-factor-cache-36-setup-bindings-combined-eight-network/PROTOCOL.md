# Exact component bundle: source preparation only

Assemble, in this fixed order:

1. Unchanged UVP base B-cache candidate `0308a99f…205fd5`.
2. Only the byte suffix of D3 prebind candidate `61a1f40b…13c6e` after its
   exact S32 prefix `f226a457…41aef`.
3. Exact `hill_climb_s32_prebind27_v1/fragment.py` from candidate
   `1b7f0244…770457`.

No component code, condition, arithmetic, source edit, cache bound or layer
scope changes. `build.py` uses only standard-library source/metadata operations;
it does not import or execute a candidate, model, array library or test.
It pins every component source/qualification receipt, extracts both exact
append fragments, verifies their identities, and independently reverses the
six B-cache source edits back to S32. `build.json` records all inverse proofs.

The resulting setup order is27 keys, nine D3 keys, then the original600 keys.
All636 keys are distinct. Expected complete fresh setup is660 bound/636 fused,
including the unchanged24 KS accumulator views. A/B/C initial counts are
197/255/184; capacities remain320/352/320. The setup wall guard remains4s.
The UVP cache retains its512,000,000-byte copy ceiling and prediction-local
cleanup. All changes use the existing arena/kernels and setup lifecycle.

The components have isolated two-network exact-output/resource receipts.
Their composition has not been executed or qualified. Prebinding changes
LRU order and first-use charges, while B-cache reuse changes later binding
incidence. Consequently neither bill reductions nor request/timing changes
may be added to predict a measured bundle gain. Cold first-use benefits are
distinct from warm behavior.

This artifact prepares a bundle for a later root-admitted measurement. No
resource driver, numerical execution, worker20, broad validation, standalone
1% claim, release admission or submission is included. Preserve source hashes
and qualify the exact bundle before making any such claim.
