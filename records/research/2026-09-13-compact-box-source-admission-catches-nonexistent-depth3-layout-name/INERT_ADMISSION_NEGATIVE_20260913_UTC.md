# Actual source-admission failure — preserve V1, no numerical stage

2026-09-13 UTC, actual stdlib-only session92047. Source derivation/inverses
and compilation completed; metadata_api then failed before geometry/cache/
parser completion with: Actual last-executed layout: _PACKED_SQUARE3.

The real frozen BOX _square_product line1114 maps depth3 to _PACKED_SQUARE,
not _PACKED_SQUARE3. The unrun FRAMES parent LAYOUT_NAMES/runtime and BOXES
NEW_VIEWS copied an inferred name incorrectly. The runtime tuple is eagerly
evaluated, so it would also NameError even on a different depth. This is a
source defect, not a numerical or runtime measurement. Independent source
review missed it; actual metadata admission correctly blocked execution.

No numerical imports, arrays, model or target read, stage, candidate file,
child process or network was allowed by the actual audit hook. The sandbox
also rejected nice for this stdlib-only call, as before. No test result is
claimed. Existing positive fixedGram unit is unaffected; it used the actual
_PACKED_SQUARE5 name and byte-authenticated original helpers.

V2 may fix ONLY the actual depth3 literal/identifier and missing final source
pins for the dynamically loaded FRAMES/client.py and PAD/client.py owners.
No rank, cache size, shape-policy, tariff, fixture, precision or timing gate
change is licensed by this factual source repair. All V1 sources and failed
preflight remain unchanged. Conditional price used the actual source-owned
depth3 layout, pending reconfirmation; no measured bill exists.

