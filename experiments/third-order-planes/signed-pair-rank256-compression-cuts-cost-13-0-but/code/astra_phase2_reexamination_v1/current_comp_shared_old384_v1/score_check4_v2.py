"""Saved-score metadata revision for the fresh CHECK4 v2 acquisition only."""
import ast
import hashlib
from pathlib import Path
import runpy

_V2_HERE = Path(__file__).resolve().parent
_V2_OWNER = _V2_HERE/'score_check4.py'
_V2_OWNER_SHA = '2dca82867555624df7df1236bb1d3b5da301f1593556cbf2a72c95560e297376'
if hashlib.sha256(_V2_OWNER.read_bytes()).hexdigest() != _V2_OWNER_SHA:
    raise RuntimeError('Frozen unexecuted CHECK4 score owner')
_V2_PARENT = runpy.run_path(str(_V2_OWNER),run_name='inert_check4_score_v1_owner')
# These are exclusively the original inert source adapter's metadata helpers.
for _key in ('_BASE','_BASE_SHA','_CHECK4_PROOF','_check4_prerequisite'):
    globals()[_key] = _V2_PARENT[_key]
_V2_SOURCE = _V2_PARENT['_derived']
_V2_EDITS = [
    ("HERE/'check4.py'", "HERE/'check4_v2.py'", 2),
    ("HERE/'watch_check4.py'", "HERE/'watch_check4_v2.py'", 2),
    ("HERE/'CHECK4_PROTOCOL.md'", "HERE/'CHECK4_V2_PROTOCOL.md'", 2),
    ("HERE/'CHECK4_WATCH_SPEC.md'", "HERE/'CHECK4_V2_WATCH_SPEC.md'", 2),
    ("HERE/'check4_v1'", "HERE/'check4_v2'", 3),
    ("HERE/'check4_watch_v1'", "HERE/'check4_watch_v2'", 1),
    ("'current_comp_shared_old384_native_check4_v1'", "'current_comp_shared_old384_native_check4_v2'", 1),
    ("'current_comp_shared_old384_native_check4_watch_v1'", "'current_comp_shared_old384_native_check4_watch_v2'", 1),
    ("HERE/'score_check4.py'", "HERE/'score_check4_v2.py'", 1),
    ("HERE/'watch_score_check4.py'", "HERE/'watch_score_check4_v2.py'", 1),
    ("HERE/'SCORE_CHECK4_PROTOCOL.md'", "HERE/'SCORE_CHECK4_V2_PROTOCOL.md'", 1),
    ("HERE/'score_check4_v1'", "HERE/'score_check4_v2'", 1),
    ("'current_comp_shared_old384_score_check4_v1'", "'current_comp_shared_old384_score_check4_v2'", 1),
    ('PINS = {\n', 'PINS = {\n    _V2_OWNER:_V2_OWNER_SHA,\n', 1),
    ("    result['score_adapter_proof'] = _CHECK4_PROOF\n",
     "    result['score_adapter_proof'] = _CHECK4_PROOF\n"
     "    result['score_metadata_revision_proof'] = _V2_PROOF\n", 1),
]
_V2_DERIVED = _V2_SOURCE
for _before,_after,_count in _V2_EDITS:
    if _V2_DERIVED.count(_before) != _count:
        raise RuntimeError('Unique score-v2 metadata span '+repr((_before,_V2_DERIVED.count(_before),_count)))
    _V2_DERIVED = _V2_DERIVED.replace(_before,_after)
_V2_INVERSE = _V2_DERIVED
for _before,_after,_count in reversed(_V2_EDITS):
    if _V2_INVERSE.count(_after) != _count:
        raise RuntimeError('Unique score-v2 inverse span '+_after)
    _V2_INVERSE = _V2_INVERSE.replace(_after,_before)
if _V2_INVERSE != _V2_SOURCE:
    raise RuntimeError('Entire frozen CHECK4 scoring source restored')
ast.parse(_V2_DERIVED)
_V2_PROOF = dict(owner_sha256=_V2_OWNER_SHA,whole_source_inverse=True,
    derived_sha256=hashlib.sha256(_V2_DERIVED.encode()).hexdigest(),
    numerical_scoring_and_gates_unchanged=True,fresh_positive_acquisition_required=True,
    old_failed_acquisition_not_relabelled=True)
exec(compile(_V2_DERIVED,str(_V2_OWNER)+':fresh-acquisition-metadata-v2','exec'),globals())
