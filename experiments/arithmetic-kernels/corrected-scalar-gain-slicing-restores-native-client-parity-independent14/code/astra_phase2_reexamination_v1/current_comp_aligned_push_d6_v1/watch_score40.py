"""Fixed VALID40 saved-output score watch; no model or scoring arithmetic."""
import ast
import hashlib
from pathlib import Path

_here=Path(__file__).resolve().parent
_parent=_here/'watch_score.py'
_parent_sha='09277db8bf2fd12c881fa6399d81ee103b6012f0832c2061c35e5f4ee77a1c6b'
_ranks=(37,39,42,48,59,63,66,67,71,73,75,79,93,101,122,130,132,135,
    139,145,146,149,150,151,155,159,163,169,173,175,181,192,198,199,204,219,224,227,233,234)
if hashlib.sha256(_parent.read_bytes()).hexdigest()!=_parent_sha:
    raise RuntimeError('Frozen successful CHECK4 score supervisor preserved')
_original=_parent.read_text()
_changes=(
    ("HERE/'SCORE_PROTOCOL_V2.md'","HERE/'SCORE40_PROTOCOL.md'",1),
    ("HERE/'native.py'","HERE/'native40.py'",1),
    ("HERE/'NATIVE_PROTOCOL.md'","HERE/'NATIVE40_PROTOCOL.md'",1),
    ("HERE/'watch_native.py'","HERE/'watch_native40.py'",1),
    ("HERE/'NATIVE_WATCH_SPEC.md'","HERE/'NATIVE40_WATCH_SPEC.md'",1),
    ("HERE/'native_terminal_v1/result.json'","HERE/'native40_terminal_v1/result.json'",1),
    ("HERE/'native_watch_v1/result.json'","HERE/'native40_watch_v1/result.json'",1),
    ("HERE/'native_terminal_admission_v1/result.json'","HERE/'native40_terminal_admission_v1/result.json'",1),
    ("HERE/'score_v2.py'","HERE/'score40.py'",3),
    ("HERE/'SCORE_WATCH_SPEC.md'","HERE/'SCORE40_WATCH_SPEC.md'",2),
    ("HERE/'SCORE40_WATCH_SPEC.md':args.spec_sha256,TRANSPORT:TRANSPORT_SHA,**h.MON_PINS}",
     "HERE/'SCORE40_WATCH_SPEC.md':args.spec_sha256,TRANSPORT:TRANSPORT_SHA,**h.MON_PINS,\n"
     "         HERE/'watch_score.py':'"+_parent_sha+"'}",1),
    ("out=HERE/'score_watch_v1';scored=HERE/'score_v2'","out=HERE/'score40_watch_v1';scored=HERE/'score40_v1'",1),
    ("schema='current_comp_aligned_push_d6_score_watch_v1'",
     "schema='current_comp_aligned_push_d6_score40_watch_v1'",1),
    ("record['schema']=='current_comp_aligned_push_d6_score_v2'",
     "record['schema']=='current_comp_aligned_push_d6_score40_v1'",1),
    ('for r in (37,39,42,48)','for r in '+repr(_ranks),1),
    ("'Only four already-scored original truth files'","'Only40 already-scored original VALID truth files'",1),
    ('all_five_new_outputs_verified','all41_new_outputs_verified',2),
    ('all_five_domains_and_selectors_reverified','all41_domains_and_selectors_reverified',1),
    ("'saved_external_watch_revalidated','original_external_watch_still_failed','repaired_metadata_only'",
     "'exact_CHECK4_first4_score_replay','original_external_watch_still_failed','check4_expansion_prerequisite_passed'",1),
    ("'frozen_geometry_price_used_for_scoring','numerical_acquisition_repeated'",
     "'frozen_geometry_price_used_for_scoring'",1),
    ('==[37,39,42,48]','=='+repr(list(_ranks)),2),
    ("'All four exact completed score rows and original worker bounds'",
     "'All40 exact completed score rows and original worker bounds'",1),
    ("'All five frames sealed before any target decode'","'All41 frames sealed before any target decode'",1),
    ("len(record['truth_bindings'])==4","len(record['truth_bindings'])==40",1),
    ("'Exactly four original CHECK4 truths used by child'","'Exactly40 original VALID truths used by child'",1),
    ('advances_to_native40','advances_to_independent14',2),
    ("comparison['at_least_three_wins']","comparison['at_least_30_wins']",1),
)
_derived=_original
for _before,_after,_count in _changes:
    if _derived.count(_before)!=_count: raise RuntimeError('Unique VALID40 score watch span: '+_before)
    _derived=_derived.replace(_before,_after)
_inverse=_derived
for _before,_after,_count in reversed(_changes):
    if _inverse.count(_after)!=_count: raise RuntimeError('Unique inverse VALID40 watch span: '+_after)
    _inverse=_inverse.replace(_after,_before)
if _inverse!=_original: raise RuntimeError('Full original bounded score supervisor restored')
ast.parse(_derived)
exec(compile(_derived,str(_parent)+':fixed-VALID40-score','exec'),dict(__name__=__name__,__file__=__file__))
