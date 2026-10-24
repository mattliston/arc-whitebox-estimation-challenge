"""Versioned metadata consumer of namespace-isolated score_panel_v2; same math."""
import ast
import hashlib
from pathlib import Path

_F2_HERE = Path(__file__).resolve().parent
_F2_OWNER = _F2_HERE/'fidelity.py'
_F2_OWNER_SHA = 'a4748ba0f4007523837f7668600e8ac11141d1a1657d3a78edf2a6a12c27ff7c'
_F2_OLD_PINS = {
    _F2_OWNER:_F2_OWNER_SHA,
    _F2_HERE/'watch_fidelity.py':'b134f289dc01a09abd7c9f8aec2642fd15d61e6c52ce1a5343f83887f620b62f',
    _F2_HERE/'FIDELITY_PROTOCOL.md':'063d868c6630ce32577df1cce197a372c8a75c55385784730221e8c576bfe6d5',
    _F2_HERE/'FIDELITY_WATCH_SPEC.md':'800abb6639d0ad5929aac105dcde8c27a826a7ef9f651fd4e90f1df99984f239',
}
for _f2_path,_f2_hash in _F2_OLD_PINS.items():
    if hashlib.sha256(_f2_path.read_bytes()).hexdigest() != _f2_hash:
        raise RuntimeError('Unchanged original fidelity source owner')
_F2_ORIGINAL = _F2_OWNER.read_text()
_F2_EDITS = [
    ('watch_score_panel.py','watch_score_panel_v2.py'),
    ('score_panel.py','score_panel_v2.py'),
    ('SCORE_PANEL_PROTOCOL.md','SCORE_PANEL_V2_PROTOCOL.md'),
    ('SCORE_PANEL_WATCH_SPEC.md','SCORE_PANEL_V2_WATCH_SPEC.md'),
    ('score_panel_watch_v1','score_panel_watch_v2'),
    ('score_panel_v1','score_panel_v2'),
    ('watch_fidelity.py','watch_fidelity_v2.py'),
    ('fidelity.py','fidelity_v2.py'),
    ('FIDELITY_PROTOCOL.md','FIDELITY_V2_PROTOCOL.md'),
    ('FIDELITY_WATCH_SPEC.md','FIDELITY_V2_WATCH_SPEC.md'),
    ('fidelity_watch_v1','fidelity_watch_v2'),
    ('fidelity_v1','fidelity_v2'),
    ('return {OWNER:OWNER_SHA,**RUNTIME_SOURCES,',
     'return {**_F2_OLD_PINS,OWNER:OWNER_SHA,**RUNTIME_SOURCES,'),
    ('    env,helper_proof = h.support(); admit = env[\'Admission\'](result,tick)',
     '    result[\'versioned_consumer_source_proof\'] = FIDELITY_V2_SOURCE_INVERSE_PROOF\n'
     '    env,helper_proof = h.support(); admit = env[\'Admission\'](result,tick)'),
]
_F2_DERIVED = _F2_ORIGINAL
_F2_SPANS = []
for _f2_before,_f2_after in _F2_EDITS:
    _f2_count = _F2_DERIVED.count(_f2_before)
    if _f2_count == 0:
        raise RuntimeError('Present explicit fidelity metadata span '+_f2_before)
    _f2_positions = []
    _f2_from = 0
    for _f2_index in range(_f2_count):
        _f2_pos = _F2_DERIVED.index(_f2_before,_f2_from)
        _f2_positions.append(_f2_pos)
        _f2_from = _f2_pos+len(_f2_before)
    for _f2_pos in reversed(_f2_positions):
        _F2_DERIVED = (_F2_DERIVED[:_f2_pos]+_f2_after+
            _F2_DERIVED[_f2_pos+len(_f2_before):])
        _F2_SPANS.append((_f2_pos,_f2_before,_f2_after))
_F2_INVERSE = _F2_DERIVED
for _f2_pos,_f2_before,_f2_after in reversed(_F2_SPANS):
    if _F2_INVERSE[_f2_pos:_f2_pos+len(_f2_after)] != _f2_after:
        raise RuntimeError('Exact fidelity metadata inverse cardinality')
    _F2_INVERSE = (_F2_INVERSE[:_f2_pos]+_f2_before+
        _F2_INVERSE[_f2_pos+len(_f2_after):])
if _F2_INVERSE != _F2_ORIGINAL:
    raise RuntimeError('Entire original fidelity source restored exactly')
ast.parse(_F2_DERIVED)
FIDELITY_V2_SOURCE_INVERSE_PROOF = dict(owner_sha256=_F2_OWNER_SHA,
    derived_sha256=hashlib.sha256(_F2_DERIVED.encode()).hexdigest(),
    whole_source_inverse=True,metadata_versioning_only=True,
    numeric_body_and_tolerances_unchanged=True,no_candidate_or_runtime_changes=True)
exec(compile(_F2_DERIVED,str(_F2_OWNER)+':versioned-score-consumer','exec'),globals())
