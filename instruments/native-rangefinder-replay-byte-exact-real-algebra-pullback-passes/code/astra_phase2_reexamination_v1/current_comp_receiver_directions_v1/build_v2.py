"""Tuple-container admission repair of the preserved receiver builder."""
import hashlib
from pathlib import Path

_here = Path(__file__).resolve().parent
_parent = _here/'build.py'
_parent_sha = 'c844c0526e1d98d7138a8958c9a9501e9a794274222294cbc6e30dbf4073aa39'
if hashlib.sha256(_parent.read_bytes()).hexdigest() != _parent_sha:
    raise RuntimeError('Preserved first builder source changed')
_original = _parent.read_text()
_changes = (
    ('isinstance(item, list)', 'isinstance(item, (list, tuple))', 1),
    ("HERE/'BUILD_PROTOCOL.md'", "HERE/'BUILD_PROTOCOL_V2.md'", 3),
    ('PINS = {\n',
     'PINS = {\n'
     "    HERE/'build.py': '"+_parent_sha+"',\n"
     "    HERE/'BUILD_PROTOCOL.md': '767e8f39000facef13a17da5636c8750229e9bdda1f4550ed8581471ad605ca5',\n", 1),
)
_derived = _original
for _before, _after, _count in _changes:
    if _derived.count(_before) != _count:
        raise RuntimeError('Unexpected tuple-repair source span '+_before)
    _derived = _derived.replace(_before, _after)
_inverse = _derived
for _before, _after, _count in reversed(_changes):
    if _inverse.count(_after) != _count:
        raise RuntimeError('Unexpected tuple-repair inverse span '+_after)
    _inverse = _inverse.replace(_after, _before)
if _inverse != _original:
    raise RuntimeError('Original builder was not restored byte-exactly')
exec(compile(_derived, str(_parent)+':tuple-container-v2', 'exec'), globals())
