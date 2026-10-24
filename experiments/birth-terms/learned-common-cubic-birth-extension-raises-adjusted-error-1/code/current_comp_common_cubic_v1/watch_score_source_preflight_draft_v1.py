"""Fixed common-cubic score supervision with unchanged120s/1GB limits."""
import hashlib
from pathlib import Path

_here = Path(__file__).resolve().parent
_parent = _here.parent/'current_comp_receiver_directions_v1/watch_score.py'
_parent_sha = '873947e1022f4b9d689527d35d4e9f47eac5c2f2f4731566141597b7798596ac'
if hashlib.sha256(_parent.read_bytes()).hexdigest() != _parent_sha:
    raise RuntimeError('Frozen offline-score supervisor changed')
_original = _parent.read_text()
_changes = (
    ('build_v2.py', 'build.py'),
    ('BUILD_PROTOCOL_V2.md', 'BUILD_PROTOCOL.md'),
    ('current_comp_receiver_directions_score_v1', 'current_comp_common_cubic_score_v1'),
    ('current_comp_receiver_directions_score_watch_v1', 'current_comp_common_cubic_score_watch_v1'),
)
_derived = _original
_counts = []
for _before, _after in _changes:
    _count = _derived.count(_before)
    if not _count:
        raise RuntimeError('Missing common-cubic score-watch source span '+_before)
    _counts.append(_count)
    _derived = _derived.replace(_before, _after)
_inverse = _derived
for (_before, _after), _count in reversed(list(zip(_changes, _counts))):
    if _inverse.count(_after) != _count:
        raise RuntimeError('Unexpected common-cubic score-watch inverse span '+_after)
    _inverse = _inverse.replace(_after, _before)
if _inverse != _original:
    raise RuntimeError('Whole score-supervisor source inverse failed')
exec(compile(_derived, str(_parent)+':common-cubic-score-watch', 'exec'),
     {'__name__': __name__, '__file__': __file__})
