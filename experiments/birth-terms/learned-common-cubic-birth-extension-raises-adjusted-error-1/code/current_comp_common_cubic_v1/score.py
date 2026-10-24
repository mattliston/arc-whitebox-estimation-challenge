"""One common-cubic CHECK4: source-inverse saved-output scorer, no model calls."""
import hashlib
from pathlib import Path

_here = Path(__file__).resolve().parent
_parent = _here.parent/'current_comp_receiver_directions_v1/score.py'
_parent_sha = 'e6697bb3f8349078ec078f342178ad5c36136d2300c2ec41af085155efc9e42b'
if hashlib.sha256(_parent.read_bytes()).hexdigest() != _parent_sha:
    raise RuntimeError('Frozen complete native CHECK4 scorer changed')
_original = _parent.read_text()
_changes = (
    ('current_comp_receiver_directions_build_v1', 'current_comp_common_cubic_build_v1'),
    ('current_comp_receiver_directions_native_v1', 'current_comp_common_cubic_native_v1'),
    ('current_comp_receiver_directions_score_v1', 'current_comp_common_cubic_score_v1'),
    ('773f6fa146ff6944fa43b125d98f6f643c71bbe2440ef9d3e669a227397afd0e',
     'f068071bca67d052da101b47952eff15216d6e408a485148d2f250829ee8d37d'),
    ('22e15a727e3e71eb11a7d289ef38a6b121328c52c127a2d87968e341074eae02',
     '6eff6e23216dbc2f5df30c368cb0a3d39ead2f2dc736c13ef2a1567b61d0ccba'),
    ('ad9b923d6c322585257013b293237a28b1d77826a86fb4a412a21a9bd9118b8c',
     'd90cec782ef7631acc227b1bb857f92d71284920521341764c3d182b87368d80'),
    ('7c989ea707c9790d92793084569378983fad0861412d233153433e70ad012531',
     '161247bc17ae861309b6504f4d9cf9636786a25d569190442fa8e76f9faf2eb3'),
    ("    HERE/'MODEL_PROTOCOL.md':", "    NEW/'current_comp_receiver_directions_v1/score.py': '"+_parent_sha+"',\n"
     "    NEW/'current_comp_receiver_directions_v1/watch_score.py': '873947e1022f4b9d689527d35d4e9f47eac5c2f2f4731566141597b7798596ac',\n"
     "    HERE/'MODEL_PROTOCOL.md':"),
    ("here/'build_v2.py'", "here/'build.py'"),
    ("here/'BUILD_PROTOCOL_V2.md'", "here/'BUILD_PROTOCOL.md'"),
    ('''        ("HERE/'build.py'", "HERE/'build_v2.py'", 3),''',
     '        # Common-cubic uses the original HERE build.py key unchanged.'),
    ('''        ("HERE/'BUILD_PROTOCOL.md'", "HERE/'BUILD_PROTOCOL_V2.md'", 1),''',
     '        # Common-cubic uses the original HERE BUILD_PROTOCOL.md key unchanged.'),
    ('receiver_literal', 'common_literal'),
    ("build['common_literal']['shape'] == [64, 4]", "build['common_literal']['shape'] == [65]"),
    ('bytes(1024)', 'bytes(260)'),
    ('_bdr_parameters', '_cc_parameters'),
    ('receiver_R', 'common_Q'),
    ('R_empty_at_entry', 'Q_empty_at_entry'),
    ('R_ready_at_binary_entry', 'Q_ready_at_binary_entry'),
    ("receiver[0]['literal_elements'] == 256", "receiver[0]['literal_elements'] == 65"),
    ('(512 if cold else 0)', '(130 if cold else 0)'),
    ('R512cold/0warm', 'Q130cold/0warm'),
    ('receiver_main_lead', 'common_main_lead'),
    ('receiver_component', 'common_component'),
)
_derived = _original
_counts = []
for _before, _after in _changes:
    _count = _derived.count(_before)
    if not _count:
        raise RuntimeError('Missing common-cubic score source span '+_before)
    _counts.append(_count)
    _derived = _derived.replace(_before, _after)
_inverse = _derived
for (_before, _after), _count in reversed(list(zip(_changes, _counts))):
    if _inverse.count(_after) != _count:
        raise RuntimeError('Unexpected common-cubic score inverse span '+_after)
    _inverse = _inverse.replace(_after, _before)
if _inverse != _original:
    raise RuntimeError('Whole original scorer source inverse failed')
exec(compile(_derived, str(_parent)+':common-cubic-score', 'exec'), globals())
