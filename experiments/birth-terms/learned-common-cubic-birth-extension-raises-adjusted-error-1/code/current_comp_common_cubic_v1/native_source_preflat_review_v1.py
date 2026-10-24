"""Fixed common-cubic native acquisition; exact inherited paid-loop inverse."""
import hashlib
from pathlib import Path

_here = Path(__file__).resolve().parent
_parent = _here.parent/'current_comp_receiver_directions_v1/native.py'
_parent_sha = '0f107dcd4a6d27da4208a563f0b79c4dcfb966ab8423df3dad492c799edc4a9e'
if hashlib.sha256(_parent.read_bytes()).hexdigest() != _parent_sha:
    raise RuntimeError('Frozen native source owner changed')
_original = _parent.read_text()
_changes = (
    ('current_comp_receiver_directions_native_v1', 'current_comp_common_cubic_native_v1'),
    ("OWNERS = {\n", "OWNERS = {\n"
     "    NEW/'current_comp_receiver_directions_v1/native.py': '"+_parent_sha+"',\n"
     "    NEW/'current_comp_receiver_directions_v1/watch_native.py': '5637a5e24411aa74a989206709c9c00c695f279b4941792672522fa68de13947',\n"),
    ('_bdr_parameters', '_cc_parameters'),
    ('_BDR_PARAMETERS', '_CC_PARAMETERS'),
    ('_BDR_LITERAL', '_CC_LITERAL'),
    ('R_empty_at_entry', 'Q_empty_at_entry'),
    ('R_ready_at_binary_entry', 'Q_ready_at_binary_entry'),
    ('receiver_R', 'common_Q'),
    ("receiver[0]['literal_elements'] == 256", "receiver[0]['literal_elements'] == 65"),
    ('Sole actual256-value receiver getter', 'Sole actual65-value common-cubic getter'),
    ('(512 if cold else 0)', '(130 if cold else 0)'),
    ("HERE/'build_v2.py'", "HERE/'build.py'"),
    ("HERE/'BUILD_PROTOCOL_V2.md'", "HERE/'BUILD_PROTOCOL.md'"),
)
_derived = _original
_counts = []
for _before, _after in _changes:
    _count = _derived.count(_before)
    if not _count:
        raise RuntimeError('Missing common-cubic native source span '+_before)
    _counts.append(_count)
    _derived = _derived.replace(_before, _after)
_inverse = _derived
for (_before, _after), _count in reversed(list(zip(_changes, _counts))):
    if _inverse.count(_after) != _count:
        raise RuntimeError('Unexpected common-cubic native inverse span '+_after)
    _inverse = _inverse.replace(_after, _before)
if _inverse != _original:
    raise RuntimeError('Whole native paid-loop adapter source inverse failed')
exec(compile(_derived, str(_parent)+':common-cubic-native-adapter', 'exec'), globals())
