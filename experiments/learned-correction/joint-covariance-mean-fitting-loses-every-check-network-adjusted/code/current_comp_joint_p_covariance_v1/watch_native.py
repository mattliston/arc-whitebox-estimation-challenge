"""One serial covariance native role; unchanged external resource supervisor."""
import hashlib
from pathlib import Path

_here = Path(__file__).resolve().parent
_parent = _here.parent/'current_comp_receiver_directions_v1/watch_native.py'
_parent_sha = '5637a5e24411aa74a989206709c9c00c695f279b4941792672522fa68de13947'
if hashlib.sha256(_parent.read_bytes()).hexdigest() != _parent_sha:
    raise RuntimeError('Frozen completed native-watch owner changed')
_original = _parent.read_text()
_binding_before = '     "          HERE.parent/\'current_comp_joint_p_node_v1/watch_native.py\':\'"+_parent_sha+"\',\\\\n"),'
_binding_after = '     "          HERE.parent/\'current_comp_joint_p_node_v1/watch_native.py\':\'"+_parent_sha+"\',\\\\n"\n' + \
    '     "          HERE.parent/\'current_comp_receiver_directions_v1/watch_native.py\':\''+_parent_sha+'\',\\\\n"),'
_changes = (
    ('receiver_R_initialization_owned', 'covariance_Q_initialization_owned'),
    (_binding_before, _binding_after),
)
_derived = _original
_counts = []
for _before, _after in _changes:
    _count = _derived.count(_before)
    if not _count:
        raise RuntimeError('Missing covariance native supervisor span '+_before)
    _counts.append(_count)
    _derived = _derived.replace(_before, _after)
_inverse = _derived
for (_before, _after), _count in reversed(list(zip(_changes, _counts))):
    if _inverse.count(_after) != _count:
        raise RuntimeError('Unexpected covariance supervisor inverse span '+_after)
    _inverse = _inverse.replace(_after, _before)
if _inverse != _original:
    raise RuntimeError('Whole native supervisor source inverse failed')
exec(compile(_derived, str(_parent)+':covariance-owned-role', 'exec'),
     {'__name__': __name__, '__file__': __file__})
