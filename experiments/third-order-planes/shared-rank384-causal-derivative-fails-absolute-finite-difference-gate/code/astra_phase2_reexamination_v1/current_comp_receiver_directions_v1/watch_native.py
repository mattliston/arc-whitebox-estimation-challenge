"""Serial receiver native zero3 or terminal5, with the original resource guard."""
import hashlib
from pathlib import Path

_here = Path(__file__).resolve().parent
_parent = _here.parent/'current_comp_joint_p_node_v1/watch_native.py'
_parent_sha = 'd35ba7cbe406ae5e6200f8778b7a582bc6ea17336be8086b5e813e5a25dbe65e'
if hashlib.sha256(_parent.read_bytes()).hexdigest() != _parent_sha:
    raise RuntimeError('Frozen native supervisor owner changed')
_original = _parent.read_text()
_changes = (
    ("('p_only','joint')", "('zero','terminal')"),
    ('args.arm', 'args.role'),
    ("'--arm'", "'--role'"),
    ("d['arm']", "d['role']"),
    ("d['retired_call_count']==5 and d['full_model_calls']==5",
     "d['retired_call_count']==(3 if args.role=='zero' else 5) and d['full_model_calls']==(3 if args.role=='zero' else 5)"),
    ("d['ranks']==[37,39,42,48]",
     "d['ranks']==([35,100] if args.role=='zero' else [37,39,42,48])"),
    ("and d['repeat_output_domain_byte_exact']",
     "and d['repeat_output_domain_byte_exact'] and d['receiver_R_initialization_owned']"),
    ("MON:MON_SHA,P1W:P1W_SHA,\\n",
     "MON:MON_SHA,P1W:P1W_SHA,\\n"
     "          HERE.parent/'current_comp_joint_p_node_v1/watch_native.py':'"+_parent_sha+"',\\n"),
)
_derived = _original
_counts = []
for _before, _after in _changes:
    _count = _derived.count(_before)
    if not _count:
        raise RuntimeError('Missing receiver native-watch source span '+_before)
    _counts.append(_count)
    _derived = _derived.replace(_before, _after)
_inverse = _derived
for (_before, _after), _count in reversed(list(zip(_changes, _counts))):
    if _inverse.count(_after) != _count:
        raise RuntimeError('Unexpected native-watch inverse span '+_after)
    _inverse = _inverse.replace(_after, _before)
if _inverse != _original:
    raise RuntimeError('Byte-exact native supervisor inverse failed')
exec(compile(_derived, str(_parent)+':receiver-role-adapter', 'exec'),
     {'__name__': __name__, '__file__': __file__})
