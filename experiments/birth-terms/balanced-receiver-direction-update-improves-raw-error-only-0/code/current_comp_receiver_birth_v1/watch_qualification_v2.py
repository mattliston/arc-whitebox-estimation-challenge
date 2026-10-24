"""Source-inverse path-only version of the frozen qualification supervisor."""
import hashlib
from pathlib import Path

_here=Path(__file__).resolve().parent
_parent=_here/'watch_qualification.py'
_old_sha='4b5de5318afc57644ab81b5bdafebe566b939b6c7098622f1f848d9dbd58a1bb'
if hashlib.sha256(_parent.read_bytes()).hexdigest()!=_old_sha:
    raise RuntimeError('Frozen original supervisor changed')
_original=_parent.read_text()
_changes=(
    ("HERE/'qualification.py'", "HERE/'qualification_v2.py'", 2),
    ("HERE/'PROTOCOL.md'", "HERE/'PROTOCOL_V2.md'", 1),
    ("HERE/'WATCH_SPEC.md'", "HERE/'WATCH_SPEC_V2.md'", 2),
    ("HERE/'qualification_watch_v1'", "HERE/'qualification_watch_v2'", 1),
    ("HERE/'qualification_v1/result.json'", "HERE/'qualification_v2/result.json'", 1),
    ("MON:MON_SHA,P1W:P1W_SHA}",
     "MON:MON_SHA,P1W:P1W_SHA,HERE/'watch_qualification.py':'"+_old_sha+"',"
     "HERE/'WATCH_SPEC.md':'36254a34071682f7ec5480a55c8400048a451be9f4da5a23febffc19431e8213'}", 1),
)
_derived=_original
for _a,_b,_n in _changes:
    if _derived.count(_a)!=_n:raise RuntimeError('Unexpected original supervisor span '+_a)
    _derived=_derived.replace(_a,_b)
_inverse=_derived
for _a,_b,_n in reversed(_changes):
    if _inverse.count(_b)!=_n:raise RuntimeError('Unexpected inverse supervisor span')
    _inverse=_inverse.replace(_b,_a)
if _inverse!=_original:raise RuntimeError('Supervisor source inverse failed')
exec(compile(_derived,str(_parent)+':path-only-v2','exec'),
     {'__name__':__name__,'__file__':__file__})
