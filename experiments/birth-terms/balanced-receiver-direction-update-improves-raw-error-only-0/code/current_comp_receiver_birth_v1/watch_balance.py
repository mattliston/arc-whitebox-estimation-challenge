"""Bounded actual-output common-direction supervisor; no numerical imports."""
import hashlib
from pathlib import Path

_here=Path(__file__).resolve().parent
_parent=_here/'watch_qualification.py'
_old_sha='4b5de5318afc57644ab81b5bdafebe566b939b6c7098622f1f848d9dbd58a1bb'
if hashlib.sha256(_parent.read_bytes()).hexdigest()!=_old_sha:
    raise RuntimeError('Frozen original supervisor changed')
_original=_parent.read_text()
_changes=(
    ("HERE/'qualification.py'", "HERE/'balance.py'"),
    ("HERE/'PROTOCOL.md'", "HERE/'BALANCE_PROTOCOL.md'"),
    ("HERE/'WATCH_SPEC.md'", "HERE/'BALANCE_WATCH_SPEC.md'"),
    ("HERE/'qualification_watch_v1'", "HERE/'balance_watch_v1'"),
    ("HERE/'qualification_v1/result.json'", "HERE/'balance_v1/result.json'"),
    ("MON:MON_SHA,P1W:P1W_SHA}",
     "MON:MON_SHA,P1W:P1W_SHA,HERE/'watch_qualification.py':'"+_old_sha+"'}"),
    ("check(result['retired_calls']==d['retired_call_count']==8\n"
     "                  and d['full_model_calls']==7 and d['tiny_model_calls']==8,'Complete fixed qualification roster')",
     "check(result['retired_calls']==d['retired_call_count']==16\n"
     "                  and d['full_model_calls']==16,'Complete fixed sixteen actual-output calls')"),
)
_derived=_original
_counts=[]
for _a,_b in _changes:
    _n=_derived.count(_a)
    if not _n:raise RuntimeError('Missing reviewed supervisor span '+_a)
    _counts.append(_n);_derived=_derived.replace(_a,_b)
_inverse=_derived
for (_a,_b),_n in reversed(list(zip(_changes,_counts))):
    if _inverse.count(_b)!=_n:raise RuntimeError('Unexpected inverse supervisor span '+_b)
    _inverse=_inverse.replace(_b,_a)
if _inverse!=_original:raise RuntimeError('Supervisor source inverse failed')
exec(compile(_derived,str(_parent)+':actual-output-common-direction-v1','exec'),
     {'__name__':__name__,'__file__':__file__})
