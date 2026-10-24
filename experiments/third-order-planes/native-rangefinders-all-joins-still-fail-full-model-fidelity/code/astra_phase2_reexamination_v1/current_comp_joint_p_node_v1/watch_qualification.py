"""Thin five-call joint-lifecycle supervisor; no numerical imports."""
import hashlib
from pathlib import Path

_here=Path(__file__).resolve().parent
_parent=_here.parent/'current_comp_receiver_birth_v1/watch_qualification.py'
_old_sha='4b5de5318afc57644ab81b5bdafebe566b939b6c7098622f1f848d9dbd58a1bb'
if hashlib.sha256(_parent.read_bytes()).hexdigest()!=_old_sha:
    raise RuntimeError('Frozen original supervisor changed')
_original=_parent.read_text()
_changes=(
    ("HERE/'PROTOCOL.md'", "HERE/'QUALIFICATION_PROTOCOL.md'"),
    ("MON:MON_SHA,P1W:P1W_SHA}",
     "MON:MON_SHA,P1W:P1W_SHA,HERE.parent/'current_comp_receiver_birth_v1/watch_qualification.py':'"+_old_sha+"'}"),
    ("check(result['retired_calls']==d['retired_call_count']==8\n"
     "                  and d['full_model_calls']==7 and d['tiny_model_calls']==8,'Complete fixed qualification roster')",
     "check(result['retired_calls']==d['retired_call_count']==5\n"
     "                  and d['full_model_calls']==5 and d['execution_integrity_passed']\n"
     "                  and not d['native_gradient_qualified'],'Complete fixed joint5 lifecycle roster')"),
)
_derived=_original;_counts=[]
for _a,_b in _changes:
    _n=_derived.count(_a)
    if not _n:raise RuntimeError('Missing reviewed supervisor span '+_a)
    _counts.append(_n);_derived=_derived.replace(_a,_b)
_inverse=_derived
for (_a,_b),_n in reversed(list(zip(_changes,_counts))):
    if _inverse.count(_b)!=_n:raise RuntimeError('Unexpected inverse supervisor span '+_b)
    _inverse=_inverse.replace(_b,_a)
if _inverse!=_original:raise RuntimeError('Supervisor source inverse failed')
exec(compile(_derived,str(_parent)+':joint-lifecycle-v1','exec'),
     {'__name__':__name__,'__file__':__file__})
