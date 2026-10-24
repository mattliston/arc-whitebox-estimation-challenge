"""Five-call common-cubic qualification, inverse-owned original resource guard."""
import hashlib
from pathlib import Path

_here = Path(__file__).resolve().parent
_parent = _here.parent/'current_comp_receiver_birth_v1/watch_qualification.py'
_old_sha = '4b5de5318afc57644ab81b5bdafebe566b939b6c7098622f1f848d9dbd58a1bb'
if hashlib.sha256(_parent.read_bytes()).hexdigest() != _old_sha:
    raise RuntimeError('Frozen qualification supervisor changed')
_original = _parent.read_text()
_changes = (
    ("HERE/'PROTOCOL.md'", "HERE/'QUALIFICATION_PROTOCOL.md'"),
    ("MON:MON_SHA,P1W:P1W_SHA}",
     "MON:MON_SHA,P1W:P1W_SHA,"
     "HERE/'MODEL_PROTOCOL.md':sha(HERE/'MODEL_PROTOCOL.md'),"
     "HERE/'DECISION.md':'d90cec782ef7631acc227b1bb857f92d71284920521341764c3d182b87368d80',"
     "HERE.parent/'current_comp_receiver_birth_v1/watch_qualification.py':'"+_old_sha+"'}"),
    ("check(result['retired_calls']==d['retired_call_count']==8\n"
     "                  and d['full_model_calls']==7 and d['tiny_model_calls']==8,'Complete fixed qualification roster')",
     "check(result['retired_calls']==d['retired_call_count']==5\n"
     "                  and d['full_model_calls']==5 and d['null_forwards']==2\n"
     "                  and d['common_backwards']==3 and d['execution_integrity_passed']\n"
     "                  and d['schema']=='current_comp_common_cubic_qualification_v1'\n"
     "                  and d['native_gradient_qualified'] is False\n"
     "                  and d['training_targets_accessed'] is False\n"
     "                  and d['validation_targets_accessed'] is False,\n"
     "                  'Complete common-cubic5 target-free lifecycle roster')\n"
     "            for owner,digest in pins.items():\n"
     "                check(d['source_bindings'].get(str(owner))==digest,\n"
     "                      'Worker authenticated the same reviewed owner '+str(owner))"),
)
_derived = _original
_counts = []
for _before, _after in _changes:
    _count = _derived.count(_before)
    if not _count:
        raise RuntimeError('Missing common-cubic supervisor source span '+_before)
    _counts.append(_count); _derived = _derived.replace(_before, _after)
_inverse = _derived
for (_before, _after), _count in reversed(list(zip(_changes, _counts))):
    if _inverse.count(_after) != _count:
        raise RuntimeError('Unexpected common-cubic supervisor inverse span '+_after)
    _inverse = _inverse.replace(_after, _before)
if _inverse != _original:
    raise RuntimeError('Whole original resource-supervisor inverse failed')
exec(compile(_derived, str(_parent)+':common-cubic-five', 'exec'),
     {'__name__': __name__, '__file__': __file__})
