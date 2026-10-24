"""CPU-only endpoint-teacher supervisor, inverse-derived from reviewed owner.

Named watch.py to match the frozen teacher's explicit watcher ownership.
"""
import hashlib
from pathlib import Path

_here=Path(__file__).resolve().parent
_parent=_here.parent/'current_comp_receiver_birth_v1/watch_qualification.py'
_old_sha='4b5de5318afc57644ab81b5bdafebe566b939b6c7098622f1f848d9dbd58a1bb'
if hashlib.sha256(_parent.read_bytes()).hexdigest()!=_old_sha:
    raise RuntimeError('Frozen original supervisor changed')
_original=_parent.read_text()
_changes=(
    ('Bounded one-MPS qualification supervisor.', 'Bounded CPU-only endpoint teacher supervisor.'),
    ("'--mps-slot-clear'", "'--numeric-slot-clear'"),
    ("('source','worker','model','protocol')", "('source','worker','protocol')"),
    ("args.mps_slot_clear,'Explicit MPS slot attestation'",
     "args.numeric_slot_clear,'Explicit exclusive numerical slot attestation'"),
    ("HERE/'qualification.py'", "HERE/'teacher.py'"),
    ("HERE/'model.py':args.model_sha256,HERE/'PROTOCOL.md':args.protocol_sha256,",
     "HERE/'DESIGN.md':'461713beeb82d47299d2f0c02e78bed2598d6f675eaee38e1a74062d09d3c839',HERE/'PROTOCOL.md':args.protocol_sha256,"),
    ("MON:MON_SHA,P1W:P1W_SHA}",
     "MON:MON_SHA,P1W:P1W_SHA,NEW/'current_comp_receiver_birth_v1/watch_qualification.py':'"+_old_sha+"'}"),
    ("HERE/'qualification_watch_v1'", "HERE/'information_watch_v1'"),
    ("mps_slot_clear_attested=True", "exclusive_numeric_slot_attested=True,cpu_only=True"),
    ("'--model-sha256',args.model_sha256,'--protocol-sha256',args.protocol_sha256,",
     "'--protocol-sha256',args.protocol_sha256,"),
    ("HERE/'qualification_v1/result.json'", "HERE/'information_v1/result.json'"),
    ("check(result['retired_calls']==d['retired_call_count']==8\n"
     "                  and d['full_model_calls']==7 and d['tiny_model_calls']==8,'Complete fixed qualification roster')",
     "check(result['retired_calls']==d['retired_call_count']==12\n"
     "                  and d['reference_calls']==4 and d['suffix_vjp_calls']==8,'Complete fixed teacher roster')"),
    ('180', '900'),
)
_derived=_original
_counts=[]
for _a,_b in _changes:
    _n=_derived.count(_a)
    if not _n:raise RuntimeError('Missing reviewed supervisor span '+_a)
    _counts.append(_n)
    _derived=_derived.replace(_a,_b)
_inverse=_derived
for (_a,_b),_n in reversed(list(zip(_changes,_counts))):
    if _inverse.count(_b)!=_n:raise RuntimeError('Unexpected inverse supervisor span '+_b)
    _inverse=_inverse.replace(_b,_a)
if _inverse!=_original:raise RuntimeError('Supervisor source inverse failed')
exec(compile(_derived,str(_parent)+':cpu-teacher-v1','exec'),
     {'__name__':__name__,'__file__':__file__})
