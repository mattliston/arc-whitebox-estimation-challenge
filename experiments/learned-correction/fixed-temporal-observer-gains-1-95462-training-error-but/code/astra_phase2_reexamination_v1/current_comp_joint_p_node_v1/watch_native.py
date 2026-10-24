"""One fixed native CHECK4 plus repeat, after retired full source admission."""
import hashlib
from pathlib import Path

_here=Path(__file__).resolve().parent
_parent=_here.parent/'current_comp_receiver_birth_v1/watch_qualification.py'
_old_sha='4b5de5318afc57644ab81b5bdafebe566b939b6c7098622f1f848d9dbd58a1bb'
if hashlib.sha256(_parent.read_bytes()).hexdigest()!=_old_sha:raise RuntimeError('Original supervisor changed')
_original=_parent.read_text()
_changes=(
    ("AP='/Users/matt/anaconda3/bin/python'", "AP=str(P2/'data_gate/.venv/bin/python')"),
    ("HERE/'qualification.py'", "HERE/'native.py'"),
    ("HERE/'PROTOCOL.md'", "HERE/'NATIVE_PROTOCOL.md'"),
    ("HERE/'WATCH_SPEC.md'", "HERE/'NATIVE_WATCH_SPEC.md'"),
    ("HERE/'qualification_watch_v1'", "HERE/('native_'+args.arm+'_watch_v1')"),
    ("HERE/'qualification_v1/result.json'", "HERE/('native_'+args.arm+'_v1/result.json')"),
    ("    args=ap.parse_args()\n",
     "    ap.add_argument('--arm',choices=('p_only','joint'))\n"
     "    ap.add_argument('--context-sha256')\n    args=ap.parse_args()\n"),
    ("    began=time.monotonic();check(args.mps_slot_clear,'Explicit MPS slot attestation')\n",
     "    began=time.monotonic();check(args.mps_slot_clear,'Explicit numerical slot attestation')\n"
     "    check(args.arm in ('p_only','joint'),'One fixed native arm')\n"),
    ("MON:MON_SHA,P1W:P1W_SHA}",
     "MON:MON_SHA,P1W:P1W_SHA,\n"
     "          HERE/('native_'+args.arm+'_admission_v1/context.json'):args.context_sha256,\n"
     "          HERE.parent/'current_comp_receiver_birth_v1/watch_qualification.py':'"+_old_sha+"'}"),
    ("                  '--watcher-sha256',args.source_sha256]\n",
     "                  '--watcher-sha256',args.source_sha256,'--arm',args.arm,\n"
     "                  '--context-sha256',args.context_sha256]\n"),
    ("check(result['retired_calls']==d['retired_call_count']==8\n"
     "                  and d['full_model_calls']==7 and d['tiny_model_calls']==8,'Complete fixed qualification roster')",
     "check(result['retired_calls']==d['retired_call_count']==5 and d['full_model_calls']==5\n"
     "                  and d['arm']==args.arm and d['ranks']==[37,39,42,48]\n"
     "                  and d['own_complete_bills'] and d['all16_domains_qualified']\n"
     "                  and d['all47_selectors_qualified'] and d['initialization_ownership_passed']\n"
     "                  and d['repeat_output_domain_byte_exact'],'Complete fixed native5 with owned bills')"),
    ('12_000_000_000','8_000_000_000'),('12GB combined RSS','8GB combined RSS'),('180','600'),
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
exec(compile(_derived,str(_parent)+':joint-native-check4-v1','exec'),
     {'__name__':__name__,'__file__':__file__})
