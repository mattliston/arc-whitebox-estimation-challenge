"""Fixed64-call receiver training supervisor; source-inverse, stdlib only."""
import hashlib
from pathlib import Path

_here = Path(__file__).resolve().parent
_parent = _here.parent/'current_comp_receiver_birth_v1/watch_qualification.py'
_old_sha = '4b5de5318afc57644ab81b5bdafebe566b939b6c7098622f1f848d9dbd58a1bb'
if hashlib.sha256(_parent.read_bytes()).hexdigest() != _old_sha:
    raise RuntimeError('Frozen qualified resource-supervisor source changed')
_original = _parent.read_text()
_extra = ('model-protocol', 'qualification-source', 'qualification-protocol',
          'qualification-watcher', 'qualification-watch-spec',
          'qualification-result', 'qualification-watch')
_changes = (
    ("HERE/'qualification.py'", "HERE/'train.py'"),
    ("HERE/'PROTOCOL.md'", "HERE/'TRAIN_PROTOCOL.md'"),
    ("HERE/'WATCH_SPEC.md'", "HERE/'TRAIN_WATCH_SPEC.md'"),
    ("HERE/'TRAIN_WATCH_SPEC.md':sha(HERE/'TRAIN_WATCH_SPEC.md')",
     "HERE/'TRAIN_WATCH_SPEC.md':'68acc32338ef68085c68993a557ee67c64704d36acc8133126f95639e4bb592b'"),
    ("HERE/'qualification_watch_v1'", "HERE/'train_watch_v1'"),
    ("HERE/'qualification_v1/result.json'", "HERE/'train_v1/result.json'"),
    ("    args=ap.parse_args()\n",
     "    for k in "+repr(_extra)+":ap.add_argument('--'+k+'-sha256')\n"
     "    args=ap.parse_args()\n"),
    ("MON:MON_SHA,P1W:P1W_SHA}",
     "MON:MON_SHA,P1W:P1W_SHA,\n"
     "          HERE/'MODEL_PROTOCOL.md':args.model_protocol_sha256,\n"
     "          HERE/'qualification.py':args.qualification_source_sha256,\n"
     "          HERE/'QUALIFICATION_PROTOCOL.md':args.qualification_protocol_sha256,\n"
     "          HERE/'watch_qualification.py':args.qualification_watcher_sha256,\n"
     "          HERE/'WATCH_SPEC.md':args.qualification_watch_spec_sha256,\n"
     "          HERE/'qualification_v1/result.json':args.qualification_result_sha256,\n"
     "          HERE/'qualification_watch_v1/result.json':args.qualification_watch_sha256,\n"
     "          HERE/'DECISION.md':'ad9b923d6c322585257013b293237a28b1d77826a86fb4a412a21a9bd9118b8c',\n"
     "          HERE.parent/'current_comp_receiver_birth_v1/watch_qualification.py':'"+_old_sha+"'}"),
    ("                  '--watcher-sha256',args.source_sha256]\n",
     "                  '--watcher-sha256',args.source_sha256]\n"
     "            for key in "+repr(_extra)+":\n"
     "                argv += ['--'+key+'-sha256',getattr(args,key.replace('-','_')+'_sha256')]\n"),
    ("check(result['retired_calls']==d['retired_call_count']==8\n"
     "                  and d['full_model_calls']==7 and d['tiny_model_calls']==8,'Complete fixed qualification roster')",
     "check(result['retired_calls']==d['retired_call_count']==64\n"
     "                  and d['full_model_calls']==64 and d['arm']=='receiver'\n"
     "                  and d['training_graphs']==60 and d['validation_graphs']==4\n"
     "                  and d['updates']==15 and d['trainable_keys']==['R']\n"
     "                  and d['frozen_P_A_cstar'] and d['frozen_nonP_and_inactive66_byte_exact']\n"
     "                  and d['all_graphs_retired'] and d['hashes_unchanged'] and d['sources_unchanged']\n"
     "                  and d['validation_targets_accessed'] is False\n"
     "                  and d['native_gradient_qualified'] is False\n"
     "                  and d['schema']=='current_comp_receiver_directions_training_v1',\n"
     "                  'Complete fixed receiver-training roster and frozen scope')\n"
     "            check(d['prerequisites']['qualification_result_sha256']==args.qualification_result_sha256\n"
     "                  and d['prerequisites']['qualification_watch_sha256']==args.qualification_watch_sha256,\n"
     "                  'Exact supplied qualification receipts')\n"
     "            for owner,digest in pins.items():\n"
     "                check(d['source_bindings'].get(str(owner),d['input_bindings'].get(str(owner)))==digest,\n"
     "                      'Worker authenticated the same reviewed dependency '+str(owner))"),
    ('180', '600'),
)
_derived = _original
_counts = []
for _old, _new in _changes:
    _count = _derived.count(_old)
    if not _count:
        raise RuntimeError('Missing exact receiver supervisor span '+_old)
    _counts.append(_count)
    _derived = _derived.replace(_old, _new)
_inverse = _derived
for (_old, _new), _count in reversed(list(zip(_changes, _counts))):
    if _inverse.count(_new) != _count:
        raise RuntimeError('Unexpected inverse receiver supervisor span '+_new)
    _inverse = _inverse.replace(_new, _old)
if _inverse != _original:
    raise RuntimeError('Byte-exact original resource-supervisor inverse failed')
exec(compile(_derived, str(_parent)+':receiver-training-v1', 'exec'),
     {'__name__': __name__, '__file__': __file__})
