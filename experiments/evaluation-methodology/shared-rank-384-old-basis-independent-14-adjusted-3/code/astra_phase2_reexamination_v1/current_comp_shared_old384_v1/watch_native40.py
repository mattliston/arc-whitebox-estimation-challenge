"""Conditional fixed41 native supervision; same owned-tree monitor and cleanup."""
import ast
import hashlib
from pathlib import Path
import runpy

_N40_WATCH_HERE = Path(__file__).resolve().parent
_N40_WATCH_OWNER = _N40_WATCH_HERE/'watch_check4_v2.py'
_N40_WATCH_OWNER_SHA = '982001c397860137a566bca4c6632611e5050724ff09bf97134781948668a194'
if hashlib.sha256(_N40_WATCH_OWNER.read_bytes()).hexdigest() != _N40_WATCH_OWNER_SHA:
    raise RuntimeError('Frozen same-candidate CHECK4 supervisor')
_N40_WATCH_PARENT = runpy.run_path(str(_N40_WATCH_OWNER),run_name='inert_native40_watch_parent')
for _n40_key,_n40_value in _N40_WATCH_PARENT.items():
    if not _n40_key.startswith('__') and not _n40_key.startswith('_N40'):
        globals()[_n40_key] = _n40_value
SEQUENCE = (37,39,42,48,59,63,66,67,71,73,75,79,93,101,122,130,132,135,139,145,
    146,149,150,151,155,159,163,169,173,175,181,192,198,199,204,219,224,227,233,234,37)
GATE_KEYS = ('check4-native','check4-native-watch','check4-score','check4-watch')
_N40_WATCH_ORIGINAL = _N40_WATCH_PARENT['_derived']
_n40_gate_start = _N40_WATCH_ORIGINAL.index(
    "        require(record['schema'] == 'current_comp_shared_old384_native_check4_v2'")
_n40_gate_end = _N40_WATCH_ORIGINAL.index("        for group in ('source_bindings','input_bindings','artifacts'):",_n40_gate_start)
_N40_OLD_GATE = _N40_WATCH_ORIGINAL[_n40_gate_start:_n40_gate_end]
_N40_NEW_GATE = """        require(record['schema'] == 'current_comp_shared_old384_native40_v1'
            and len(record['trials']) == len(record['input_preparation']) == len(record['diagnostic_records']) == 41
            and len(record['artifacts']) == 287 and record['initialization_flops'] == 1944970298
            and record['check4_prerequisite_seals'] == result['check4_prerequisite_seals']
            and record['native40_source_inverse_proof'] == probe.NATIVE40_SOURCE_INVERSE_PROOF
            and all(record.get(k) is True for k in ('check4_prerequisite_qualified','check4_first4_exact',
                'check4_repeat37_physics_exact','operation_ledgers_verified','operation_counts_reported_not_rebased',
                'repeat_operation_counts_history_aware')) and probe._v2_validate_ledgers(record),
            'Actual fixed41 acquisition, original paid arithmetic, direct positive CHECK4 and complete own ledgers')
        prior = probe.read(HERE/'check4_v2/result.json')
        require(sha(HERE/'check4_v2/result.json',tick) == args.check4_native_sha256
            and record['check4_parity_checks'] == probe._n40_parity(record,prior,tick=tick),
            'Independent first4 exact replay and final37 history-aware physical replay')
        del prior
        result.update(check4_first4_exact=True,check4_repeat37_physics_exact=True,
            operation_ledgers_verified=True,operation_counts_reported_not_rebased=True,
            repeat_operation_counts_history_aware=True)
"""
_N40_WATCH_EDITS = [
    ("CAP, RSS, PROGRESS = 660.,8_000_000_000,120.","CAP, RSS, PROGRESS = 3000.,8_000_000_000,120.",1),
    ("KEYS = ('source','worker','protocol','spec','compose','fnp','design')",
     "KEYS = ('source','worker','protocol','spec','compose','fnp','design')+GATE_KEYS",1),
    ("HERE/'watch_check4_v2.py'","HERE/'watch_native40.py'",1),
    ("HERE/'check4_v2.py'","HERE/'native40.py'",3),
    ("HERE/'CHECK4_V2_PROTOCOL.md'","HERE/'NATIVE40_PROTOCOL.md'",1),
    ("HERE/'CHECK4_V2_WATCH_SPEC.md'","HERE/'NATIVE40_WATCH_SPEC.md'",1),
    ("_V2_WATCH_OWNER:_V2_WATCH_OWNER_SHA}",
     "_V2_WATCH_OWNER:_V2_WATCH_OWNER_SHA,_N40_WATCH_OWNER:_N40_WATCH_OWNER_SHA}",1),
    ("HERE/'check4_v2',HERE/'check4_watch_v2'","HERE/'native40_v1',HERE/'native40_watch_v1'",1),
    ("current_comp_shared_old384_native_check4_watch_v2","current_comp_shared_old384_native40_watch_v1",1),
    ("check4_watch_source_inverse_proof=SOURCE_INVERSE_PROOF",
     "native40_watch_source_inverse_proof=NATIVE40_WATCH_SOURCE_INVERSE_PROOF",1),
    ("'Whole660s/next retired call120s'","'Whole3000s/next retired call120s'",1),
    ("index < 5 and item['call'] == index","index < 41 and item['call'] == index",1),
    ("        prior,prerequisite_seals = probe.prerequisites(result['input_bindings'],tick)\n"
     "        failed,failed_seals = probe._v2_failed_attempt(result['input_bindings'],tick)\n"
     "        result.update(prior_failed_attempt_preserved=True,failed_attempt_seals=failed_seals,\n"
     "            v2_watch_source_inverse_proof=V2_WATCH_SOURCE_INVERSE_PROOF)\n        del failed\n"
     "        result.update(probe_prerequisite_qualified=True,prerequisite_seals=prerequisite_seals)",
     "        prior,check4_seals = probe._n40_prerequisites(args,result['input_bindings'],tick)\n"
     "        result.update(check4_prerequisite_qualified=True,check4_prerequisite_seals=check4_seals)",1),
    ("'--compose-sha256',args.compose_sha256,'--fnp-sha256',args.fnp_sha256,'--design-sha256',args.design_sha256]",
     "'--compose-sha256',args.compose_sha256,'--fnp-sha256',args.fnp_sha256,'--design-sha256',args.design_sha256]\n"
     "        for key in GATE_KEYS:\n"
     "            argv.extend(['--'+key+'-sha256',getattr(args,key.replace('-','_')+'_sha256')])",1),
    ("child.wait(timeout=1) == 0 and result['retired_calls'] == 5,'Five complete calls and exit0'",
     "child.wait(timeout=1) == 0 and result['retired_calls'] == 41,'All41 complete calls and exit0'",1),
    ("record['full_model_calls'] == record['retired_call_count'] == 5",
     "record['full_model_calls'] == record['retired_call_count'] == 41",1),
    ("for i in range(5)]","for i in range(41)]",1),
    (_N40_OLD_GATE,_N40_NEW_GATE,1),
]
_N40_WATCH_DERIVED = _N40_WATCH_ORIGINAL
for _n40_before,_n40_after,_n40_count in _N40_WATCH_EDITS:
    if _N40_WATCH_DERIVED.count(_n40_before) != _n40_count:
        raise RuntimeError('Unique native40-watch metadata span '+_n40_before[:100])
    _N40_WATCH_DERIVED = _N40_WATCH_DERIVED.replace(_n40_before,_n40_after)
_N40_WATCH_INVERSE = _N40_WATCH_DERIVED
for _n40_before,_n40_after,_n40_count in reversed(_N40_WATCH_EDITS):
    if _N40_WATCH_INVERSE.count(_n40_after) != _n40_count:
        raise RuntimeError('Unique native40-watch inverse span '+_n40_after[:100])
    _N40_WATCH_INVERSE = _N40_WATCH_INVERSE.replace(_n40_after,_n40_before)
if _N40_WATCH_INVERSE != _N40_WATCH_ORIGINAL:
    raise RuntimeError('Entire frozen CHECK4 v2 supervisor restored byte-exact')
ast.parse(_N40_WATCH_DERIVED)
NATIVE40_WATCH_SOURCE_INVERSE_PROOF = dict(owner=str(_N40_WATCH_OWNER),owner_sha256=_N40_WATCH_OWNER_SHA,
    whole_source_inverse_verified=True,derived_sha256=hashlib.sha256(_N40_WATCH_DERIVED.encode()).hexdigest(),
    original_sampling_resource_and_cleanup_paths_unchanged=True,whole_cap_s=3000.,
    fixed_roster=list(SEQUENCE),positive_CHECK4_objective_required=True,
    repeat_operation_count_history_aware=True)
exec(compile(_N40_WATCH_DERIVED,str(_N40_WATCH_OWNER)+':conditional-native40-metadata','exec'),globals())
