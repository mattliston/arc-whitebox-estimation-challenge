"""Fresh CHECK4 v2 supervisor; the original660s/8GB transport is unchanged."""
import ast
import hashlib
from pathlib import Path

_V2_WATCH_OWNER = Path(__file__).resolve().parent/'watch_check4.py'
_V2_WATCH_OWNER_SHA = 'd7f99ac0b97563a064784acf88624a1ccdaa853b7f38bfe0472dda0ac6a6586a'
_V2_WATCH_ORIGINAL = _V2_WATCH_OWNER.read_text()
if hashlib.sha256(_V2_WATCH_ORIGINAL.encode()).hexdigest() != _V2_WATCH_OWNER_SHA:
    raise RuntimeError('Frozen negative CHECK4 supervisor preserved')

_V2_WATCH_MAIN_EDITS = [
    ("        prior,prerequisite_seals = probe.prerequisites(result['input_bindings'],tick)",
     "        prior,prerequisite_seals = probe.prerequisites(result['input_bindings'],tick)\n"
     "        failed,failed_seals = probe._v2_failed_attempt(result['input_bindings'],tick)\n"
     "        result.update(prior_failed_attempt_preserved=True,failed_attempt_seals=failed_seals,\n"
     "            v2_watch_source_inverse_proof=V2_WATCH_SOURCE_INVERSE_PROOF)\n        del failed"),
    ("            and record['probe37_parity_checks'] == [dict(call=c,probe_call=p,rank=37,\n"
     "                output_byte_exact=True,complete_flops_exact=True,domains_exact=True,dispatches_exact=True)\n"
     "                for c,p in ((0,0),(4,1))]",
     "            and [dict(call=x['call'],probe_call=x['probe_call'],rank=x['rank'],\n"
     "                output_byte_exact=x['output_byte_exact'],complete_flops_exact=x['complete_flops_exact'],\n"
     "                domains_exact=x['domains_exact'],dispatches_exact=x['dispatches_exact'])\n"
     "                for x in record['probe37_parity_checks']] == [dict(call=c,probe_call=p,rank=37,\n"
     "                output_byte_exact=True,complete_flops_exact=True,domains_exact=True,dispatches_exact=True)\n"
     "                for c,p in ((0,0),(4,1))]\n"
     "            and all(x['operation_count_history'] == dict(prior=old,\n"
     "                current=record['trials'][c]['operation_count'],\n"
     "                delta=record['trials'][c]['operation_count']-old,prior_sequence=[37,37],\n"
     "                current_sequence=list(SEQUENCE),equality_required=False)\n"
     "                for x,c,old in zip(record['probe37_parity_checks'],(0,4),(60085,59803)))"),
    ("        result.update(probe37_output_byte_exact=True,probe37_complete_flops_exact=True,\n"
     "            probe37_domains_exact=True,probe37_dispatches_exact=True)",
     "        require(record['failed_attempt_seals'] == result['failed_attempt_seals']\n"
     "            and record['v2_source_inverse_proof'] == probe.V2_SOURCE_INVERSE_PROOF\n"
     "            and record['complete_bill_equality_fields'] == ['flops','multiplier','includes_initialization']\n"
     "            and all(record.get(k) is True for k in ('prior_failed_attempt_preserved',\n"
     "                'all5_failed_attempt_outputs_flops_domains_dispatches_exact','operation_ledgers_verified',\n"
     "                'operation_counts_reported_not_rebased')) and probe._v2_validate_ledgers(record),\n"
     "            'Fresh actual full bills; negative-v1 never becomes release authority')\n"
     "        failed = probe.read(HERE/'check4_v1/result.json')\n"
     "        require(record['failed_attempt_parity_checks'] == probe._v2_failed_replay(record,failed,tick=tick),\n"
     "            'Independent same-five metadata/output replay and complete ledgers')\n        del failed\n"
     "        result.update(probe37_output_byte_exact=True,probe37_complete_flops_exact=True,\n"
     "            probe37_domains_exact=True,probe37_dispatches_exact=True,\n"
     "            operation_ledgers_verified=True,operation_counts_reported_not_rebased=True,\n"
     "            all5_failed_attempt_outputs_flops_domains_dispatches_exact=True)"),
]
_V2_WATCH_EDITS = [
    ("HERE/'watch_check4.py'","HERE/'watch_check4_v2.py'",1),
    ("HERE/'check4.py'","HERE/'check4_v2.py'",1),
    ("HERE/'CHECK4_PROTOCOL.md'","HERE/'CHECK4_V2_PROTOCOL.md'",1),
    ("HERE/'CHECK4_WATCH_SPEC.md'","HERE/'CHECK4_V2_WATCH_SPEC.md'",1),
    ("HERE/'check4_v1',HERE/'check4_watch_v1'","HERE/'check4_v2',HERE/'check4_watch_v2'",1),
    ("current_comp_shared_old384_native_check4_watch_v1","current_comp_shared_old384_native_check4_watch_v2",1),
    ("current_comp_shared_old384_native_check4_v1","current_comp_shared_old384_native_check4_v2",1),
    ("complete_bills_exact","complete_flops_exact",3),
    ("TRANSPORT:TRANSPORT_SHA,_OWNER:_OWNER_SHA}",
     "TRANSPORT:TRANSPORT_SHA,_OWNER:_OWNER_SHA,_V2_WATCH_OWNER:_V2_WATCH_OWNER_SHA}",1),
    ("_derived = _original","_changes.extend((a,b,1) for a,b in _V2_WATCH_MAIN_EDITS)\n_derived = _original",1),
]
_V2_WATCH_DERIVED = _V2_WATCH_ORIGINAL
for _v2_before,_v2_after,_v2_count in _V2_WATCH_EDITS:
    if _V2_WATCH_DERIVED.count(_v2_before) != _v2_count:
        raise RuntimeError('Unique v2 watch adaptation: '+_v2_before[:100])
    _V2_WATCH_DERIVED = _V2_WATCH_DERIVED.replace(_v2_before,_v2_after)
_V2_WATCH_INVERSE = _V2_WATCH_DERIVED
for _v2_before,_v2_after,_v2_count in reversed(_V2_WATCH_EDITS):
    if _V2_WATCH_INVERSE.count(_v2_after) != _v2_count:
        raise RuntimeError('Unique v2 watch inverse: '+_v2_after[:100])
    _V2_WATCH_INVERSE = _V2_WATCH_INVERSE.replace(_v2_after,_v2_before)
if _V2_WATCH_INVERSE != _V2_WATCH_ORIGINAL:
    raise RuntimeError('Entire frozen v1 supervisor restored byte-exact')
ast.parse(_V2_WATCH_DERIVED)
V2_WATCH_SOURCE_INVERSE_PROOF = dict(owner=str(_V2_WATCH_OWNER),owner_sha256=_V2_WATCH_OWNER_SHA,
    whole_source_inverse_verified=True,derived_sha256=hashlib.sha256(_V2_WATCH_DERIVED.encode()).hexdigest(),
    original_resource_deadlines_sampling_cleanup_and_seals_unchanged=True,
    old_negative_receipts_preserved=True,splices=len(_V2_WATCH_EDITS))
exec(compile(_V2_WATCH_DERIVED,str(_V2_WATCH_OWNER)+':fresh-v2-accounting-metadata','exec'),globals())
