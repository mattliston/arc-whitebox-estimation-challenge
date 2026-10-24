"""Exact successful probe supervisor adapted only to fixed CHECK4 and its boundary."""
import ast
import hashlib
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_OWNER = _HERE/'watch_probe.py'
_OWNER_SHA = '7ade1c9025a2932e7929498062f894b72a3817d50027e7c5d1cfbf22a1f0fd18'
_original = _OWNER.read_text()
if hashlib.sha256(_original.encode()).hexdigest() != _OWNER_SHA:
    raise RuntimeError('Exact successful original native probe watcher')
SEQUENCE = (37,39,42,48,37)
COMPOSED_SHA = '4d98fef2c11c571efe1eca945018bb9fe9ece1454d16b4abde2eb52d71e755c3'
_changes = [
    ("CAP, RSS, PROGRESS = 300.,8_000_000_000,120.", "CAP, RSS, PROGRESS = 660.,8_000_000_000,120.",1),
    ("HERE/'watch_probe.py'", "HERE/'watch_check4.py'",1),
    ("HERE/'probe.py'", "HERE/'check4.py'",3),
    ("HERE/'PROBE_PROTOCOL.md'", "HERE/'CHECK4_PROTOCOL.md'",1),
    ("HERE/'PROBE_WATCH_SPEC.md'", "HERE/'CHECK4_WATCH_SPEC.md'",1),
    ("HERE/'PROTOCOL.md':args.design_sha256,TRANSPORT:TRANSPORT_SHA}",
     "HERE/'PROTOCOL.md':args.design_sha256,TRANSPORT:TRANSPORT_SHA,_OWNER:_OWNER_SHA}",1),
    ("child_folder,out = HERE/'probe_v1',HERE/'probe_watch_v1'",
     "child_folder,out = HERE/'check4_v1',HERE/'check4_watch_v1'",1),
    ("schema='current_comp_shared_old384_native37_watch_v1'",
     "schema='current_comp_shared_old384_native_check4_watch_v1'",1),
    ("retired_calls=0,ranks=[37,37],completed_sample_intervals_s=[],markers=[])",
     "retired_calls=0,ranks=list(SEQUENCE),completed_sample_intervals_s=[],markers=[],\n"
     "        check4_watch_source_inverse_proof=SOURCE_INVERSE_PROOF)",1),
    ("'Whole300s/next retired call120s'", "'Whole660s/next retired call120s'",1),
    ("require(index < 2 and item['call'] == index and item['rank'] == 37,'Exact ordered37/37')",
     "require(index < 5 and item['call'] == index and item['rank'] == SEQUENCE[index],'Exact fixed CHECK4/repeat order')",1),
    ("and record['call'] == index and record['rank'] == 37 and record['context_retired']",
     "and record['call'] == index and record['rank'] == SEQUENCE[index] and record['context_retired']",1),
    ("dict(path=str(path),sha256=digest,call=index,rank=37)",
     "dict(path=str(path),sha256=digest,call=index,rank=SEQUENCE[index])",1),
    ("        probe.environment()\n        composition = load",
     "        probe.environment()\n"
     "        for path,digest in probe.EXTRA_SOURCES.items():\n"
     "            require(sha(path,tick) == digest, 'Direct frozen source authority')\n"
     "            result['source_bindings'][str(path)] = digest\n"
     "        prior,prerequisite_seals = probe.prerequisites(result['input_bindings'],tick)\n"
     "        result.update(probe_prerequisite_qualified=True,prerequisite_seals=prerequisite_seals)\n"
     "        del prior\n        composition = load",1),
    ("        result['candidate_sha256'] = hashlib.sha256(text.encode()).hexdigest()",
     "        result['candidate_sha256'] = hashlib.sha256(text.encode()).hexdigest()\n"
     "        require(result['candidate_sha256'] == COMPOSED_SHA, 'Only the successful frozen representation')",1),
    ("child.wait(timeout=1) == 0 and result['retired_calls'] == 2,'Two complete calls and exit0'",
     "child.wait(timeout=1) == 0 and result['retired_calls'] == 5,'Five complete calls and exit0'",1),
    ("and record['full_model_calls'] == record['retired_call_count'] == 2",
     "and record['full_model_calls'] == record['retired_call_count'] == 5",1),
    ("and record['ranks'] == [37,37] and record['all16_domains_qualified']",
     "and record['ranks'] == list(SEQUENCE) and record['all16_domains_qualified']",1),
    ("for i in range(2)]", "for i in range(5)]",1),
    ("            'Actual target-blind same-source own-cost native subject')\n        for group",
     "            'Actual target-blind same-source own-cost native subject')\n"
     "        require(record['schema'] == 'current_comp_shared_old384_native_check4_v1'\n"
     "            and len(record['trials']) == len(record['input_preparation']) == len(record['diagnostic_records']) == 5\n"
     "            and len(record['artifacts']) == 35 and record['initialization_flops'] == 1944970298\n"
     "            and record['prerequisite_seals'] == result['prerequisite_seals']\n"
     "            and record['probe37_parity_checks'] == [dict(call=c,probe_call=p,rank=37,\n"
     "                output_byte_exact=True,complete_bills_exact=True,domains_exact=True,dispatches_exact=True)\n"
     "                for c,p in ((0,0),(4,1))]\n"
     "            and all(record.get(k) is True for k in ('probe_prerequisite_qualified','probe37_output_byte_exact',\n"
     "                'probe37_complete_bills_exact','probe37_domains_exact','probe37_dispatches_exact'))\n"
     "            and record['check4_source_inverse_proof'] == probe.SOURCE_INVERSE_PROOF,\n"
     "            'Actual fixed five-call expansion, exact37 replay and unchanged numerical source')\n"
     "        result.update(probe37_output_byte_exact=True,probe37_complete_bills_exact=True,\n"
     "            probe37_domains_exact=True,probe37_dispatches_exact=True)\n        for group",1),
]
_derived = _original
for _before,_after,_count in _changes:
    if _derived.count(_before) != _count:
        raise RuntimeError('Unique CHECK4-watch source adaptation: '+_before[:100])
    _derived = _derived.replace(_before,_after)
_inverse = _derived
for _before,_after,_count in reversed(_changes):
    if _inverse.count(_after) != _count:
        raise RuntimeError('Unique CHECK4-watch source inverse: '+_after[:100])
    _inverse = _inverse.replace(_after,_before)
if _inverse != _original:
    raise RuntimeError('Whole original qualified resource supervisor restored byte-exact')
ast.parse(_derived)
SOURCE_INVERSE_PROOF = dict(owner=str(_OWNER),owner_sha256=_OWNER_SHA,whole_source_inverse_verified=True,
    derived_sha256=hashlib.sha256(_derived.encode()).hexdigest(),splices=len(_changes),
    original_transport_sampling_progress_and_cleanup_unchanged=True,whole_cap_s=660.,
    fixed_roster_and_positive_boundary_metadata_only=True)
exec(compile(_derived,str(_OWNER)+':fixed-CHECK4-metadata-adapter','exec'),globals())
