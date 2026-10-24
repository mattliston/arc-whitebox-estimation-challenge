"""Original300s native supervisor; fixed production-backend35/35 diagnostic only."""
import hashlib
from pathlib import Path

_BW_HERE = Path(__file__).resolve().parent
_BW_OWNER = _BW_HERE/'watch_probe.py'
_BW_OWNER_SHA = '7ade1c9025a2932e7929498062f894b72a3817d50027e7c5d1cfbf22a1f0fd18'
_BW_PINS = {
    _BW_OWNER:_BW_OWNER_SHA,
    _BW_HERE/'probe.py':'6193396f83150e7e95f3c16ad5095eeb8d4c62da17d69d5bd7dcb08fad9a743b',
    _BW_HERE/'PROBE_PROTOCOL.md':'43e0c94e729a7004d2d51fb6424d4122e7d6f168a1f241095ff33abac2ad04c2',
    _BW_HERE/'PROBE_WATCH_SPEC.md':'743fc63fcb7ff5a7517fcb4e4311ee586fbd8ec72ed36004e2883810298cdc3f'}


def derive_source():
    original = _BW_OWNER.read_text()
    if hashlib.sha256(original.encode()).hexdigest() != _BW_OWNER_SHA:
        raise RuntimeError('Immutable original fully sampled native probe watch')
    edits = [
        ("HERE/'watch_probe.py'","HERE/'watch_backend_probe.py'",1),
        ("HERE/'probe.py'","HERE/'backend_probe.py'",3),
        ("HERE/'PROBE_PROTOCOL.md'","HERE/'BACKEND_PROBE_PROTOCOL.md'",1),
        ("HERE/'PROBE_WATCH_SPEC.md'","HERE/'BACKEND_PROBE_WATCH_SPEC.md'",1),
        ("HERE/'probe_v1'","HERE/'backend_probe_v1'",1),
        ("HERE/'probe_watch_v1'","HERE/'backend_probe_watch_v1'",1),
        ("'current_comp_shared_old384_native37_watch_v1'","'current_comp_shared_old384_backend_probe_watch_v1'",1),
        ("    require(all(re.fullmatch('[0-9a-f]{64}',v or '') and sha(p) == v for p,v in own.items()),",
         "    own.update(_BW_PINS)\n    require(all(re.fullmatch('[0-9a-f]{64}',v or '') and sha(p) == v for p,v in own.items()),",1),
        ("ranks=[37,37]","ranks=[35,35]",1),
        ("item['rank'] == 37,'Exact ordered37/37'","item['rank'] == 35,'Exact ordered35/35'",1),
        ("record['rank'] == 37 and record['context_retired']","record['rank'] == 35 and record['context_retired']",1),
        ("sha256=digest,call=index,rank=37","sha256=digest,call=index,rank=35",1),
        ("record['ranks'] == [37,37]","record['ranks'] == [35,35]",1),
        ("record['unaffected_prefix_checks'] == [dict(call=i,layers=5,byte_exact=True) for i in range(2)]",
         "probe.backend_contract(record,tick)",1),
        ("        result.update(passed=True,completed=True,child_execution_completed=True,whole_owned_tree_retired=True,",
         "        result.update(backend_probe_watch_source_proof=BACKEND_WATCH_SOURCE_PROOF,\n"
         "            same_production_backend_qualified=True,target_free_backend_diagnostic_passed=True,\n"
         "            all16_production_vector_parity=True,production_parity_checks=record['production_parity_checks'],\n"
         "            backend_attestation=record['backend_attestation'],initialization_flops=record['initialization_flops'],\n"
         "            control_seals=record['control_seals'],old_fidelity_still_failed=True,\n"
         "            fresh_same_backend_objective_required_before_bank=True,original_mse_ratio_gate_not_evaluated=True,\n"
         "            production_five_qualified=False,native_fidelity_qualified=False,\n"
         "            automatic_next_stage=False,submission_authorized=False)\n"
         "        result.update(passed=True,completed=True,child_execution_completed=True,whole_owned_tree_retired=True,",1)]
    changed,spans = original,[]
    for before,after,count in edits:
        if changed.count(before) != count:
            raise RuntimeError('Exact backend watch splice count: '+before[:100])
        cursor = 0
        for _ in range(count):
            pos = changed.index(before,cursor)
            spans.append((pos,before,after))
            changed = changed[:pos]+after+changed[pos+len(before):]
            cursor = pos+len(after)
    inverse = changed
    for pos,before,after in reversed(spans):
        if inverse[pos:pos+len(after)] != after:
            raise RuntimeError('Positional backend watch inverse')
        inverse = inverse[:pos]+before+inverse[pos+len(after):]
    if inverse != original:
        raise RuntimeError('Entire original sampled supervisor byte inverse')
    return changed,dict(owner=str(_BW_OWNER),owner_sha256=_BW_OWNER_SHA,
        whole_source_inverse_verified=True,derived_source_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        splices=len(spans),original_live_resource_cleanup_and_three_seal_loop_unchanged=True,
        supervisor_backend_not_changed=True,worker_only_explicit_backend_selection=True)


_BW_DERIVED, BACKEND_WATCH_SOURCE_PROOF = derive_source()
exec(compile(_BW_DERIVED,str(_BW_OWNER)+':production-backend-diagnostic','exec'),globals())
