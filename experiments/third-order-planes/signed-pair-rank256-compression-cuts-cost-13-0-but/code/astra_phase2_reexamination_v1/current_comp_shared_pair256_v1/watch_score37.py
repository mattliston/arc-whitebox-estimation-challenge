"""Independent120s/1GB saved-output score supervisor; inert without --run.

Derives the qualified original one-rank score watch by an exact positional
whole-source inverse. The original resource loop, cleanup and bounded1s
publication tail are unchanged; only explicit current metadata/owners differ.
"""
import hashlib
from pathlib import Path
import re
import shlex
import sys
import time

_W37_HERE = Path(__file__).resolve().parent
_W37_OWNER = _W37_HERE.parent/'current_comp_shared_old384_v1/watch_score_probe.py'
_W37_OWNER_SHA = 'ce125eeb2bf774f9647b925018ba3ea1bf4e7b0a4e18277d6fab2d1c09d7d6a5'
_W37_KEYS = ('source','worker','protocol','watch-spec','native-source','native-watch-source',
    'native-protocol','native-watch-spec','native','native-watch','builder','build-protocol','build','candidate')


def foreign_servers(rows, allowed):
    """Reject the underscore python -m flopscope_server spelling as well."""
    found = []
    for row in rows:
        if row['pid'] in allowed:
            continue
        try:
            argv = shlex.split(row['command'])
        except ValueError:
            argv = row['command'].split()
        if argv and re.fullmatch(r'(?:python|pypy)[0-9.]*',Path(argv[0]).name.lower()) and any(
            x == 'flopscope_server' or x.startswith('flopscope_server.') for x in argv[1:]):
            found.append(row)
    return found


def derive_source():
    original = _W37_OWNER.read_text()
    if hashlib.sha256(original.encode()).hexdigest() != _W37_OWNER_SHA:
        raise RuntimeError('Frozen qualified one-rank score supervision owner')
    start = original.index('    own = {',original.index('def main():'))
    stop = original.index('    folder,out = ',start)
    ownership = """    own = {HERE/'watch_score37.py':args.source_sha256,HERE/'score37.py':args.worker_sha256,
        HERE/'SCORE37_PROTOCOL.md':args.protocol_sha256,HERE/'SCORE37_WATCH_SPEC.md':args.watch_spec_sha256,
        _W37_OWNER:_W37_OWNER_SHA,TRANSPORT:TRANSPORT_SHA,**h.MON_PINS}
    inputs = {**h.MOCK_PINS,HERE/'native37_v1/result.json':args.native_sha256,
        HERE/'native37_watch_v1/result.json':args.native_watch_sha256,
        HERE/'native37.py':args.native_source_sha256,HERE/'watch_native37.py':args.native_watch_source_sha256,
        HERE/'NATIVE37_PROTOCOL.md':args.native_protocol_sha256,HERE/'NATIVE37_WATCH_SPEC.md':args.native_watch_spec_sha256,
        HERE/'build37.py':args.builder_sha256,HERE/'BUILD37_PROTOCOL.md':args.build_protocol_sha256,
        HERE/'build37_v1/result.json':args.build_sha256,HERE/'build37_v1/estimator.py':args.candidate_sha256}
"""
    flags_start = original.index("        flags = ('integrity_passed'")
    flags_stop = original.index("        require(r['requested_seals']",flags_start)
    contract = """        scorer.watch_contract(args,r,folder,sha,tick)
        require(r['source_bindings'].get(str(_W37_OWNER)) == _W37_OWNER_SHA,
            'Immutable original supervisor belongs to the actual child source union')
"""
    old_owners = """        require(all(r['source_bindings'].get(str(p)) == v for p,v in own.items()
            if p in (HERE/'score_probe.py',HERE/'watch_score_probe.py',HERE/'SCORE_PROBE_PROTOCOL.md')),
            'Exact three current scorer/watch/protocol source owners')
"""
    new_owners = """        require(all(r['source_bindings'].get(str(p)) == v for p,v in own.items()
            if p in (HERE/'score37.py',HERE/'watch_score37.py',HERE/'SCORE37_PROTOCOL.md',
                HERE/'SCORE37_WATCH_SPEC.md',_W37_OWNER)),
            'Exact four current scorer/watch/protocol/spec and original watch owners')
"""
    edits = [
        ("KEYS = ('source','worker','protocol','probe-source','probe-watch-source',\n        'probe-protocol','probe-watch-spec','probe','probe-watch')",'KEYS = '+repr(_W37_KEYS),1),
        ('began = previous = time.monotonic()','began = previous = WATCH37_BEGAN',1),
        (original[start:stop],ownership,1),
        (original[flags_start:flags_stop],contract,1),
        (old_owners,new_owners,1),
        ("HERE/'score_probe.py'","HERE/'score37.py'",2),
        ("HERE/'score_probe_v1'","HERE/'score37_v1'",1),
        ("HERE/'score_probe_watch_v1'","HERE/'score37_watch_v1'",1),
        ("'current_comp_shared_old384_score_probe_watch_v1'","'current_comp_shared_pair256_score37_watch_v1'",1),
        ("        require(not classifier(snapshot['rows'],scope.pop('allowed')),'No foreign numerical worker')",
         "        allowed = scope.pop('allowed')\n"
         "        require(not classifier(snapshot['rows'],allowed) and not foreign_servers(snapshot['rows'],allowed),\n"
         "            'No foreign numerical worker or underscore-module RPC server')",1),
        ("        env.update({k:'1' for k in BLAS})",
         "        env.update({k:'1' for k in BLAS})\n        env.pop('PYTHONPATH',None)\n"
         "        env.pop('PYTHONOPTIMIZE',None)\n        env.pop('FLOPSCOPE_SERVER_URL',None)",1),
        ("        result.update(passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True,",
         "        result.update(candidate_sha256=r['candidate_sha256'],control_candidate_sha256=r['control_candidate_sha256'],\n"
         "            comparison=r['comparison'],record=r['record'],lineage=r['lineage'],\n"
         "            scientific_gate_passed=r['scientific_gate_passed'],score37_wrapper_proof=r['score37_wrapper_proof'],\n"
         "            source_adapter_proof=WATCH37_SOURCE_PROOF,repeat_excluded=True,ranks=[37],\n"
         "            estimator_executed=False,training_or_refit_performed=False,production_qualified=False,\n"
         "            official_submission_success_claimed=False)\n"
         "        result.update(passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True,",1),
        ("{'numpy','torch','flopscope','threadpoolctl'}","{'numpy','scipy','torch','flopscope','flopscope_server','threadpoolctl'}",1),
    ]
    changed,spans = original,[]
    for before,after,count in edits:
        if changed.count(before) != count:
            raise RuntimeError('Exact score37 watch splice: '+repr((before[:100],changed.count(before),count)))
        cursor = 0
        for unused in range(count):
            position = changed.index(before,cursor); spans.append((position,before,after))
            changed = changed[:position]+after+changed[position+len(before):]; cursor = position+len(after)
    inverse = changed
    for position,before,after in reversed(spans):
        if inverse[position:position+len(after)] != after:
            raise RuntimeError('Exact positional score37 watch inverse')
        inverse = inverse[:position]+before+inverse[position+len(after):]
    if inverse != original:
        raise RuntimeError('Entire original qualified scorer supervisor restored')
    return changed,dict(owner_sha256=_W37_OWNER_SHA,whole_source_inverse_verified=True,splices=len(spans),
        derived_source_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        original120s_1GB_1s_sampling_cleanup_and_publication_tail_unchanged=True,
        actual_native_build_watch_authorities=True,scientific_gate_unchanged=True,
        foreign_rpc_underscore_spelling_rejected=True,no_supervisor_truth_bytes_or_numerical_imports=True)


def main():
    if '--run' not in sys.argv:
        print('Inert saved score watch: --run --numerical-slot-clear and14 actual SHA flags required.')
        return 0
    began = time.monotonic()
    source,proof = derive_source()
    env = dict(__file__=str(Path(__file__).resolve()),__name__='_pair256_score37_watch_effective',
        _W37_OWNER=_W37_OWNER,_W37_OWNER_SHA=_W37_OWNER_SHA,WATCH37_BEGAN=began,
        WATCH37_SOURCE_PROOF=proof,foreign_servers=foreign_servers)
    exec(compile(source,str(Path(__file__).resolve())+':qualified-watch-inverse','exec'),env)
    return env['main']()


if __name__ == '__main__':
    raise SystemExit(main())
