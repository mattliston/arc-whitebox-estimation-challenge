"""Inert source-inverse of the qualified300s/8GB two-call native supervisor."""
import hashlib
from pathlib import Path
import re
import shlex
import types

HERE = Path(__file__).resolve().parent
OWNER = HERE.parent/'current_comp_shared_old384_v1/watch_probe.py'
OWNER_SHA = '7ade1c9025a2932e7929498062f894b72a3817d50027e7c5d1cfbf22a1f0fd18'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def foreign_servers(rows, allowed):
    """Also cover the original classifier's underscore-spelled server gap."""
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


def worker_args(args):
    return types.SimpleNamespace(source_sha256=args.worker_sha256,watcher_sha256=args.source_sha256,
        protocol_sha256=args.protocol_sha256,watch_spec_sha256=args.spec_sha256)


def child_command(args):
    offpy = HERE.parent.parent/'data_gate/.venv/bin/python'
    return [str(offpy),'-B','-u',str(HERE/'capture.py'),'--run',
        '--source-sha256',args.worker_sha256,'--watcher-sha256',args.source_sha256,
        '--protocol-sha256',args.protocol_sha256,'--watch-spec-sha256',args.spec_sha256]


def derive_source():
    original = OWNER.read_text()
    require(hashlib.sha256(original.encode()).hexdigest() == OWNER_SHA, 'Exact completed resource-loop owner')
    begin = original.index('        probe.environment()')
    end = original.index("        require(not {'numpy','torch','flopscope','threadpoolctl'}",begin)
    old_admission = original[begin:end]
    begin = original.index('        argv = [str(OFFPY)')
    end = original.index('        env = dict(os.environ',begin)
    old_argv = original[begin:end]
    old_own = """    own = {HERE/'watch_probe.py':args.source_sha256,HERE/'probe.py':args.worker_sha256,
        HERE/'PROBE_PROTOCOL.md':args.protocol_sha256,HERE/'PROBE_WATCH_SPEC.md':args.spec_sha256,
        HERE/'compose.py':args.compose_sha256,HERE/'fnp.py':args.fnp_sha256,
        HERE/'PROTOCOL.md':args.design_sha256,TRANSPORT:TRANSPORT_SHA}"""
    new_own = """    own = {HERE/'watch_capture.py':args.source_sha256,HERE/'capture.py':args.worker_sha256,
        HERE/'PROTOCOL.md':args.protocol_sha256,HERE/'WATCH_SPEC.md':args.spec_sha256,
        OWNER:OWNER_SHA,TRANSPORT:TRANSPORT_SHA}"""
    edits = [
        ("KEYS = ('source','worker','protocol','spec','compose','fnp','design')",
         "KEYS = ('source','worker','protocol','spec')",1),
        (old_own,new_own,1),
        ("HERE/'probe_v1',HERE/'probe_watch_v1'",
         "HERE/'capture_qualification_v1',HERE/'capture_qualification_watch_v1'",1),
        ("'current_comp_shared_old384_native37_watch_v1'",
         "'current_c2_terminal_quadratic_capture_qualification_watch_v1'",1),
        ("ranks=[37,37]","ranks=[35,35]",1),
        ("item['rank'] == 37,'Exact ordered37/37'","item['rank'] == 35,'Exact ordered35/35'",1),
        ("record['rank'] == 37 and record['context_retired']","record['rank'] == 35 and record['context_retired']",1),
        ("sha256=digest,call=index,rank=37","sha256=digest,call=index,rank=35",1),
        ("('diagnostics','diagnostic_sha256')):",
         "('diagnostics','diagnostic_sha256'),('capture','capture_manifest_sha256'),\n"
         "                            ('replay_bill','replay_bill_sha256'),('replay_lifecycle','replay_lifecycle_sha256')):",1),
        ("probe = load(HERE/'probe.py','shared384_passive_probe_api',args.worker_sha256)",
         "probe = load(HERE/'capture.py','current_c2_passive_capture_api',args.worker_sha256)",1),
        (old_admission,"""        context = probe.admission(worker_args(args),tick)
        for group in ('source_bindings','input_bindings'):
            for path,digest in context[group].items():
                require(path not in result[group] or result[group][path] == digest,'Consistent direct admission')
                result[group][path] = digest
        result['candidate_sha256'] = context['candidate_sha256']
        result['source_proof'] = context['source_proof']
        result['control_seals'] = context['control_seals']
        result['watch_source_proof'] = WATCH_SOURCE_PROOF
""",1),
        (old_argv,"        argv = child_command(args)\n",1),
        ("record['composition_proof'] == result['composition_proof']",
         "record['source_proof'] == result['source_proof']",1),
        ("record['ranks'] == [37,37]","record['ranks'] == [35,35]",1),
        ("record['unaffected_prefix_checks'] == [dict(call=i,layers=5,byte_exact=True) for i in range(2)]",
         "probe.verify_result(record,tick)",1),
        ("        require(not classifier(sample['rows'],scope.pop('allowed')),'No foreign numerical worker')",
         "        allowed = scope.pop('allowed')\n"
         "        require(not classifier(sample['rows'],allowed) and not foreign_servers(sample['rows'],allowed),\n"
         "                'No foreign numerical worker or underscore-spelled server')",1),
        ("        result.update(passed=True,completed=True,child_execution_completed=True,whole_owned_tree_retired=True,",
         "        result.update(exact_head_replay=True,capture_repeat_byte_exact=True,\n"
         "            current_control_output_byte_exact=True,current_control_complete_flops_exact=True,\n"
         "            all16_domains_qualified=True,all47_selectors_qualified=True,own_complete_bills=True,\n"
         "            all_parameter_and_estimator_owners_retired=True,actual_production_backend_qualified=True,\n"
         "            backend_attestation=record['backend_attestation'],initialization_flops=record['initialization_flops'],\n"
         "            research_flops=record['research_flops'],full_model_calls=2,parameter_updates=0,\n"
         "            fit_performed=False,training_expansion_authorized=False,automatic_next_stage=False,\n"
         "            submission_authorized=False)\n"
         "        result.update(passed=True,completed=True,child_execution_completed=True,whole_owned_tree_retired=True,",1),
    ]
    changed,spans = original,[]
    for before,after,count in edits:
        require(changed.count(before) == count, 'Exact prospective capture-watch splice: '+before[:85])
        cursor = 0
        for _ in range(count):
            position = changed.index(before,cursor)
            spans.append((position,before,after))
            changed = changed[:position]+after+changed[position+len(before):]
            cursor = position+len(after)
    inverse = changed
    for position,before,after in reversed(spans):
        require(inverse[position:position+len(after)] == after, 'Exact positional inverse')
        inverse = inverse[:position]+before+inverse[position+len(after):]
    require(inverse == original, 'Whole qualified resource/retirement/seal loop byte inverse')
    return changed,dict(owner_sha256=OWNER_SHA,whole_source_inverse_verified=True,splices=len(spans),
        derived_source_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        whole300s_fulltree8GB_zero_added_swap_120s_progress_1s_sampling_and_sealed_tail_retained=True)


def main():
    source,proof = derive_source()
    env = dict(__name__='_current_c2_capture_watch_body',__file__=str(Path(__file__).resolve()),
        OWNER=OWNER,OWNER_SHA=OWNER_SHA,WATCH_SOURCE_PROOF=proof,worker_args=worker_args,
        child_command=child_command,foreign_servers=foreign_servers)
    exec(compile(source,'<exact-qualified-two-call-capture-watch>','exec'),env)
    return env['main']()


if __name__ == '__main__':
    raise SystemExit(main())
