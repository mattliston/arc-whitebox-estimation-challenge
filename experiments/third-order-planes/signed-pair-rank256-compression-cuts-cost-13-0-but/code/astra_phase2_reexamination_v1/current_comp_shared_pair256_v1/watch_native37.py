"""DRAFT: exact original300s/8GB fixed37/37 supervisor; no numerical imports."""
import ast
import hashlib
from pathlib import Path
import re
import shlex
import sys
import time

HERE = Path(__file__).resolve().parent
OFFPY = HERE.parent.parent/'data_gate/.venv/bin/python'
OWNER = HERE.parent/'current_comp_shared_old384_v1/watch_probe.py'
OWNER_SHA = '7ade1c9025a2932e7929498062f894b72a3817d50027e7c5d1cfbf22a1f0fd18'
KEYS = ('source','worker','protocol','spec','builder','build-protocol','build',
        'build-watch','build-watch-source','build-watch-spec','candidate')


def foreign_servers(rows,allowed):
    """Cover python -m flopscope_server, omitted by the old dash-spelled classifier."""
    found=[]
    for row in rows:
        if row['pid'] in allowed:
            continue
        try:
            argv=shlex.split(row['command'])
        except ValueError:
            argv=row['command'].split()
        if argv and re.fullmatch(r'(?:python|pypy)[0-9.]*',Path(argv[0]).name.lower()) and any(
            x=='flopscope_server' or x.startswith('flopscope_server.') for x in argv[1:]):
            found.append(row)
    return found


def child_command(args):
    """The runtime and preflight use this same explicit11-field translation."""
    names = (('source','worker'),('watch','source'),('protocol','protocol'),('watch_spec','spec'),
        ('builder','builder'),('build_protocol','build_protocol'),('build','build'),
        ('build_watch','build_watch'),('build_watch_source','build_watch_source'),
        ('build_watch_spec','build_watch_spec'),('candidate','candidate'))
    return [str(OFFPY),'-B','-u',str(HERE/'native37.py'),'--run',
        *(v for worker,watch in names for v in
          ('--'+worker.replace('_','-')+'-sha256',getattr(args,watch+'_sha256')))]


def derive_source():
    original = OWNER.read_text()
    if hashlib.sha256(original.encode()).hexdigest() != OWNER_SHA:
        raise RuntimeError('Frozen original two-call supervisor')
    old_own = """    own = {HERE/'watch_probe.py':args.source_sha256,HERE/'probe.py':args.worker_sha256,
        HERE/'PROBE_PROTOCOL.md':args.protocol_sha256,HERE/'PROBE_WATCH_SPEC.md':args.spec_sha256,
        HERE/'compose.py':args.compose_sha256,HERE/'fnp.py':args.fnp_sha256,
        HERE/'PROTOCOL.md':args.design_sha256,TRANSPORT:TRANSPORT_SHA}
"""
    new_own = """    own = {HERE/'watch_native37.py':args.source_sha256,HERE/'native37.py':args.worker_sha256,
        HERE/'NATIVE37_PROTOCOL.md':args.protocol_sha256,HERE/'NATIVE37_WATCH_SPEC.md':args.spec_sha256,
        HERE/'build37.py':args.builder_sha256,HERE/'BUILD37_PROTOCOL.md':args.build_protocol_sha256,
        HERE/'watch_build37.py':args.build_watch_source_sha256,
        HERE/'BUILD37_WATCH_SPEC.md':args.build_watch_spec_sha256,TRANSPORT:TRANSPORT_SHA,OWNER:OWNER_SHA}
"""
    start = original.index("        probe = load(HERE/'probe.py'")
    stop = original.index('        env = dict(os.environ',start)
    old_admission = original[start:stop]
    new_admission = """        probe = load(HERE/'native37.py','pair256_native37_metadata',args.worker_sha256)
        probe.environment()
        argv = child_command(args)
        worker_args = probe.parse_child_command(argv)
        qualified = probe.admission(worker_args,tick)
        for group in ('source_bindings','input_bindings'):
            for path,digest in qualified[group].items():
                require(path not in result[group] or result[group][path] == digest,'No admission owner conflict')
                result[group][path] = digest
        result.update(candidate_sha256=qualified['candidate_sha256'],composition_proof=qualified['proof'],
            build_sha256=qualified['build_sha256'],build_watch_sha256=qualified['build_watch_sha256'],
            build_seals=qualified['build_seals'],build_watch_seals=qualified['build_watch_seals'],
            control_seals=qualified['control_seals'],current_c2_controls_qualified=True,
            build_watch_qualified=True,tiny_prerequisite_qualified=True,actual_worker_cli_namespace_qualified=True,
            native37_watch_source_proof=NATIVE37_WATCH_SOURCE_PROOF,authority=qualified['authority'],
            automatic_next_stage=False,submission_authorized=False,raw_error_measured=False,
            adjusted_score_measured=False,production_five_qualified=False,current_candidate_native40_qualified=False)
        del qualified
        require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules),'No supervisor numerical imports')
"""
    edits = (
        ("KEYS = ('source','worker','protocol','spec','compose','fnp','design')",'KEYS = '+repr(KEYS)),
        ('began = previous = last_progress = time.monotonic()','began = previous = last_progress = WATCH_BEGAN'),
        (old_own,new_own),
        ("child_folder,out = HERE/'probe_v1',HERE/'probe_watch_v1'",
         "child_folder,out = HERE/'native37_v1',HERE/'native37_watch_v1'"),
        ("'current_comp_shared_old384_native37_watch_v1'","'current_comp_shared_pair256_native37_watch_v1'"),
        ("        require(not classifier(sample['rows'],scope.pop('allowed')),'No foreign numerical worker')",
         "        allowed = scope.pop('allowed')\n"
         "        require(not classifier(sample['rows'],allowed),'No foreign numerical worker')\n"
         "        require(not foreign_servers(sample['rows'],allowed),'No foreign underscore-spelled numerical server')"),
        (old_admission,new_admission),
        ("        env.update({k:'1' for k in BLAS})",
         "        env.update({k:'1' for k in BLAS})\n        env.pop('PYTHONPATH',None)\n"
         "        env.pop('PYTHONOPTIMIZE',None)\n        env.pop('FLOPSCOPE_SERVER_URL',None)"),
        ("            'Actual target-blind same-source own-cost native subject')",
         "            'Actual target-blind same-source own-cost native subject')\n"
         "        checked = probe.verify_native(record,tick)\n"
         "        result.update(checked,actual_production_backend_qualified=True,\n"
         "            all_parameter_and_estimator_owners_retired=True,initialization_flops=record['initialization_flops'],\n"
         "            all16_domains_qualified=True,all47_selectors_qualified=True,repeat_byte_exact=True,\n"
         "            own_complete_bills=True,full_model_calls=2)\n"
         "        require(record['authority'] == result['authority']\n"
         "            and record['build_seals'] == result['build_seals']\n"
         "            and record['build_watch_seals'] == result['build_watch_seals']\n"
         "            and record['control_seals'] == result['control_seals'],'Identical actual admission and child lineage')"),
    )
    changed,spans = original,[]
    for before,after in edits:
        if changed.count(before) != 1:
            raise RuntimeError('Unique native37 watch metadata splice: '+before[:90])
        position=changed.index(before);spans.append((position,before,after))
        changed=changed[:position]+after+changed[position+len(before):]
    inverse=changed
    for position,before,after in reversed(spans):
        if inverse[position:position+len(after)] != after:
            raise RuntimeError('Exact positional fixed-watch inverse')
        inverse=inverse[:position]+before+inverse[position+len(after):]
    if inverse != original:
        raise RuntimeError('Entire original fixed37/37 resource supervisor restored')
    return changed,dict(owner_sha256=OWNER_SHA,whole_source_inverse_verified=True,splices=len(spans),
        derived_source_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        original300s_8GB_120s_progress_1s_samples_cleanup_tail_unchanged=True,
        no_new_numerical_supervisor_operations=True,actual_worker_parser_used_for_admission=True)


def main():
    if '--run' not in sys.argv:
        print('Inert native37 supervisor; explicit --run, --numerical-slot-clear and11 SHA flags required.')
        return 0
    began=time.monotonic()
    source,proof=derive_source()
    scope=dict(__file__=str(Path(__file__).resolve()),__name__='_pair256_native37_watch_effective',
        OWNER=OWNER,OWNER_SHA=OWNER_SHA,WATCH_BEGAN=began,NATIVE37_WATCH_SOURCE_PROOF=proof,
        foreign_servers=foreign_servers)
    exec(compile(source,str(Path(__file__).resolve())+':whole-watch-inverse','exec'),scope)
    scope['child_command'] = child_command
    return scope['main']()


if __name__ == '__main__':
    raise SystemExit(main())
