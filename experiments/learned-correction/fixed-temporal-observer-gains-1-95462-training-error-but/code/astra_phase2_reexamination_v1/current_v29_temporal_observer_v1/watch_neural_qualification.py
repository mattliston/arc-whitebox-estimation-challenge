"""Inert180s observer-only qualification inverse of the qualified AP supervisor."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
OWNER = HERE.parent/'current_comp_cstar_proposal_v1/watch.py'
OWNER_SHA = 'd3258f391ca501698e6c5f14a9dce21f28ae4e63d05731800ce6b2da8c94f23e'
SITE = Path('/Users/matt/anaconda3/lib/python3.11/site-packages')
EXTERNAL_PINS = {SITE/'torch/optim/adam.py',SITE/'torch/optim/adamw.py',
    SITE/'torch/__init__.py',SITE/'numpy/__init__.py'}
CAPTURE_FLAGS = ('capture-worker','capture-watch-source','capture-protocol','capture-watch-spec',
    'capture','capture-watch','projection-qualification','projection-qualification-watch')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def worker_args(args):
    values = dict(vars(args)); values.update(source_sha256=args.worker_sha256,watcher_sha256=args.source_sha256)
    return types.SimpleNamespace(**values)


def load_worker(path, digest):
    require(not path.is_symlink() and hashlib.sha256(path.read_bytes()).hexdigest()==digest, 'Exact inert neural worker')
    spec=importlib.util.spec_from_file_location('neural_qualification_metadata',path)
    value=importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def check_roster(path, stage):
    require(stage=='qualify' and not path.is_symlink() and path.stat().st_size<65536, 'Fixed small qualification roster')
    value=json.loads(path.read_text())
    rows=((None,'tiny_null','cpu'),(None,'tiny_nonzero_backward','cpu'),
        (None,'tiny_null','mps'),(None,'tiny_nonzero_backward','mps'),
        (None,'tiny_step7_backward','cpu'),(None,'tiny_step7_backward','mps'),
        (35,'cached_null','mps'),(58,'cached_null','mps'),
        (35,'cached_nonzero_backward','mps'),(58,'cached_nonzero_backward','mps'),(35,'cached_nonzero_backward','mps'))
    expected=[dict(graph_index=i,rank=r,kind=k,device=d,update=None) for i,(r,k,d) in enumerate(rows)]
    require(value==dict(schema='current_v29_observer_neural_qualification_roster_v1',stage='qualify',
        initialization_seed=93091229,entries=expected), 'Exact four tiny rollouts/two steps/five cached evaluations')
    return expected


def rewrite(original, edits):
    changed,history=original,[]
    for old,new,count in edits:
        require(old and changed.count(old)==count, 'Exact neural-watch substitution '+old[:90])
        positions=[]; cursor=0
        while True:
            at=changed.find(old,cursor)
            if at<0:
                break
            positions.append(at); cursor=at+len(old)
        parts=[]; cursor=shift=0
        for at in positions:
            parts.extend((changed[cursor:at],new)); history.append((at+shift,old,new))
            cursor=at+len(old); shift+=len(new)-len(old)
        parts.append(changed[cursor:]); changed=''.join(parts)
    inverse=changed
    for at,old,new in reversed(history):
        require(inverse[at:at+len(new)]==new, 'Exact recorded neural-watch inverse span')
        inverse=inverse[:at]+old+inverse[at+len(new):]
    require(inverse==original, 'Whole qualified supervisor byte inverse')
    return changed


def derive_source():
    require(not OWNER.is_symlink() and hashlib.sha256(OWNER.read_bytes()).hexdigest()==OWNER_SHA, 'Frozen complete AP supervisor')
    original=OWNER.read_text(); lines=original.splitlines(keepends=True)
    nodes=[n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef) and n.name=='check_roster']
    require(len(nodes)==1, 'Original unique roster validator')
    n=nodes[0]; old_roster=''.join(lines[n.lineno-1:n.end_lineno])
    old_owners="""    own = {HERE/'watch.py':args.source_sha256,HERE/'worker.py':args.worker_sha256,
        HERE/'common.py':args.common_sha256,HERE/'model.py':args.model_sha256,
        HERE/'PROTOCOL.md':args.protocol_sha256,HERE/'WATCH_SPEC.md':args.watch_spec_sha256}
"""
    new_owners="""    own = {HERE/'watch_neural_qualification.py':args.source_sha256,HERE/'neural_qualification.py':args.worker_sha256,
        HERE/'projected_model.py':args.model_sha256,HERE/'NEURAL_QUALIFICATION_PROTOCOL.md':args.protocol_sha256,
        HERE/'NEURAL_QUALIFICATION_WATCH_SPEC.md':args.watch_spec_sha256,_neural.OWNER:_neural.OWNER_SHA}
"""
    old_argv="""        argv = [AP,'-B','-u',str(HERE/'worker.py'),'--run','--stage',args.stage,
            '--source-sha256',args.worker_sha256,'--common-sha256',args.common_sha256,
            '--model-sha256',args.model_sha256,'--protocol-sha256',args.protocol_sha256,
            '--watcher-sha256',args.source_sha256,'--watch-spec-sha256',args.watch_spec_sha256]
        if args.stage == 'train':
            argv += ['--qualification-sha256',args.qualification_sha256,
                '--qualification-watch-sha256',args.qualification_watch_sha256]
"""
    new_argv="""        argv = [AP,'-B','-u',str(HERE/'neural_qualification.py'),'--run',
            '--source-sha256',args.worker_sha256,'--model-sha256',args.model_sha256,
            '--protocol-sha256',args.protocol_sha256,'--watcher-sha256',args.source_sha256,
            '--watch-spec-sha256',args.watch_spec_sha256]
        for key in _neural.CAPTURE_FLAGS:
            argv += ['--'+key+'-sha256',getattr(args,key.replace('-','_')+'_sha256')]
"""
    edits=[
        ("SCHEMA = 'current_comp_cstar_proposal_watch_v1'","SCHEMA = 'current_v29_temporal_observer_neural_qualification_watch_v1'",1),
        ('result = dict(schema=SCHEMA,stage=args.stage,passed=False,completed=False,',
         'result = dict(schema=SCHEMA,stage=args.stage,watch_source_proof=_neural_watch_proof,passed=False,completed=False,',1),
        ("        'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')","        'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')",1),
        ("require(path.is_relative_to(P2) and 'truthpost' not in path.parts and",
         "require((path.is_relative_to(P2) or path in _neural.EXTERNAL_PINS) and 'truthpost' not in path.parts and",1),
        (old_roster,"def check_roster(path, stage):\n    return _neural.check_roster(path,stage)",1),
        ("    parser.add_argument('--stage',choices=('qualify','train'))","    parser.set_defaults(stage='qualify')",1),
        ("    for key in ('source','worker','common','model','protocol','watch-spec',\n                'qualification','qualification-watch'):",
         "    for key in ('source','worker','model','protocol','watch-spec')+_neural.CAPTURE_FLAGS:",1),
        ("    limit,expected = (300.,5) if args.stage == 'qualify' else (1800.,200)","    limit,expected = 180.,11",1),
        (old_owners,new_owners,1),
        ("    require(bool(args.qualification_sha256) == bool(args.qualification_watch_sha256) == (args.stage == 'train'),\n"
         "        'Only training receives new qualification authorities')",
         "    require(all(re.fullmatch('[0-9a-f]{64}',getattr(args,k.replace('-','_')+'_sha256') or '')\n"
         "        for k in _neural.CAPTURE_FLAGS),'Every actual projected capture/qualification source and result hash')",1),
        ("folder = HERE/('qualification_v1' if args.stage == 'qualify' else 'training_v1')","folder = HERE/'neural_qualification_v1'",1),
        ("out = HERE/('qualification_watch_v1' if args.stage == 'qualify' else 'training_watch_v1')","out = HERE/'neural_qualification_watch_v1'",1),
        ("                    require(all(record[k] == v for k,v in roster[index].items()),\n"
         "                        'Every observed completion matches its prospectively fixed roster')",
         "                    require(all(record[k] == v for k,v in roster[index].items())\n"
         "                        and item['device'] == record['device'], 'Every completion matches exact device/roster')\n"
         "                    require(record['maximum_live_mps_driver_bytes'] < 16_000_000_000\n"
         "                        and all(0 <= v < 16_000_000_000 for v in record['live_mps_driver_samples']),\n"
         "                        'Driver samples before graph/cache retirement satisfy original16GB cap')",1),
        ("                        value = item[key]","                        value = item[key]\n                        require(value == record[key],'Actual marker resource equals its completed record')",1),
        ("        observe()\n        child_env = dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',",
         "        observe()\n"
         "        api = _neural.load_worker(HERE/'neural_qualification.py',args.worker_sha256)\n"
         "        worker_args = _neural.worker_args(args)\n"
         "        admitted = api.admit(worker_args,result['source_bindings'],result['input_bindings'],{},tick)\n"
         "        result['prerequisite_seals'] = admitted['prerequisite_seals']\n"
         "        result['model_source_proof'] = admitted['model_proof']\n"
         "        require(not {'numpy','torch','threadpoolctl','flopscope'} & set(sys.modules),'Admission remains stdlib-only')\n"
         "        child_env = dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',",1),
        (old_argv,new_argv,1),
        ("        union = {}","        api.verify_result(record,worker_args,admitted,tick)\n        union = {}",1),
        ("            foreign_numerical_workers_checked=True)",
         "            foreign_numerical_workers_checked=True,neural_execution_qualified=True,\n"
         "            physical_model_calls=0,training_targets_accessed=False,validation_targets_accessed=False,\n"
         "            downstream_training_authorized=False)",1),
        ("        for handle in (log,reader,stream):",
         "        publication_deadline = min(previous+1.,began+limit,last_progress+120.)\n"
         "        result.update(last_actual_sample_elapsed_s=previous-began,\n"
         "            publication_deadline_elapsed_s=publication_deadline-began,\n"
         "            no_postpublication_observations=True)\n"
         "        for handle in (log,reader,stream):",1),
        ("            try:\n                deadline()\n            except BaseException:",
         "            try:\n                deadline()\n"
         "                require(time.monotonic()<publication_deadline,'One-second sealed publication tail')\n"
         "            except BaseException:",1),
        ("        write(out/'result.json',result)",
         "        if time.monotonic()>=publication_deadline:\n"
         "            result.update(passed=False,completed=False,publication_tail_error='Deadline before result seal')\n"
         "        result['elapsed_s'] = time.monotonic()-began\n"
         "        result['publication_tail_elapsed_before_result_s'] = time.monotonic()-previous\n"
         "        write(out/'result.json',result)",1),
        ("        ok = bool(result['passed'] and time.monotonic()-began < limit and time.monotonic()-last_progress < 120.)",
         "        ok = bool(result['passed'] and time.monotonic()<publication_deadline)",1),
        ("        ok = bool(ok and elapsed < limit and time.monotonic()-last_progress < 120.)",
         "        ok = bool(ok and time.monotonic()<publication_deadline)",1),
        ("            completion_sha256=completion_sha,elapsed_s=elapsed))",
         "            completion_sha256=completion_sha,elapsed_s=elapsed,\n"
         "            publication_tail_elapsed_s=time.monotonic()-previous,\n"
         "            publication_deadline_elapsed_s=publication_deadline-began,\n"
         "            no_postpublication_observations=True))",1),
        ("        ok = bool(ok and time.monotonic()-began < limit and time.monotonic()-last_progress < 120.)",
         "        ok = bool(ok and time.monotonic()<publication_deadline)",1),
        ("    print(json.dumps(dict(stage=args.stage,passed=ok,retired_calls=result['retired_calls'],",
         "    ok = bool(ok and time.monotonic()<publication_deadline)\n"
         "    print(json.dumps(dict(stage=args.stage,passed=ok,retired_calls=result['retired_calls'],",1),
        ("    return 0 if ok else 1",
         "    return 0 if ok and time.monotonic()<publication_deadline else 1",1),
    ]
    changed=rewrite(original,edits)
    return changed,dict(owner_sha256=OWNER_SHA,whole_source_inverse=True,
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),fixed_evaluations=11,whole_cap_s=180,
        unchanged_12GB_host_16GB_driver_zero_swap_120s_progress_1s_completed_samples=True,
        unchanged_live_monitor_transport_cleanup_and_three_seals=True,
        final_publication_bounded_by_last_completed_sample_plus_one_second=True,
        physical_model_calls=0)


if __name__=='__main__':
    require('--run' in sys.argv,'Inert. Explicit --run --mps-slot-clear and reviewed actual hashes required.')
    source,proof=derive_source()
    scope=dict(__name__='__main__',__file__=str(Path(__file__).resolve()),
        _neural=types.SimpleNamespace(**dict(globals())),_neural_watch_proof=proof)
    exec(compile(source,str(Path(__file__).resolve())+':exact-qualified-resource-loop','exec'),scope)
