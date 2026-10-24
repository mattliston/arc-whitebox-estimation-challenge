"""Inert stage adapter of the original v29 actual-sample/reap watchdog.

TRAIN and CHECK are separate invocations and fresh numerical processes.
No numerical module, array decoder, target reader or trainer is imported here.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
BASE_OWNER = HERE/'watch.py'
BASE_SHA = 'f7783cd003fdca3bd625ffa62733fbb12655704998cb2547e00dcee736b8e0b6'
BASE_WORKER_SHA = '551644da1dd6a0d7fe193df03894e889e8e6589bb839c65da64bf2917acb271a'
SHARED_OWNER = HERE.parent/'current_comp_shared_old384_v1/watch_probe.py'
SHARED_SHA = '7ade1c9025a2932e7929498062f894b72a3817d50027e7c5d1cfbf22a1f0fd18'
TRAIN = (35,36,38,40,41,43,44,45,46,47,49,50,51,52,53,54,55,56,57,58)
CHECK = (37,39,42,48,37)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def configuration(args):
    require(args.stage in ('train','check'), 'Explicit TRAIN or fresh-process CHECK')
    return (2460.,TRAIN) if args.stage == 'train' else (660.,CHECK)


def prerequisite_cli(args):
    values = ['--qualification-sha256',args.qualification_sha256,
        '--qualification-watch-sha256',args.qualification_watch_sha256]
    if args.stage == 'check':
        values += ['--train-sha256',args.train_sha256,'--train-watch-sha256',args.train_watch_sha256]
    return values


def seal_joint(probe, args, result, record, digests, admitted, out, tick):
    """Metadata only, after current child.wait/cleanup; authority needs final watch seals."""
    if args.stage == 'train':
        result['all25_capture_inventory_sealed'] = False
        return
    train = admitted['train']
    require(train is not None and train['worker_pid'] != record['worker_pid'], 'Two distinct fresh numerical processes')
    calls, metadata = {},{}
    for stage,value in (('train',train),('check',record)):
        folder = HERE/f'{stage}_capture_v1'
        for path,digest in value['artifacts'].items():
            destination = calls if Path(path).parent == folder and Path(path).name.startswith('call_') else metadata
            probe.add(destination,path,digest,tick)
    require(len(calls) == 200 and len(metadata) == 4, 'Exactly160 TRAIN plus40 CHECK call artifacts and four stage metadata files')
    fixtures = dict(train['fixture_bindings'])
    for rank,value in record['fixture_bindings'].items():
        require(rank not in fixtures, 'Disjoint prescribed20 TRAIN and4 CHECK fixtures')
        fixtures[rank] = value
    require(len(fixtures) == 24, 'Exactly24 unique physical inputs')
    seals = dict(admitted['seals'])
    seals['check'] = dict(zip(('result_sha256','completion_sha256','finalization_sha256'),digests))
    packet = dict(schema='current_v29_temporal_joint_capture_inventory_v1',passed=True,completed=True,
        candidate_sha256=record['candidate_sha256'],composition_proof=record['composition_proof'],
        train_ranks=list(TRAIN),check_ranks=list(CHECK),full_model_calls=25,setup_calls=2,
        train_worker_pid=train['worker_pid'],check_worker_pid=record['worker_pid'],
        two_serial_fresh_processes=True,train_prior_process_retired=True,current_check_process_retired=True,
        prerequisite_seals=seals,source_bindings=dict(result['source_bindings']),
        input_bindings=dict(result['input_bindings']),fixture_bindings=fixtures,
        call_artifacts=calls,stage_metadata_artifacts=metadata,
        train_capture_records=train['capture_records'],check_capture_records=record['capture_records'],
        train_complete_bills=train['trials'],check_complete_bills=record['trials'],
        train_repeat_applicable=False,check_cold_qualification_replay=record['cold_qualification_replay'],
        truth_accessed=False,score_measured=False,head_attached=False,parameter_updates=0,
        requires_final_check_watch_three_seals=True)
    path = out/'joint_capture_inventory.json'
    with path.open('x') as stream:
        json.dump(packet,stream,indent=2,sort_keys=True,allow_nan=False); stream.write('\n')
    digest = probe.sha(path,tick)
    result['artifacts'][str(path)] = digest
    result.update(all25_capture_inventory_sealed=True,joint_capture_inventory=str(path),
        joint_capture_inventory_sha256=digest,two_serial_fresh_processes=True)


def derive_source():
    require(hashlib.sha256(BASE_OWNER.read_bytes()).hexdigest() == BASE_SHA, 'Frozen original v29 watchdog adapter')
    spec = importlib.util.spec_from_file_location('v29_original_watch_metadata',BASE_OWNER)
    base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
    original,prior = base.derive_source()
    old_owners = """    own = {HERE/'watch.py':args.source_sha256,HERE/'worker.py':args.worker_sha256,
        HERE/'ACQUISITION_PROTOCOL.md':args.protocol_sha256,HERE/'WATCH_SPEC.md':args.spec_sha256,
        HERE/'adapter.py':args.adapter_sha256,HERE/'PROTOCOL.md':args.adapter_protocol_sha256,
        TRANSPORT:TRANSPORT_SHA,OWNER:OWNER_SHA}
"""
    new_owners = """    own = {HERE/'capture_watch.py':args.source_sha256,HERE/'capture_worker.py':args.worker_sha256,
        HERE/'CAPTURE_PROTOCOL.md':args.protocol_sha256,HERE/'CAPTURE_WATCH_SPEC.md':args.spec_sha256,
        HERE/'adapter.py':args.adapter_sha256,HERE/'PROTOCOL.md':args.adapter_protocol_sha256,
        TRANSPORT:TRANSPORT_SHA,OWNER:OWNER_SHA,
        _capture_watch.BASE_OWNER:_capture_watch.BASE_SHA,HERE/'worker.py':_capture_watch.BASE_WORKER_SHA}
"""
    old_flags = """            and record['full_model_calls'] == record['retired_call_count'] == 2
            and record['schema'] == 'current_v29_temporal_acquisition_v1'
            and record['ranks'] == [37,37] and record['all16_raw_domains_qualified']
            and record['all16_features_qualified'] and record['feature_columns'] == 30
            and record['repeat_byte_exact'] and record['feature_repeat_byte_exact'] and record['own_complete_bills']
            and record['setup_calls'] == 1 and record['setup_seed'] == 0 and record['setup_owned']
            and record['estimator_retired'] and record['head_attached'] is False and record['feedback_enabled'] is False
"""
    new_flags = """            and record['full_model_calls'] == record['retired_call_count'] == len(SEQUENCE)
            and record['schema'] == 'current_v29_temporal_capture_v1' and record['stage'] == args.stage
            and record['ranks'] == list(SEQUENCE) and record['all16_raw_domains_qualified']
            and record['all16_features_qualified'] and record['feature_columns'] == 30
            and record['own_complete_bills'] and record['setup_calls'] == 1
            and record['setup_seed'] == 0 and record['setup_owned'] and record['estimator_retired']
            and record['head_attached'] is False and record['feedback_enabled'] is False
"""
    edits = (
        ("    parser.add_argument('--numerical-slot-clear',action='store_true')",
         "    parser.add_argument('--numerical-slot-clear',action='store_true')\n"
         "    parser.add_argument('--stage',choices=('train','check'))\n"
         "    for key in ('qualification','qualification-watch','train','train-watch'):\n"
         "        parser.add_argument('--'+key+'-sha256')"),
        ('    began = previous = last_progress = time.monotonic()',
         '    CAP, SEQUENCE = _capture_watch.configuration(args)\n    began = previous = last_progress = time.monotonic()'),
        (old_owners,new_owners),
        ("child_folder,out = HERE/'acquisition_v1',HERE/'acquisition_watch_v1'",
         "child_folder,out = HERE/f'{args.stage}_capture_v1',HERE/f'{args.stage}_capture_watch_v1'"),
        ("schema='current_v29_temporal_acquisition_watch_v1'",
         "schema='current_v29_temporal_capture_watch_v1',stage=args.stage,capture_watch_source_proof=_capture_watch_proof"),
        ('retired_calls=0,ranks=[37,37],completed_sample_intervals_s=[],markers=[])',
         'retired_calls=0,ranks=list(SEQUENCE),completed_sample_intervals_s=[],markers=[])'),
        ("'Whole300s/next retired call120s'", "'Prospective stage whole clock/next retired call120s'"),
        ("require(index < 2 and item['call'] == index and item['rank'] == 37,'Exact ordered37/37')",
         "require(index < len(SEQUENCE) and item['call'] == index and item['rank'] == SEQUENCE[index],'Exact fixed stage roster')"),
        ("and record['call'] == index and record['rank'] == 37 and record['context_retired'],",
         "and record['call'] == index and record['rank'] == SEQUENCE[index] and record['context_retired'],"),
        ("dict(path=str(path),sha256=digest,call=index,rank=37)",
         "dict(path=str(path),sha256=digest,call=index,rank=SEQUENCE[index])"),
        ("probe = load(HERE/'worker.py','v29_passive_worker_api',args.worker_sha256)",
         "probe = load(HERE/'capture_worker.py','v29_expanded_passive_worker_api',args.worker_sha256)"),
        ('        probe.environment()', '        probe.environment()\n'
         '        worker_args = probe.worker_args(args)\n'
         '        require(probe.configuration(worker_args) == (CAP,SEQUENCE), \'Identical worker/watch stage roster and caps\')\n'
         '        for path,digest in probe.new_owners(worker_args).items():\n'
         '            probe.add(result[\'source_bindings\'],path,digest,tick)\n'
         '        admitted = probe.prerequisites(worker_args,result[\'source_bindings\'],result[\'input_bindings\'],tick)\n'
         '        result[\'prerequisite_seals\'] = admitted[\'seals\']\n'
         '        rows,result[\'fixture_bindings\'] = probe.fixture_rows(SEQUENCE,result[\'input_bindings\'],tick)\n'
         '        del rows\n'
         '        worker_source,result[\'capture_source_proof\'] = probe.derive_source()\n'
         '        del worker_source\n'
         '        tick()'),
        ("        result['composition_proof'] = proof\n        del text",
         "        result['composition_proof'] = proof\n"
         "        require(proof == admitted['qualification']['composition_proof']\n"
         "            and result['candidate_sha256'] == admitted['qualification']['candidate_sha256'], 'Original qualified physical source')\n"
         "        del text"),
        ("str(HERE/'worker.py'),'--run',", "str(HERE/'capture_worker.py'),'--run','--stage',args.stage,"),
        ("'--adapter-sha256',args.adapter_sha256,'--adapter-protocol-sha256',args.adapter_protocol_sha256]",
         "'--adapter-sha256',args.adapter_sha256,'--adapter-protocol-sha256',args.adapter_protocol_sha256]\n"
         "        argv += _capture_watch.prerequisite_cli(args)"),
        ("require(child.wait(timeout=1) == 0 and result['retired_calls'] == 2,'Two complete calls and exit0')",
         "require(child.wait(timeout=1) == 0 and result['retired_calls'] == len(SEQUENCE),'Exact stage calls and exit0')"),
        (old_flags,new_flags),
        ("            'Actual target-blind same-source own-cost native subject')\n        for group",
         "            'Actual target-blind same-source own-cost native subject')\n"
         "        probe.verify_expanded(record,args.stage,worker_args,admitted['qualification'],tick)\n"
         "        require(record['worker_pid'] == child.pid\n"
         "            and record['prerequisite_seals'] == result['prerequisite_seals']\n"
         "            and record['fixture_bindings'] == result['fixture_bindings']\n"
         "            and record['capture_source_proof'] == result['capture_source_proof'], 'Exact completed fresh-process capture lineage')\n"
         "        for group"),
        ("        for path,digest in result['source_bindings'].items():",
         "        _capture_watch.seal_joint(probe,args,result,record,digests,admitted,out,tick)\n"
         "        for path,digest in result['source_bindings'].items():"),
        ("        require([sha(p,tick) for p in files] == digests,'Seals immutable through verification')",
         "        require([sha(p,tick) for p in files] == digests,'Seals immutable through verification')\n"
         "        if args.stage == 'check':\n"
         "            require(sha(result['joint_capture_inventory'],tick) == result['joint_capture_inventory_sha256'],\n"
         "                'Joint target-blind inventory unchanged through final source/input verification')"),
        ('result.update(passed=True,completed=True,child_execution_completed=True,whole_owned_tree_retired=True,',
         'result.update(passed=True,completed=True,expanded_capture_verified=True,child_execution_completed=True,whole_owned_tree_retired=True,'),
    )
    changed = original
    for old,new in edits:
        require(changed.count(old) == 1, 'Unique expanded watchdog substitution '+old[:80])
        changed = changed.replace(old,new)
    inverse = changed
    for old,new in reversed(edits):
        require(inverse.count(new) == 1, 'Unique expanded watchdog inverse '+new[:80])
        inverse = inverse.replace(new,old)
    require(inverse == original, 'Whole original qualified v29 watchdog inverse')
    return changed,dict(parent=prior,owner_sha256=BASE_SHA,whole_source_inverse=True,edit_count=len(edits),
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        stage_caps_s=dict(train=2460,check=660),unchanged_live_sampler_cleanup_and_three_seal_loop=True,
        unchanged_8GB_zero_swap_120s_progress_1s_actual_observations=True)


if __name__ == '__main__':
    if '--run' not in sys.argv:
        raise SystemExit('Inert. Explicit --run --numerical-slot-clear --stage and reviewed hashes are required.')
    source,proof = derive_source()
    scope = dict(__name__='__main__',__file__=str(Path(__file__).resolve()),OWNER=SHARED_OWNER,OWNER_SHA=SHARED_SHA,
        _capture_watch=types.SimpleNamespace(**globals()),_capture_watch_proof=proof)
    exec(compile(source,str(Path(__file__).resolve())+':expanded-qualified-watch','exec'),scope)
