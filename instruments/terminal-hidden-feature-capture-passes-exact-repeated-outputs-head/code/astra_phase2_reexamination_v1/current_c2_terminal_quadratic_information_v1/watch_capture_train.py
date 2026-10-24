"""Inert fixed19 serial fresh-child expansion of the qualified H15 monitor."""
import hashlib
import importlib.util
from pathlib import Path
import textwrap
import types

HERE = Path(__file__).resolve().parent
BASE = HERE/'watch_capture.py'
BASE_SHA = '1ffc1f7b3352c981b2f7776619af30347aec69394317aa9f21d841053b61798d'
ROW_CAP = 180.
SEQUENCE = (36,38,40,41,43,44,45,46,47,49,50,51,52,53,54,55,56,57,58)


def require(ok, message):
    if not ok: raise RuntimeError(message)


def owner():
    require(hashlib.sha256(BASE.read_bytes()).hexdigest() == BASE_SHA, 'Actual qualified monitor source')
    spec = importlib.util.spec_from_file_location('h15_qualified_watch_owner',BASE)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def worker_args(args, ordinal=None):
    return types.SimpleNamespace(source_sha256=args.worker_sha256,watcher_sha256=args.source_sha256,
        protocol_sha256=args.protocol_sha256,watch_spec_sha256=args.spec_sha256,ordinal=ordinal)


def child_command(args, ordinal):
    offpy = HERE.parent.parent/'data_gate/.venv/bin/python'
    return [str(offpy),'-B','-u',str(HERE/'capture_train.py'),'--run','--ordinal',str(ordinal),
        '--source-sha256',args.worker_sha256,'--watcher-sha256',args.source_sha256,
        '--protocol-sha256',args.protocol_sha256,'--watch-spec-sha256',args.spec_sha256]


def row_manifest(h, record, result_path, seals, train_ordinal):
    trial = record['trials'][0]; capture = record['captures'][0]; m = h.read(capture['path'])
    rank = trial['rank']
    if train_ordinal == 0:
        fixture = h.read(h.ROW/'row_manifest.json')['files']['weights.npy']
        weights,weight_hash = str(h.ROW/'weights.npy'),fixture['sha256']
        seed = h.read(h.ROW/'row_manifest.json')['mlp_seed']
    else:
        fixture = record['fixtures'][str(rank)]
        weights,weight_hash,seed = fixture['weights'],fixture['weights_sha256'],fixture['mlp_seed']
    return dict(train_ordinal=train_ordinal,rank=rank,source_call=0,fresh_cold=True,
        result_path=str(result_path),worker_seals=seals,
        capture_manifest=capture['path'],capture_manifest_sha256=capture['sha256'],
        capture_npz=m['path'],capture_npz_sha256=m['sha256'],arrays=m['arrays'],
        model_bill=str(Path(trial['output']).parent/'call_00_bill.json'),
        model_bill_sha256=record['artifacts'][str(Path(trial['output']).parent/'call_00_bill.json')],
        own_cold_flops=trial['flops'],input_bill=trial['input_bill'],input_bill_sha256=trial['input_bill_sha256'],
        replay_bill=record['replay_bills'][0],weights=weights,weights_sha256=weight_hash,mlp_seed=seed,
        head_replay=m['head_replay'],candidate_sha256=record['candidate_sha256'],
        prior_current_c2_full_output_control=(train_ordinal == 0))


def splice(original, edits, label):
    changed,spans = original,[]
    for before,after,count in edits:
        require(changed.count(before) == count, 'Exact '+label+' splice: '+before[:90])
        cursor = 0
        for _ in range(count):
            at = changed.index(before,cursor); spans.append((at,before,after))
            changed = changed[:at]+after+changed[at+len(before):]; cursor = at+len(after)
    inverse = changed
    for at,before,after in reversed(spans):
        require(inverse[at:at+len(after)] == after, 'Exact recorded-position inverse')
        inverse = inverse[:at]+before+inverse[at+len(after):]
    require(inverse == original, 'Whole qualified '+label+' inverse')
    return changed,dict(original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),whole_source_inverse=True,splices=len(spans))


def derive_source():
    base = owner(); original,base_proof = base.derive_source()
    begin = original.index('        argv = child_command(args)')
    end = original.index('\n        observe()\n    except BaseException:',begin)
    child_body = original[begin:end]
    positive_begin = child_body.index('        result.update(exact_head_replay=True')
    child_body_prefix = child_body[:positive_begin]
    original_positive = child_body[positive_begin:]
    child_edits = [
        ('argv = child_command(args)','argv = child_command(args,ordinal)',1),
        ("log = (out/'worker.log').open('x'); reader = (out/'worker.log').open('r')",
         "log_path = out/f'worker_{ordinal:02d}.log'\n        log = log_path.open('x'); reader = log_path.open('r')",1),
        ('        child = subprocess.Popen(argv', '        child_started = time.monotonic()\n        child = subprocess.Popen(argv',1),
        ("result['child_pid'],result['command'] = child.pid,argv", "result['current_child_pid'] = child.pid\n        result['commands'].append(dict(ordinal=ordinal,rank=rank,pid=child.pid,argv=argv))",1),
        ("        write(out/'started.json',result)\n",'',1),
        ("child.wait(timeout=1) == 0 and result['retired_calls'] == 2,'Two complete calls and exit0'",
         "child.wait(timeout=1) == 0 and result['retired_calls'] == ordinal+1,'One fresh child and exact cumulative retired count'",1),
        ("final['elapsed_s'] < CAP,'Actual worker three seals'", "final['elapsed_s'] < ROW_CAP,'Actual child180s three seals'",1),
        ("record['full_model_calls'] == record['retired_call_count'] == 2", "record['full_model_calls'] == record['retired_call_count'] == 1",1),
        ("record['ranks'] == [35,35]", "record['ranks'] == [rank] and record['ordinal'] == ordinal",1),
        ("record['all47_selectors_qualified'] and record['repeat_byte_exact'] and record['own_complete_bills']",
         "record['all47_selectors_qualified'] and record['repeat_byte_exact'] is False and record['own_complete_bills']",1),
    ]
    child_body_prefix,child_proof = splice(child_body_prefix,child_edits,'per-child qualified monitor')
    child_tail = """        require(record['qualification_seals'] == context['qualification_seals'], 'Same actual cold35 qualification')
        result['children'].append(dict(ordinal=ordinal,rank=rank,result_path=str(files[0]),
            result_sha256=digests[0],completion_sha256=digests[1],finalization_sha256=digests[2],
            research_flops=record['research_flops'],full_model_calls=1,retired_call_count=1,
            worker_elapsed_s=record['elapsed_s'],child_reaped=True))
        result['rows'].append(row_manifest(probe.owner(),record,files[0],digests,ordinal+1))
        for path,digest in zip(files,digests): result['artifacts'][str(path)] = digest
        log.close(); reader.close(); log = reader = None
        result['artifacts'][str(log_path)] = sha(log_path,tick)
        require(time.monotonic()-child_started < ROW_CAP, 'Whole fresh child180s including reaping/verification')
        child = None; pending = ''; observe()
"""
    loop = """        launch = dict(schema='current_c2_terminal_quadratic_train_capture_launch_v1',
            ranks=list(SEQUENCE),qualification_seals=context['qualification_seals'],
            worker_authority={k:getattr(worker_args(args),k) for k in
                ('source_sha256','watcher_sha256','protocol_sha256','watch_spec_sha256')},
            source_bindings=context['source_bindings'],input_bindings=context['input_bindings'],
            truth_accessed=False,fit_performed=False,automatic_next_stage=False)
        write(capture_root/'launch.json',launch)
        result['artifacts'][str(capture_root/'launch.json')] = sha(capture_root/'launch.json',tick)
        q = read(HERE/'capture_qualification_v1/result.json')
        result['rows'] = [row_manifest(probe.owner(),q,HERE/'capture_qualification_v1/result.json',
            context['qualification_seals']['capture_qualification_v1'],0)]
        write(out/'started.json',result)
        for ordinal,rank in enumerate(SEQUENCE):
            child_folder = probe.folder(ordinal)
"""+textwrap.indent(child_body_prefix+child_tail,'    ')+"""        require(result['retired_calls'] == len(result['children']) == 19
            and [x['rank'] for x in result['rows']] == [35]+list(SEQUENCE), 'Exactly20 primary rows, no repeat example')
        for group in ('source_bindings','input_bindings','artifacts'):
            for path,digest in result[group].items():
                require(sha(path,tick) == digest, 'Final complete fixed19 inventory unchanged')
        manifest = dict(schema='current_c2_terminal_quadratic_train_capture_manifest_v1',
            candidate_sha256=result['candidate_sha256'],train=[35]+list(SEQUENCE),rows=result['rows'],
            qualification_seals=context['qualification_seals'],child_seals=result['children'],
            new_model_calls=19,total_model_calls_including_qualification_repeat=21,
            qualification_research_flops=context['qualification_research_flops'],
            new_research_flops=sum(x['research_flops'] for x in result['children']),
            source_bindings=result['source_bindings'],input_bindings=result['input_bindings'],
            artifacts=dict(result['artifacts']),truth_accessed=False,fit_performed=False,score_measured=False,
            exploratory_TRAIN_network_CV_only=True,automatic_next_stage=False)
        manifest['total_research_flops'] = manifest['qualification_research_flops']+manifest['new_research_flops']
        write(capture_root/'manifest.json',manifest)
        result['manifest_path'] = str(capture_root/'manifest.json')
        result['manifest_sha256'] = sha(capture_root/'manifest.json',tick)
        result['artifacts'][result['manifest_path']] = result['manifest_sha256']
        result.update(passed=True,completed=True,child_execution_completed=True,whole_owned_tree_retired=True,
            hashes_unchanged=True,sources_unchanged=True,actual_completed_samples=True,
            foreign_numerical_workers_checked=True,full_model_calls=19,retired_call_count=19,
            all19_children_reaped=True,all20_primary_captures_sealed=True,exact_head_replay=True,
            all16_domains_qualified=True,all47_selectors_qualified=True,own_complete_bills=True,
            all_parameter_and_estimator_owners_retired=True,actual_production_backend_qualified=True,
            initialization_flops_per_cold_call=1944970298,parameter_updates=0,fit_performed=False,
            automatic_next_stage=False,training_expansion_authorized=False,submission_authorized=False,
            research_flops=manifest['total_research_flops'])
"""
    edits = [
        ('CAP, RSS, PROGRESS = 300.,8_000_000_000,120.', 'CAP, RSS, PROGRESS = 3000.,8_000_000_000,120.',1),
        ("HERE/'watch_capture.py':args.source_sha256,HERE/'capture.py':args.worker_sha256,\n        HERE/'PROTOCOL.md':args.protocol_sha256,HERE/'WATCH_SPEC.md':args.spec_sha256,\n        OWNER:OWNER_SHA,TRANSPORT:TRANSPORT_SHA",
         "HERE/'watch_capture_train.py':args.source_sha256,HERE/'capture_train.py':args.worker_sha256,\n        HERE/'TRAIN_CAPTURE_PROTOCOL.md':args.protocol_sha256,HERE/'TRAIN_CAPTURE_WATCH_SPEC.md':args.spec_sha256,\n        OWNER:OWNER_SHA,BASE:BASE_SHA,TRANSPORT:TRANSPORT_SHA",1),
        ("    child_folder,out = HERE/'capture_qualification_v1',HERE/'capture_qualification_watch_v1'",
         "    capture_root,out = HERE/'train_capture_v1',HERE/'train_capture_watch_v1'\n    child_folder = capture_root/'row_00_rank0036'",1),
        ("not child_folder.exists() and not out.exists()", "not capture_root.exists() and not out.exists()",1),
        ('    out.mkdir()', '    out.mkdir(); capture_root.mkdir()',1),
        ("'current_c2_terminal_quadratic_capture_qualification_watch_v1'", "'current_c2_terminal_quadratic_train_capture_watch_v1'",1),
        ("retired_calls=0,ranks=[35,35],completed_sample_intervals_s=[],markers=[])",
         "retired_calls=0,ranks=list(SEQUENCE),completed_sample_intervals_s=[],markers=[],children=[],commands=[])",1),
        ("    child=log=reader=stream=None", "    child=log=reader=stream=None; child_started=None",1),
        ("require(now-began < CAP and now-last_progress < PROGRESS,'Whole300s/next retired call120s')",
         "require(now-began < CAP and now-last_progress < PROGRESS,'Whole3000s/next retired call120s')\n        require(child is None or now-child_started < ROW_CAP,'Fresh child180s whole lifetime')",1),
        ("index < 2 and item['call'] == index and item['rank'] == 35,'Exact ordered35/35'",
         "index < 19 and item['call'] == 0 and item['ordinal'] == index and item['rank'] == SEQUENCE[index],'Exact fixed19 ordinal roster'",1),
        ("path = child_folder/f'call_{index:02d}_complete.json'", "path = child_folder/'call_00_complete.json'",1),
        ("record['call'] == index and record['rank'] == 35 and record['context_retired']",
         "record['call'] == 0 and record['ordinal'] == index and record['rank'] == SEQUENCE[index] and record['context_retired']",1),
        ("child_folder/f'call_{index:02d}_{suffix}.json'", "child_folder/f'call_00_{suffix}.json'",1),
        ("sha256=digest,call=index,rank=35", "sha256=digest,call=0,ordinal=index,rank=SEQUENCE[index]",1),
        ("probe = load(HERE/'capture.py','current_c2_passive_capture_api',args.worker_sha256)",
         "probe = load(HERE/'capture_train.py','current_c2_fixed_train_capture_api',args.worker_sha256)",1),
        ("result['control_seals'] = context['control_seals']", "result['qualification_seals'] = context['qualification_seals']",1),
        (original[begin:end],loop,1),
    ]
    source,proof = splice(original,edits,'serial19 completed-observation resource loop')
    proof.update(qualified_watch_sha256=BASE_SHA,qualified_effective_source=base_proof,
        child_body=child_proof,replaced_single_positive_tail_sha256=hashlib.sha256(original_positive.encode()).hexdigest(),
        fixed19_fresh_children=True,whole3000s_tree8GB_zero_swap_120s_progress_1s_samples_and_tail=True)
    return source,proof


def main():
    base = owner(); source,proof = derive_source()
    env = dict(__name__='_h15_fixed19_watch_body',__file__=str(Path(__file__).resolve()),
        OWNER=base.OWNER,OWNER_SHA=base.OWNER_SHA,BASE=BASE,BASE_SHA=BASE_SHA,ROW_CAP=ROW_CAP,
        SEQUENCE=SEQUENCE,WATCH_SOURCE_PROOF=proof,worker_args=worker_args,child_command=child_command,
        foreign_servers=base.foreign_servers,row_manifest=row_manifest)
    exec(compile(source,'<qualified-monitor-fixed19-fresh-children>','exec'),env)
    return env['main']()


if __name__ == '__main__':
    raise SystemExit(main())
