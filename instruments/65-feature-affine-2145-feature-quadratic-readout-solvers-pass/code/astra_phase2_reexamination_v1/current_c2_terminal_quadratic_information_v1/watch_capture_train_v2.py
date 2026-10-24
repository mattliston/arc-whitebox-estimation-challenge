"""DRAFT: source-inverse fixed17 continuation of the preserved H15 monitor.

No inherited whole-stage PASS: V1 failed40 remains negative and fully charged.
"""
import hashlib
import importlib.util
from pathlib import Path
import types

HERE = Path(__file__).resolve().parent
PRESERVED_WATCH = HERE/'watch_capture_train.py'
PRESERVED_WATCH_SHA = 'a00884a5d3236892a8a652dd1af235ee457fdbb5ffd2c94739624253517d246e'
PRIOR_TRAIN_ELAPSED = 243.16458587441593
SEQUENCE = (40,41,43,44,45,46,47,49,50,51,52,53,54,55,56,57,58)
_OLD = None


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def owner():
    global _OLD
    if _OLD is None:
        require(hashlib.sha256(PRESERVED_WATCH.read_bytes()).hexdigest() == PRESERVED_WATCH_SHA,
            'Preserved exact V1 resource adapter')
        spec = importlib.util.spec_from_file_location('h15_preserved_train_watch',PRESERVED_WATCH)
        _OLD = importlib.util.module_from_spec(spec); spec.loader.exec_module(_OLD)
    return _OLD


def worker_args(args, ordinal=None):
    return types.SimpleNamespace(source_sha256=args.worker_sha256,watcher_sha256=args.source_sha256,
        protocol_sha256=args.protocol_sha256,watch_spec_sha256=args.spec_sha256,ordinal=ordinal)


def child_command(args, ordinal):
    return [str(HERE.parent.parent/'data_gate/.venv/bin/python'),'-B','-u',
        str(HERE/'capture_train_v2.py'),'--run','--ordinal',str(ordinal),
        '--source-sha256',args.worker_sha256,'--watcher-sha256',args.source_sha256,
        '--protocol-sha256',args.protocol_sha256,'--watch-spec-sha256',args.spec_sha256]


def row_manifest(h, record, result_path, seals, train_ordinal):
    row = owner().row_manifest(h,record,result_path,seals,train_ordinal)
    row.update(initialization_flops=record['initialization_flops'],
        watch_folder=str(HERE/('capture_qualification_watch_v1' if train_ordinal == 0 else 'train_capture_watch_v2')),
        watch_scope='positive_qualification' if train_ordinal == 0 else 'positive_v2_continuation',
        prior_failure_unchanged=True)
    return row


def failed_child_evidence(result, child_folder, out, read, sha):
    """Metadata-only failure ledger; never invent a context or positive child."""
    if result.get('passed') or not (child_folder/'result.json').is_file():
        return
    if any(x['result_path'] == str(child_folder/'result.json') for x in result.get('children',[])):
        return
    files = [child_folder/n for n in ('result.json','completion.json','finalization.json')]
    if not all(p.is_file() for p in files):
        result['failed_child_seals_incomplete'] = True
        return
    r,e,f = map(read,files); ds = [sha(p) for p in files]
    require(e['result_sha256'] == f['result_sha256'] == ds[0] and f['completion_sha256'] == ds[1],
        'Actual failed-child links, without relabeling polarity')
    for p,d in zip(files,ds): result['artifacts'][str(p)] = d
    ordinal = r.get('ordinal')
    require(type(ordinal) is int and 0 <= ordinal < 17 and r['ranks'] == [SEQUENCE[ordinal]],
        'Only actual current fixed child evidence')
    totals = {k:sum(x['flops'] for x in r.get(k,[])) for k in ('trials','input_preparation','replay_bills')}
    result['failed_child_observation'] = dict(ordinal=ordinal,rank=SEQUENCE[ordinal],worker_seals=ds,
        passed=r['passed'],completed=r['completed'],closed_model_contexts=len(r.get('trials',[])),
        closed_input_contexts=len(r.get('input_preparation',[])),closed_replay_contexts=len(r.get('replay_bills',[])),
        closed_bill_research_flops=sum(totals.values()),closed_bill_inventory_only=True,
        unclosed_context_cost_not_claimed=True,not_a_qualified_row=True)
    for p,d in r.get('artifacts',{}).items():
        require(sha(p) == d, 'Failed child saved artifact remains immutable')
        result['artifacts'][p] = d
    log = out/f'worker_{ordinal:02d}.log'
    if log.is_file(): result['artifacts'][str(log)] = sha(log)


def derive_source():
    old = owner(); original,old_proof = old.derive_source()
    edits = [
        ('CAP, RSS, PROGRESS = 3000.,8_000_000_000,120.',
         'CAP, RSS, PROGRESS = 3000.-PRIOR_TRAIN_ELAPSED,8_000_000_000,120.',1),
        ("HERE/'watch_capture_train.py':args.source_sha256,HERE/'capture_train.py':args.worker_sha256,\n        HERE/'TRAIN_CAPTURE_PROTOCOL.md':args.protocol_sha256,HERE/'TRAIN_CAPTURE_WATCH_SPEC.md':args.spec_sha256,\n        OWNER:OWNER_SHA,BASE:BASE_SHA,TRANSPORT:TRANSPORT_SHA",
         "HERE/'watch_capture_train_v2.py':args.source_sha256,HERE/'capture_train_v2.py':args.worker_sha256,\n        HERE/'TRAIN_CAPTURE_V2_PROTOCOL.md':args.protocol_sha256,HERE/'TRAIN_CAPTURE_V2_SPEC.md':args.spec_sha256,\n        OWNER:OWNER_SHA,BASE:BASE_SHA,PRESERVED_WATCH:PRESERVED_WATCH_SHA,TRANSPORT:TRANSPORT_SHA",1),
        ("HERE/'train_capture_v1',HERE/'train_capture_watch_v1'",
         "HERE/'train_capture_v2',HERE/'train_capture_watch_v2'",1),
        ("capture_root/'row_00_rank0036'","capture_root/'row_00_rank0040'",1),
        ("'current_c2_terminal_quadratic_train_capture_watch_v1'",
         "'current_c2_terminal_quadratic_train_capture_watch_v2'",1),
        ("'current_c2_terminal_quadratic_train_capture_launch_v1'",
         "'current_c2_terminal_quadratic_train_capture_launch_v2'",1),
        ("'current_c2_terminal_quadratic_train_capture_manifest_v1'",
         "'current_c2_terminal_quadratic_train_capture_manifest_v2'",1),
        ("probe = load(HERE/'capture_train.py','current_c2_fixed_train_capture_api',args.worker_sha256)",
         "probe = load(HERE/'capture_train_v2.py','current_c2_fixed_train_capture_v2_api',args.worker_sha256)",1),
        ("result['qualification_seals'] = context['qualification_seals']",
         "result['qualification_seals'] = context['qualification_seals']\n        result['continuation_history'] = context['continuation_history']\n        result['binding_source_price'] = context['binding_source_price']\n        result['prior_failure_unchanged'] = True\n        result['watch_folder'] = str(out)\n        result['aggregate_train_cap_s'] = 3000.\n        result['prior_train_elapsed_s'] = PRIOR_TRAIN_ELAPSED",1),
        ("context['qualification_seals']['capture_qualification_v1'],0)]",
         "context['qualification_seals']['capture_qualification_v1'],0)]+context['continuation_history']['rows']",1),
        ('index < 19 and item[\'call\'] == 0','index < 17 and item[\'call\'] == 0',1),
        ('probe.owner(),record,files[0],digests,ordinal+1)',
         'probe.owner(),record,files[0],digests,ordinal+3)',1),
        ("require(record['qualification_seals'] == context['qualification_seals'], 'Same actual cold35 qualification')",
         "require(record['qualification_seals'] == context['qualification_seals']\n                and record['continuation_history'] == context['continuation_history']\n                and record['binding_source_price'] == context['binding_source_price'], 'Same actual qualification, failure history and source price')",1),
        ("result['retired_calls'] == len(result['children']) == 19\n            and [x['rank'] for x in result['rows']] == [35]+list(SEQUENCE), 'Exactly20 primary rows, no repeat example'",
         "result['retired_calls'] == len(result['children']) == 17\n            and [x['rank'] for x in result['rows']] == [35,36,38]+list(SEQUENCE), 'Exactly20 primary rows; failed40 and qualification repeat excluded'",1),
        ("train=[35]+list(SEQUENCE),rows=result['rows']", "train=[35,36,38]+list(SEQUENCE),rows=result['rows']",1),
        ('new_model_calls=19,total_model_calls_including_qualification_repeat=21,',
         'new_model_calls=17,total_model_calls_including_qualification_repeat=22,\n            total_model_calls_includes_failed40=True,total_successful_model_calls=21,failed_model_calls=1,\n            reused_positive_children=2,watch_folder=str(out),prior_failure_unchanged=True,\n            continuation_history=context[\'continuation_history\'],\n            prior_train_research_flops=context[\'continuation_history\'][\'prior_train_research_flops\'],\n            prior_train_elapsed_s=PRIOR_TRAIN_ELAPSED,aggregate_train_cap_s=3000.,',1),
        ("manifest['qualification_research_flops']+manifest['new_research_flops']",
         "manifest['qualification_research_flops']+manifest['prior_train_research_flops']+manifest['new_research_flops']",1),
        ("        write(capture_root/'manifest.json',manifest)",
         "        require(manifest['prior_train_research_flops']+manifest['new_research_flops']\n            <= 19*(probe.owner().BUDGET+16777216+probe.owner().REPLAY_CAP),\n            'Original19-row maximum complete research allowance includes the failed attempt')\n        write(capture_root/'manifest.json',manifest)",1),
        ('full_model_calls=19,retired_call_count=19,', 'full_model_calls=17,retired_call_count=17,',1),
        ('all19_children_reaped=True,all20_primary_captures_sealed=True,',
         'all17_children_reaped=True,all20_primary_captures_sealed=True,',1),
        ('initialization_flops_per_cold_call=1944970298,parameter_updates=0,fit_performed=False,',
         "initialization_flops_per_cold_call=[x['initialization_flops'] for x in result['rows']],\n            parameter_updates=0,fit_performed=False,",1),
        ('absolute = min(previous+1.,began+CAP,last_progress+PROGRESS)',
         'absolute = min(previous+1.,began+CAP,last_progress+PROGRESS,\n            float(\'inf\') if child is None else child_started+ROW_CAP)',1),
        ("        for name in ('started.json','worker.log','samples.jsonl'):",
         "        try:\n            failed_child_evidence(result,child_folder,out,read,sha)\n        except BaseException:\n            result.update(passed=False,completed=False,failure_ledger_error=traceback.format_exc())\n        for name in ('started.json','worker.log','samples.jsonl'):",1),
    ]
    source,proof = old.splice(original,edits,'V2 fixed17/prefix/cost metadata; original resource loop')
    proof.update(preserved_watch_sha256=PRESERVED_WATCH_SHA,preserved_effective_source=old_proof,
        candidate_and_paid_loop_unchanged=True,original_aggregate_3000s_counts_prior_failure=True,
        reused_only_positive36_38=True,failed40_rerun_once_not_promoted=True)
    return source,proof


def main():
    old = owner(); base = old.owner(); source,proof = derive_source()
    env = dict(__name__='_h15_train_v2_watch_body',__file__=str(Path(__file__).resolve()),
        OWNER=base.OWNER,OWNER_SHA=base.OWNER_SHA,BASE=old.BASE,BASE_SHA=old.BASE_SHA,ROW_CAP=180.,
        PRESERVED_WATCH=PRESERVED_WATCH,PRESERVED_WATCH_SHA=PRESERVED_WATCH_SHA,
        PRIOR_TRAIN_ELAPSED=PRIOR_TRAIN_ELAPSED,SEQUENCE=SEQUENCE,WATCH_SOURCE_PROOF=proof,
        worker_args=worker_args,child_command=child_command,foreign_servers=base.foreign_servers,
        row_manifest=row_manifest,failed_child_evidence=failed_child_evidence)
    exec(compile(source,'<whole-inverse-h15-fixed17-watch-v2>','exec'),env)
    return env['main']()


if __name__ == '__main__':
    raise SystemExit(main())
