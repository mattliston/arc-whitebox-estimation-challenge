"""Inert fixed19 fresh-cold continuation of the qualified passive H15 capture.

Only the outer fixed-roster supervisor may launch these ordinal-only children.
No targets, fitting, altered participant arithmetic, repeat or resume route.
"""
import argparse
import ast
import hashlib
import importlib.util
from pathlib import Path
import resource
import sys
import time
import types

HERE = Path(__file__).resolve().parent
OWNER = HERE/'capture.py'
QUALIFIED_OWNERS = {
    'capture.py':'95287251980b79d377906c067366714f2a35edd396da00646ebfd37300c98dbe',
    'watch_capture.py':'1ffc1f7b3352c981b2f7776619af30347aec69394317aa9f21d841053b61798d',
    'PROTOCOL.md':'b913306c9d103d3ea7f99e3a95ebe6df85cf28f0cfecc432b3e2430aeca882f6',
    'WATCH_SPEC.md':'ae11acbcbe927e9d03ba40b0f2d232f3f6f4bb29c00de785d01318f16435bf56',
}
OWNER_SHA = QUALIFIED_OWNERS['capture.py']
QUALIFICATION = {
    'capture_qualification_v1':[
        'e05c279198f2ce989d9c130e89b1869358a8b46004b90f93591e25a4796717ac',
        '8779b439d60212fdd7e283be3f21827eb525178a5fb566f832f6aa733e580c8b',
        'f43e644fba8c51759bc2f379768e88ee94b3f71d7db9aa1bf09b962da34f1ba2'],
    'capture_qualification_watch_v1':[
        'b4cf07afbff74b9140fc120765902d1662273f3a81205085adb0a00ee3fd641a',
        '87ab7877a2a571e80082261dd1a1a3b5039ede3d00eee453a40bae3ca926c2ef',
        'bda743d81feef5f622b9cb3e5eb45bdcafc155a72aa943a1f49ac5e39cf7b752'],
}
SEQUENCE = (36,38,40,41,43,44,45,46,47,49,50,51,52,53,54,55,56,57,58)
TRAIN = (35,)+SEQUENCE
OUTPUT = HERE/'train_capture_v1'
SCHEMA = 'current_c2_terminal_quadratic_train_capture_row_v1'
CAP, RSS = 180.,8_000_000_000


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def owner():
    require(hashlib.sha256(OWNER.read_bytes()).hexdigest() == OWNER_SHA, 'Qualified capture source before inert import')
    spec = importlib.util.spec_from_file_location('h15_qualified_capture_owner',OWNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def folder(ordinal):
    require(type(ordinal) is int and 0 <= ordinal < 19, 'Fixed remaining TRAIN ordinal only')
    return OUTPUT/f'row_{ordinal:02d}_rank{SEQUENCE[ordinal]:04d}'


def splice(original, edits, label):
    changed,spans = original,[]
    for before,after,count in edits:
        require(changed.count(before) == count, 'Exact '+label+' splice: '+before[:90])
        cursor = 0
        for _ in range(count):
            at = changed.index(before,cursor); spans.append((at,before,after))
            changed = changed[:at]+after+changed[at+len(before):]; cursor = at+len(after)
    restored = changed
    for at,before,after in reversed(spans):
        require(restored[at:at+len(after)] == after, 'Exact positional source inverse')
        restored = restored[:at]+before+restored[at+len(after):]
    require(restored == original, 'Whole '+label+' byte inverse')
    return changed,dict(original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),whole_source_inverse=True,splices=len(spans))


def qualification(h, tick=None):
    """A direct actual completed boundary, not another ancestral admission."""
    sources,inputs,records = {},{},{}
    for name,digest in QUALIFIED_OWNERS.items():
        sources[str(HERE/name)] = digest
    h.bind(sources,tick)
    for name,expected in QUALIFICATION.items():
        paths = [HERE/name/f for f in ('result.json','completion.json','finalization.json')]
        hashes = [h.sha(p,tick) for p in paths]
        require(hashes == expected, 'Actual positive qualification three seals')
        r,e,f = map(h.read,paths)
        require(all(x.get('passed') is True and x.get('completed') is True for x in (r,e,f))
            and e['result_sha256'] == f['result_sha256'] == hashes[0]
            and f['completion_sha256'] == hashes[1], 'Linked positive qualification')
        inputs.update({str(p):d for p,d in zip(paths,hashes)}); records[name] = r
    q,w = records.values()
    require(q['schema'] == 'current_c2_terminal_quadratic_capture_qualification_v1'
        and w['schema'] == 'current_c2_terminal_quadratic_capture_qualification_watch_v1'
        and q['candidate_sha256'] == w['candidate_sha256'] == h.CANDIDATE_SHA
        and w['ranks'] == [35,35] and w['retired_calls'] == w['full_model_calls'] == 2
        and [w['child_'+k] for k in ('result_sha256','completion_sha256','finalization_sha256')]
            == QUALIFICATION['capture_qualification_v1']
        and all(w[k] for k in ('cleanup_passed','whole_owned_tree_retired','observational_groups_retired',
            'actual_completed_samples','hashes_unchanged','sources_unchanged'))
        and w['maximum_combined_rss_bytes'] < RSS and w['maximum_interval_s'] <= 1
        and w['maximum_swap_growth_bytes'] == 0, 'Qualified cold35/repeat35 operational boundary')
    for record in (q,w):
        for group in ('source_bindings','input_bindings','artifacts'):
            h.bind(record[group],tick)
            destination = sources if group == 'source_bindings' else inputs
            for path,digest in record[group].items():
                require(path not in destination or destination[path] == digest, 'Unambiguous direct inventory')
                destination[path] = digest
    h.verify_result(q,tick)
    require(q['research_flops'] == 804891105422, 'Actual complete qualification research cost')
    return q,dict(source_bindings=sources,input_bindings=inputs,qualification_seals=QUALIFICATION,
        primary35_capture=q['captures'][0],qualification_research_flops=q['research_flops'])


def fixture(h, rank, inputs, tick=None):
    manifest_path = h.STORE/'manifest.json'
    require(h.sha(manifest_path,tick) == h.FIXTURE[manifest_path], 'Original fixed canonical STORE')
    inputs[str(manifest_path)] = h.FIXTURE[manifest_path]
    items = [x for x in h.read(manifest_path)['roles']['observer_fit']['row_manifests'] if x['role_rank'] == rank]
    require(len(items) == 1, 'One canonical fixed TRAIN row')
    item = items[0]; path = h.STORE/item['path']
    require(path == h.STORE/f'rows/observer_fit/{rank:04d}/row_manifest.json'
        and h.sha(path,tick) == item['sha256'], 'Canonical row path/hash')
    row = h.read(path); data = row['files']['weights.npy']; wp = path.parent/'weights.npy'
    require(data['dtype'] == '<f4' and data['shape'] == [16,1024,1024]
        and data['bytes'] == 67108992 and isinstance(row['mlp_seed'],int)
        and len(data['array_sha256']) == 64 and h.sha(wp,tick) == data['sha256'], 'Original physical FP32 file and framed metadata')
    inputs[str(path)] = item['sha256']; inputs[str(wp)] = data['sha256']
    return dict(rank=rank,row_manifest=str(path),row_manifest_sha256=item['sha256'],weights=str(wp),
        weights_sha256=data['sha256'],array_sha256=data['array_sha256'],mlp_seed=row['mlp_seed'],
        dtype=data['dtype'],shape=data['shape'],bytes=data['bytes'])


def prepare(h, original, observer, legacy):
    prepared = h.prepare(original,observer,legacy)
    loop,proof = splice(prepared['loop'],[
        ('for call,rank in enumerate(tuple(ranks)+(ranks[0],)):',
         'for call,rank in enumerate(tuple(ranks)):',1)],'qualified paid loop, no appended repeat')
    prepared['loop'] = loop
    prepared['proof'] = dict(prepared['proof'],single_cold_loop=proof,
        single_cold_only=True,no_extra_repeat=True,qualified_capture_owner_sha256=OWNER_SHA)
    return prepared


def admission(args, tick=None):
    """All fixed19 metadata when ordinal is absent, otherwise one child row."""
    h = owner(); started = time.monotonic()
    def guard():
        require(time.monotonic()-started < 60 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 1_000_000_000,
            'Complete stdlib admission60s/1GB')
        if tick is not None: tick()
    q,context = qualification(h,guard)
    own = {HERE/'capture_train.py':args.source_sha256,HERE/'watch_capture_train.py':args.watcher_sha256,
        HERE/'TRAIN_CAPTURE_PROTOCOL.md':args.protocol_sha256,HERE/'TRAIN_CAPTURE_WATCH_SPEC.md':args.watch_spec_sha256}
    require(all(isinstance(v,str) and h.re.fullmatch('[0-9a-f]{64}',v) for v in own.values()), 'Explicit four reviewed new owners')
    h.bind(own,guard); context['source_bindings'].update({str(p):d for p,d in own.items()})
    ordinal = getattr(args,'ordinal',None)
    if ordinal is not None: folder(ordinal)
    ranks = SEQUENCE if ordinal is None else (SEQUENCE[ordinal],)
    context['fixtures'] = {str(rank):fixture(h,rank,context['input_bindings'],guard) for rank in ranks}
    original,backend,observer,legacy = h.providers()
    prepared = prepare(h,original,observer,legacy)
    _,_,source_proof = execution_sources(h)
    context.update(candidate_sha256=h.CANDIDATE_SHA,source_proof=dict(prepared['proof'],expansion_sources=source_proof),
        qualified_cold_startup=h.read(q['diagnostic_records'][0]['path'])['startup'],truth_accessed=False,
        ordinal=ordinal,rank=None if ordinal is None else SEQUENCE[ordinal])
    require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules), 'Admission imports no numerical module')
    return context


def verify_backend(h, record):
    a = record['backend_attestation']; site = h.NEW/'openblas_runtime_env_v1/lib/python3.11/site-packages'
    require(a['site'] == str(site) and a['actual_origins_qualified'] and a['production_blas_qualified']
        and a['numpy_version'] == '2.4.6' and a['flopscope_version'] == '0.12.0'
        and a['whestbench_version'] == '0.16.0' and not a['scipy_loaded'] and not a['torch_loaded'], 'Actual qualified backend')
    pools = [p for p in a['threadpools'] if p.get('user_api') == 'blas']
    require(len(pools) == 1 and pools[0]['internal_api'] == 'openblas' and pools[0]['num_threads'] == 1
        and pools[0]['version'] == '0.3.31.188.0' and pools[0]['threading_layer'] == 'pthreads'
        and Path(pools[0]['filepath']).resolve() == (site/'numpy/.dylibs/libscipy_openblas64_.dylib').resolve(), 'Exact actual production BLAS')
    require(all(record['source_bindings'].get(x['path']) == x['sha256'] for x in a['modules'].values()), 'Every imported origin bound')


def verify_result(record, tick=None):
    h = owner(); ordinal = record['ordinal']; rank = SEQUENCE[ordinal]; path = folder(ordinal)
    require(record['schema'] == SCHEMA and record['candidate_sha256'] == h.CANDIDATE_SHA
        and record['ranks'] == [rank] and record['full_model_calls'] == record['retired_call_count'] == 1
        and record['qualification_seals'] == QUALIFICATION and record['initialization_flops'] == h.INIT
        and all(record[k] is True for k in ('passed','completed','hashes_unchanged','sources_unchanged',
            'all16_domains_qualified','all47_selectors_qualified','exact_head_replay','own_complete_bills',
            'all_parameter_and_estimator_owners_retired','actual_production_backend_qualified','fresh_cold_process'))
        and all(record[k] is False for k in ('truth_accessed','score_measured','fit_performed',
            'training_expansion_authorized','automatic_next_stage','uploaded','submission_authorized',
            'repeat_byte_exact','capture_repeat_byte_exact','current_control_output_byte_exact','current_control_complete_flops_exact'))
        and record['parameter_updates'] == 0, 'One actual target-free fresh cold fixed TRAIN capture')
    require(len(record['captures']) == len(record['trials']) == len(record['replay_bills']) == len(record['input_preparation']) == 1
        and len(record['artifacts']) == 11, 'Exactly one original7 plus capture/replay4')
    for group in ('source_bindings','input_bindings','artifacts'): h.bind(record[group],tick)
    verify_backend(h,record)
    trial,capture,rb = record['trials'][0],record['captures'][0],record['replay_bills'][0]
    require(trial['call'] == capture['call'] == 0 and trial['rank'] == capture['rank'] == rank
        and 0 < trial['flops'] <= h.BUDGET and trial['includes_initialization'] is True
        and trial['wall_s'] < 120 and trial['operation_ledger']['total_flops'] == trial['flops']
        and trial['multiplier'] == max(.1,trial['flops']/h.BUDGET), 'Own complete cold model bill, never borrowed')
    dp = str(path/'call_00_diagnostics.json')
    require(len(record['diagnostic_records']) == 1 and record['diagnostic_records'][0]['path'] == dp
        and record['artifacts'][dp] == record['diagnostic_records'][0]['sha256'] == h.sha(dp,tick), 'Bound actual same-call diagnostics')
    d = h.read(dp); m = h.read(capture['path'])
    require(d['startup'] == record['qualified_cold_startup']
        and d['startup']['observed_startup_region_flops'] == h.INIT
        and record['initialization_accounting'] == 'observed fixed source-owned startup, not cold-minus-repeat'
        and d['head_replay'] == m['head_replay']
        and all(d['head_replay'].values()) and d['current_output_byte_exact'] is False
        and d['current_complete_flops_exact'] is False and d['operation_counts_reported_not_rebased'], 'Honest same-call replay and original cold startup')
    for receipt in (trial['lifecycle_receipt'],trial['input_reset_receipt'],rb['lifecycle']):
        life = h.read(receipt)
        require(life['complete'] and life['context_released'] and life['closed_context_bill_unchanged_by_reset'], 'All three owned contexts retired')
    require(h.read(rb['lifecycle'])['replay_array_owners_retired'], 'Replay array ownership retired')
    require(h.read(trial['input_bill'])['flops'] == 16777216
        and h.read(trial['input_bill'])['weight_file_sha256'] == record['fixtures'][str(rank)]['weights_sha256'], 'Own exact canonical paid input')
    for p,digest in ((capture['path'],capture['sha256']),(m['path'],m['sha256']),
        (rb['path'],rb['sha256']),(rb['lifecycle'],rb['lifecycle_sha256'])):
        require(record['artifacts'].get(p) == digest == h.sha(p,tick), 'Actual complete capture and replay artifacts')
    require(m['rank'] == rank and m['call'] == 0 and m['candidate_sha256'] == h.CANDIDATE_SHA
        and m['arrays']['hidden']['shape'] == [1024,64] and m['arrays']['hidden']['dtype'] == 'float32'
        and m['arrays']['sigma']['shape'] == [1024] and m['arrays']['sigma']['dtype'] == 'float64'
        and set(m['arrays']) == {'hidden','sigma','host','delta','hook_delta','raw','served','P_head','P_head_b','P_gain','P_rms'}, 'Same qualified eleven-array frame')
    replay = h.read(rb['path'])
    require(replay['flops'] == rb['flops'] and 0 < replay['flops'] <= h.REPLAY_CAP
        and replay['wall_s'] < 30 and replay['charged_to_participant'] is False, 'Complete paid replay cap')
    require(record['research_flops'] == trial['flops']+16777216+rb['flops']
        and record['research_flops'] <= h.BUDGET+16777216+h.REPLAY_CAP, 'Complete research cost, diagnostic never credited')
    complete = h.read(path/'call_00_complete.json')
    require(complete['passed'] and complete['completed'] and complete['ordinal'] == ordinal
        and complete['call'] == 0 and complete['rank'] == rank and complete['context_retired'], 'Actual fixed ordinal retired marker')
    for suffix,key in (('bill','bill_sha256'),('lifecycle','lifecycle_sha256'),('diagnostics','diagnostic_sha256'),
            ('capture','capture_manifest_sha256'),('replay_bill','replay_bill_sha256'),('replay_lifecycle','replay_lifecycle_sha256')):
        file = str(path/f'call_00_{suffix}.json')
        require(record['artifacts'][file] == complete[key] == h.sha(file,tick), 'Complete marker linked to actual six JSON artifacts')
    return True


def execution_sources(h):
    original = OWNER.read_text(); main = h.source_segment(original,'main')
    start = main.index("            control = context['controls'][call]")
    end = main.index("            path = folder/f'call_{call:02d}_diagnostics.json'",start)
    old_finish = main[start:end]
    new_finish = """            require(startup == context['qualified_cold_startup']
                and startup['observed_startup_region_flops'] == INIT, 'Actual observed identical fixed cold initialization ownership')
            record = dict(call=call,rank=rank,domains=domain,startup=startup,dispatches=dispatches,
                current_control_call=None,current_output_byte_exact=False,current_complete_flops_exact=False,
                head_replay=checks,charged_and_non_unstack_sequence_exact=False,
                operation_counts_reported_not_rebased=True,fresh_cold_process=True)
"""
    begin = main.index("        require(result['full_model_calls'] == result['retired_call_count'] == 2")
    end = main.index('        backend.backend_origins(result,tick)',begin)
    old_repeat = main[begin:end]
    new_repeat = """        require(result['full_model_calls'] == result['retired_call_count'] == 1, 'Only one fresh cold TRAIN call')
        result['initialization_flops'] = INIT
        result['initialization_accounting'] = 'observed fixed source-owned startup, not cold-minus-repeat'
        result['fresh_cold_process'] = True
"""
    edits = [
        ("    parser.add_argument('--run',action='store_true')", "    parser.add_argument('--run',action='store_true')\n    parser.add_argument('--ordinal',type=int,required=True)",1),
        ("'Exclusive first35 qualification; no resume'","'Exclusive fixed ordinal child; no resume'",1),
        ("ranks=[35,35],trials=[]","ranks=[RANK],ordinal=ORDINAL,trials=[]",1),
        ("'Whole300s/process8GB; external supervisor owns completed samples and full tree'",
         "'Whole child180s/process8GB; outer3000s completed samples and full tree'",1),
        ("('source_bindings','input_bindings','controls','control_seals','source_proof')",
         "('source_bindings','input_bindings','qualification_seals','source_proof','fixtures','qualified_cold_startup','primary35_capture','qualification_research_flops')",1),
        (old_finish,new_finish,1),
        ("record = dict(passed=True,completed=True,call=call,rank=rank,context_retired=True,",
         "record = dict(passed=True,completed=True,call=call,rank=rank,ordinal=ORDINAL,context_retired=True,",1),
        ("dict(event='retired',call=call,rank=rank,complete_sha256=sha(path))",
         "dict(event='retired',call=call,rank=rank,ordinal=ORDINAL,complete_sha256=sha(path))",1),
        ("        rows = {35:dict(weights=str(ROW/'weights.npy'),mlp_seed=context['mlp_seed'])}",
         "        row = context['fixtures'][str(RANK)]\n        rows = {RANK:dict(weights=row['weights'],mlp_seed=row['mlp_seed'])}",1),
        ("rows=rows,ranks=(35,),result=result", "rows=rows,ranks=(RANK,),result=result",1),
        (old_repeat,new_repeat,1),
        ("all47_selectors_qualified=True,repeat_byte_exact=True,", "all47_selectors_qualified=True,repeat_byte_exact=False,",1),
        ("capture_repeat_byte_exact=True,exact_head_replay=True,current_control_output_byte_exact=True,",
         "capture_repeat_byte_exact=False,exact_head_replay=True,current_control_output_byte_exact=False,",1),
        ("current_control_complete_flops_exact=True,own_complete_bills=True,", "current_control_complete_flops_exact=False,own_complete_bills=True,",1),
        ("input_preparation_in_participant_bill=False,primary35_reusable_only_after_separate_expansion_selection=True)",
         "input_preparation_in_participant_bill=False,primary35_reusable_only_after_separate_expansion_selection=False)",1),
    ]
    main,main_proof = splice(main,edits,'qualified main')
    replay,replay_proof = splice(h.source_segment(original,'replay'),[
        ('rank=35',"rank=trial['rank']",4)],'qualified paid replay metadata only')
    return main,replay,dict(main=main_proof,replay=replay_proof,original_capture_sha256=OWNER_SHA,
        unchanged_head_replay_and_domain_arithmetic=True,no_extra_participant_arithmetic=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true'); parser.add_argument('--ordinal',type=int)
    for key in ('source','watcher','protocol','watch-spec'): parser.add_argument('--'+key+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    out = folder(args.ordinal); h = owner()
    require(OUTPUT.is_dir() and not OUTPUT.is_symlink() and not out.exists(), 'Outer fresh fixed19 launch owns this child')
    launch = h.read(OUTPUT/'launch.json')
    require(launch['ranks'] == list(SEQUENCE) and launch['qualification_seals'] == QUALIFICATION
        and launch['worker_authority'] == {k:getattr(args,k) for k in
            ('source_sha256','watcher_sha256','protocol_sha256','watch_spec_sha256')}, 'Actual outer fixed authority/roster')
    launch_sha = h.sha(OUTPUT/'launch.json')
    if args.ordinal:
        prior = h.read(folder(args.ordinal-1)/'result.json')
        verify_result(prior)
        r,e,f = [h.read(folder(args.ordinal-1)/name) for name in ('result.json','completion.json','finalization.json')]
        require(all(x.get('passed') is True and x.get('completed') is True for x in (r,e,f))
            and e['result_sha256'] == f['result_sha256'] == h.sha(folder(args.ordinal-1)/'result.json')
            and f['completion_sha256'] == h.sha(folder(args.ordinal-1)/'completion.json'), 'No skipped or failed predecessor')
    main_source,replay_source,proof = execution_sources(h)
    def adapted_prepare(original,observer,legacy):
        prepared = prepare(h,original,observer,legacy)
        prepared['proof'] = dict(prepared['proof'],expansion_sources=proof)
        return prepared
    def admitted_child(parsed, tick=None):
        context = admission(parsed,tick)
        require(h.sha(OUTPUT/'launch.json',tick) == launch_sha, 'Exact outer launch immutable through child admission')
        context['input_bindings'][str(OUTPUT/'launch.json')] = launch_sha
        return context
    env = dict(vars(h),__name__='_fixed_cold_h15_train_body',__file__=str(Path(__file__).resolve()),
        OUTPUT=out,SCHEMA=SCHEMA,CAP=CAP,RSS=RSS,RANK=SEQUENCE[args.ordinal],ORDINAL=args.ordinal,
        admission=admitted_child,prepare=adapted_prepare,verify_result=verify_result)
    exec(compile(replay_source,'<qualified-paid-replay-rank-metadata-adapter>','exec'),env)
    exec(compile(main_source,'<qualified-single-cold-capture-main>','exec'),env)
    return env['main']()


if __name__ == '__main__':
    raise SystemExit(main())
