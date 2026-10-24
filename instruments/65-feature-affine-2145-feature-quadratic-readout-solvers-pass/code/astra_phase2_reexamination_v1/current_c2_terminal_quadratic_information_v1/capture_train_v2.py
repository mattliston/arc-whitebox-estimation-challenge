"""DRAFT: fixed17 H15 continuation; actual lazy bindings, no numerical change.

The failed V1 row40 and whole-stage receipts remain negative and charged.
Only its independently positive36/38 children may be reused. No stage at import.
"""
import ast
import hashlib
import importlib.util
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD_SOURCES = {
    'capture_train.py':'30b4ba8d63d4bf255432b8a4fea485e52d0abb4fc58fcfbc5b935a2bbc9654af',
    'watch_capture_train.py':'a00884a5d3236892a8a652dd1af235ee457fdbb5ffd2c94739624253517d246e',
    'TRAIN_CAPTURE_PROTOCOL.md':'ed6095457ba8f2bb544c273d7bb0f474622c1987002a8bda26376aad14225e0a',
    'TRAIN_CAPTURE_WATCH_SPEC.md':'8eed8263c98e35332a2f60cddc7527a6e57fc99b2b015bef81a8ebcb96a44547',
}
OLD_WATCH_SEALS = [
    'a73ccd04b697cc2ff436e39d3ca55838c2f31f32f01f8f71f6ae7fb6bb7e4a57',
    'f031bb041da2ce4b4543be93e243a558f6646d9b2483cdf18795aad0f837e09c',
    '06e44d951dfa2d23c22e6569a806e9c2a396312a4414c153e7a13cf9cb76c4d9']
FAILED40_SEALS = [
    '757d1c32847f2058cce86a9dea2355e2e1207e19866a0a32c42c7db4a68a9662',
    'e26a75fd88a80098e74adfd0856549a14508dfbe5a9f5fa08e23d9152fe96964',
    '02446d406603301b574fbbc65760f75c0a015671b9d655e62c87ab2367d58d83']
PREFIX_SEALS = [
    ['e30381a9df1507208528feb26a8e05bd434ea8afedbd5b4be8a42bd22f136025',
     '773b6feb0b12aa405bb7274c27125a497b3ae5c33e79de0337d4cd630b2730c7',
     '7cc21b11f0fe3b04d1d7b747dd160ffffe4a1bb93d8c27bf72f352be9ce51117'],
    ['29fae99d5631abc04039148c9fdeb422de518907053e51c5b1726138b0d32b56',
     'a4656c5ad18abb2b36d0d267e8338a58e3ee8881c49a187b3e376b987da087e4',
     '9c5d59f0c2ada6bab7c64de44809c4fa9572361869fb3f1b0f564009c6e85f7b']]
PRIOR_TRAIN_ELAPSED = 243.16458587441593
PRIOR_TRAIN_FLOPS = 1192846894590
FAILED40_FLOPS = 381270577516
LAYOUTS = {
    '_PACKED_PUSH':(524226560,24), '_PACKED_READ':(544727040,24),
    '_P1_D6_832_PUSH_SPECS':(278566656,34), '_P1_D6_832_READ_SPECS':(282489344,34),
    '_PACKED_SQUARE':(48758784,18), '_PACKED_SQUARE4':(93716480,24),
    '_PACKED_SQUARE5':(172392448,30),
}
DEPTH_NAMES = {3:'_PACKED_SQUARE',4:'_PACKED_SQUARE4',5:'_PACKED_SQUARE5'}
_OLD = _API = None


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def old_owner():
    global _OLD
    if _OLD is None:
        path = HERE/'capture_train.py'
        require(hashlib.sha256(path.read_bytes()).hexdigest() == OLD_SOURCES[path.name], 'Frozen V1 source')
        spec = importlib.util.spec_from_file_location('h15_preserved_v1_capture',path)
        _OLD = importlib.util.module_from_spec(spec); spec.loader.exec_module(_OLD)
    return _OLD


def layout_price(h, tick=None):
    """Literal-only independent price of the actual last bound source layouts."""
    require(h.sha(h.CANDIDATE,tick) == h.CANDIDATE_SHA, 'Unchanged actual c2 source')
    tree = ast.parse(h.CANDIDATE.read_text()); values = {}
    if tick is not None: tick()
    reshapes = [n for n in ast.walk(tree) if isinstance(n,ast.Call)
        and isinstance(n.func,ast.Attribute) and n.func.attr == 'reshape']
    require(len(reshapes) == 1 and isinstance(reshapes[0].func.value,ast.Name)
        and reshapes[0].func.value.id == 'fnp', 'Only the source-owned binding helper reshapes')
    for node in tree.body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in LAYOUTS:
                    values[target.id] = ast.literal_eval(node.value)
    require(set(values) == set(LAYOUTS), 'All seven actual source layouts')
    for name,specs in values.items():
        require(all(type(a) is int and type(b) is int and 0 <= a < b <= 292225024
            and b-a == math.prod(shape) for a,b,shape in specs), 'Exact legal original-arena spans')
        require((sum(b-a for a,b,_ in specs),len(specs)) == LAYOUTS[name], 'Independent binding F/call tariff')
    return {name:dict(flops=f,reshape_calls=n) for name,(f,n) in LAYOUTS.items()}


def startup_check(startup, trial, qualified):
    """No network-dependent chronology or unused depth is forced to rank35."""
    variable = {'bindings','square_depths_bound_after','observed_startup_region_flops'}
    require({k:v for k,v in startup.items() if k not in variable}
        == {k:v for k,v in qualified.items() if k not in variable}, 'All unchanged cold prefix/arena/materialization guards')
    require(startup['prefix']['flops'] == 92986 and startup['square_depths_bound_before'] == [], 'Actual cold prefix')
    rows = startup['bindings']; names = [x['name'] for x in rows]
    depths = startup['square_depths_bound_after']
    require(depths == sorted(set(depths)) and all(type(d) is int and d in DEPTH_NAMES for d in depths),
        'Only actually reached unchanged square schedules3/4/5')
    expected = set(LAYOUTS)-set(DEPTH_NAMES.values()) | {DEPTH_NAMES[d] for d in depths}
    require(len(names) == len(set(names)) and set(names) == expected, 'Mandatory four layouts plus each actual depth once')
    reference = {x['name']:x for x in qualified['bindings']}
    for item in rows:
        f,n = LAYOUTS[item['name']]
        require(item == reference[item['name']] and item['flops'] == item['expected_flops'] == f
            and item['reshape_calls'] == n and item['same_original_arena'] and item['arena_dtype'] == 'float32',
            'Exact named binding tariff, dtype, ownership and no duplicate or extra work')
    binding_f = sum(LAYOUTS[name][0] for name in names)
    binding_n = sum(LAYOUTS[name][1] for name in names)
    ledger = trial['operation_ledger']
    reshape = [x for x in ledger['by_operation'] if x['operation'] == 'reshape']
    require(reshape == [dict(operation='reshape',count=binding_n,flops=binding_f)]
        and ledger['total_flops'] == trial['flops'] and ledger['unchanged_closed_context_bill'],
        'Actual complete closed ledger exactly owns every source binding reshape')
    actual = 92986+binding_f
    require(startup['observed_startup_region_flops'] == actual
        and startup['outside_prefix_materialization_flops'] == 0, 'Observed startup matches actual source-owned sum')
    comparison = failed40_comparison(trial)
    require(not comparison['applicable'] or (comparison['output_bytes_equal'] and comparison['complete_bill_equal']),
        'Rerun40 exact saved output/full bill; negative V1 is a control, not a qualified row')
    return actual


def failed40_comparison(trial):
    applicable = trial['rank'] == 40
    return dict(applicable=applicable,prior_stage_positive=False,
        negative_result_sha256=FAILED40_SEALS[0] if applicable else None,
        output_bytes_equal=applicable and trial['output_bytes_sha256']
            == '7d08d3218ba6355271d037ad2635e0e4050ae39d2020a36eb36003e85a05dd59',
        complete_bill_equal=applicable and trial['flops'] == 381252839525)


def closed_cost(record):
    totals = {}
    for key in ('trials','input_preparation','replay_bills'):
        values = record.get(key,[])
        require(all(type(x['flops']) is int and x['flops'] >= 0 for x in values), 'Actual nonnegative closed bills')
        totals[key] = sum(x['flops'] for x in values)
    return dict(model_flops=totals['trials'],input_flops=totals['input_preparation'],
        replay_flops=totals['replay_bills'],research_flops=sum(totals.values()),
        closed_model_contexts=len(record.get('trials',[])),
        closed_input_contexts=len(record.get('input_preparation',[])),
        closed_replay_contexts=len(record.get('replay_bills',[])),
        closed_bill_inventory_only=True,unclosed_context_cost_not_claimed=True)


def history_admission(h, tick=None):
    """Positive children plus a negative, but cleanly observed, immutable prefix."""
    old = old_owner(); sources = {str(HERE/n):d for n,d in OLD_SOURCES.items()}; inputs = {}
    h.bind(sources,tick)
    def merge(record):
        for group in ('source_bindings','input_bindings','artifacts'):
            destination = sources if group == 'source_bindings' else inputs
            for p,d in record[group].items():
                require(p not in destination or destination[p] == d, 'Consistent historical inventory')
                destination[p] = d
    def chain(folder, expected, positive):
        paths = [folder/n for n in ('result.json','completion.json','finalization.json')]
        ds = [h.sha(p,tick) for p in paths]; r,e,f = map(h.read,paths)
        require(ds == expected and all(x.get('passed') is positive and x.get('completed') is positive for x in (r,e,f))
            and e['result_sha256'] == f['result_sha256'] == ds[0] and f['completion_sha256'] == ds[1],
            'Actual historical three-seal polarity and linkage')
        inputs.update({str(p):d for p,d in zip(paths,ds)}); merge(r)
        return r,f
    watch_path = HERE/'train_capture_watch_v1'
    w,wf = chain(watch_path,OLD_WATCH_SEALS,False)
    require(w['schema'] == 'current_c2_terminal_quadratic_train_capture_watch_v1'
        and w['candidate_sha256'] == h.CANDIDATE_SHA and w['retired_calls'] == 2
        and w['ranks'] == list(old.SEQUENCE) and len(w['commands']) == 3
        and [x['rank'] for x in w['commands']] == [36,38,40]
        and len(w['children']) == len(w['markers']) == 2 and [x['rank'] for x in w['rows']] == [35,36,38]
        and 'One fresh child and exact cumulative retired count' in w['error']
        and wf['elapsed_s'] == PRIOR_TRAIN_ELAPSED
        and all(w[k] for k in ('cleanup_passed','whole_owned_tree_retired','observational_groups_retired'))
        and w['maximum_combined_rss_bytes'] < 8_000_000_000 and 0 < w['maximum_interval_s'] <= 1
        and w['maximum_swap_growth_bytes'] == 0 and w['numerical_imports'] is False,
        'Negative whole19 stays negative; only its clean actual prefix is reusable')
    sp = watch_path/'samples.jsonl'
    require(sp.stat().st_size < 16*1024**2 and h.sha(sp,tick) == w['artifacts'][str(sp)], 'Bound actual resource history')
    samples = []
    for line in sp.read_text().splitlines():
        samples.append(h.json.loads(line))
        if tick is not None: tick()
    require(len(samples) == w['observation_samples'] == len(w['completed_sample_intervals_s'])
        and max(w['completed_sample_intervals_s']) == w['maximum_interval_s']
        and all(0 < x <= 1 for x in w['completed_sample_intervals_s'])
        and all(0 <= s['elapsed_s'] < PRIOR_TRAIN_ELAPSED and 0 < s['combined_rss_bytes'] < 8_000_000_000
            and s['swap_bytes'] <= w['initial_swap_bytes'] and s['observational_groups_retired'] for s in samples)
        and max(s['combined_rss_bytes'] for s in samples) == w['maximum_combined_rss_bytes'],
        'Completed <=1s observations, original8GB/zero-added-swap across all three attempted children')
    require(all(any(command['pid'] in s['child_pids'] for s in samples) for command in w['commands']),
        'Every attempted child was actually observed in the owned numerical tree')
    rows,positive_cost = [],0
    for i,rank in enumerate((36,38)):
        r,f = chain(old.folder(i),PREFIX_SEALS[i],True); old.verify_result(r,tick)
        child,marker = w['children'][i],w['markers'][i]
        require(child['rank'] == rank and child['ordinal'] == i and child['child_reaped']
            and [child[k] for k in ('result_sha256','completion_sha256','finalization_sha256')] == PREFIX_SEALS[i]
            and child['research_flops'] == r['research_flops'] and f['elapsed_s'] < 180
            and marker['ordinal'] == i and marker['rank'] == rank
            and marker['sha256'] == r['artifacts'][marker['path']] == h.sha(marker['path'],tick),
            'Each reused row owns a positive final child chain and an observed/reaped completion')
        row = dict(w['rows'][i+1])
        require(row['worker_seals'] == PREFIX_SEALS[i] and row['result_path'] == str(old.folder(i)/'result.json')
            and row['capture_manifest'] == r['captures'][0]['path']
            and row['capture_manifest_sha256'] == r['captures'][0]['sha256']
            and row['own_cold_flops'] == r['trials'][0]['flops'], 'Exact positive saved row, not a replacement')
        row.update(watch_folder=str(watch_path),watch_scope='clean_negative_prefix_positive_child',
            prior_failure_unchanged=True,initialization_flops=r['initialization_flops'])
        rows.append(row); positive_cost += r['research_flops']
    failed,ff = chain(old.folder(2),FAILED40_SEALS,False)
    require(failed['ranks'] == [40] and failed['ordinal'] == 2
        and failed['full_model_calls'] == failed['retired_call_count'] == 0 and failed['diagnostic_records'] == []
        and len(failed['trials']) == len(failed['input_preparation']) == len(failed['replay_bills']) == 1
        and len(failed['artifacts']) == 9
        and 'Actual observed identical fixed cold initialization ownership' in failed['error'],
        'Preserve exact V1 metadata failure; no completed row40 is invented')
    cost = closed_cost(failed)
    require(cost['research_flops'] == FAILED40_FLOPS and positive_cost+FAILED40_FLOPS == PRIOR_TRAIN_FLOPS,
        'Include all failed40 model/input/replay F once without credit')
    for receipt in (failed['trials'][0]['lifecycle_receipt'],failed['trials'][0]['input_reset_receipt'],
                    failed['replay_bills'][0]['lifecycle']):
        life = h.read(receipt)
        require(life['complete'] and life['context_released'] and life['closed_context_bill_unchanged_by_reset'],
            'All three actually billed rank40 contexts closed before process cleanup')
    require(h.read(failed['replay_bills'][0]['lifecycle'])['replay_array_owners_retired']
        and ff['elapsed_s'] < 180, 'Rank40 paid replay owners retired within original child cap')
    log = watch_path/'worker_02.log'
    inputs[str(log)] = '6f721759539e50349e7e7edd9446ba83c61dfbda7d62bbe36b209c5b4429d3d1'
    h.bind(sources,tick); h.bind(inputs,tick)
    return dict(source_bindings=sources,input_bindings=inputs,rows=rows,
        negative_watch_folder=str(watch_path),negative_watch_seals=OLD_WATCH_SEALS,
        positive_child_seals=PREFIX_SEALS,failed40_seals=FAILED40_SEALS,
        failed40_closed_cost=cost,prior_train_research_flops=PRIOR_TRAIN_FLOPS,
        prior_train_elapsed_s=PRIOR_TRAIN_ELAPSED,prior_actual_model_calls=3,
        prior_failure_unchanged=True,failed40_not_a_train_row=True)


def derive_source():
    old = old_owner(); original = (HERE/'capture_train.py').read_text()
    edits = [
        ('SEQUENCE = (36,38,40,41,43,44,45,46,47,49,50,51,52,53,54,55,56,57,58)',
         'SEQUENCE = (40,41,43,44,45,46,47,49,50,51,52,53,54,55,56,57,58)',1),
        ('TRAIN = (35,)+SEQUENCE','TRAIN = (35,36,38)+SEQUENCE',1),
        ("OUTPUT = HERE/'train_capture_v1'","OUTPUT = HERE/'train_capture_v2'",1),
        ("SCHEMA = 'current_c2_terminal_quadratic_train_capture_row_v1'",
         "SCHEMA = 'current_c2_terminal_quadratic_train_capture_row_v2'",1),
        ('0 <= ordinal < 19','0 <= ordinal < 17',1),
        ("own = {HERE/'capture_train.py':args.source_sha256,HERE/'watch_capture_train.py':args.watcher_sha256,\n        HERE/'TRAIN_CAPTURE_PROTOCOL.md':args.protocol_sha256,HERE/'TRAIN_CAPTURE_WATCH_SPEC.md':args.watch_spec_sha256}",
         "own = {HERE/'capture_train_v2.py':args.source_sha256,HERE/'watch_capture_train_v2.py':args.watcher_sha256,\n        HERE/'TRAIN_CAPTURE_V2_PROTOCOL.md':args.protocol_sha256,HERE/'TRAIN_CAPTURE_V2_SPEC.md':args.watch_spec_sha256}",1),
        ("    ordinal = getattr(args,'ordinal',None)",
         "    history = history_admission(h,guard)\n    for group in ('source_bindings','input_bindings'):\n        for p,d in history[group].items():\n            require(p not in context[group] or context[group][p] == d, 'Consistent V1 history')\n            context[group][p] = d\n    context['continuation_history'] = {k:v for k,v in history.items() if k not in ('source_bindings','input_bindings')}\n    context['binding_source_price'] = layout_price(h,guard)\n    ordinal = getattr(args,'ordinal',None)",1),
        ("record['qualification_seals'] == QUALIFICATION and record['initialization_flops'] == h.INIT",
         "record['qualification_seals'] == QUALIFICATION and record['continuation_history']['prior_failure_unchanged']",1),
        ("require(d['startup'] == record['qualified_cold_startup']\n        and d['startup']['observed_startup_region_flops'] == h.INIT\n        and record['initialization_accounting'] == 'observed fixed source-owned startup, not cold-minus-repeat'",
         "require(record['initialization_flops'] == startup_check(d['startup'],trial,record['qualified_cold_startup'])\n        and record['binding_source_price'] == {n:dict(flops=f,reshape_calls=c) for n,(f,c) in LAYOUTS.items()}\n        and record['initialization_accounting'] == 'observed actual unique lazy bindings plus fixed prefix, not cold-minus-repeat'",1),
        ("new_finish = \"\"\"            require(startup == context['qualified_cold_startup']\n                and startup['observed_startup_region_flops'] == INIT, 'Actual observed identical fixed cold initialization ownership')\n            record = dict",
         "new_finish = \"\"\"            # Actual startup is validated after publishing its detached same-call diagnostics.\n            record = dict",1),
        ('operation_counts_reported_not_rebased=True,fresh_cold_process=True)',
         'operation_counts_reported_not_rebased=True,fresh_cold_process=True,\n                failed40_comparison=failed40_comparison(trial))',1),
        ("and d['current_complete_flops_exact'] is False and d['operation_counts_reported_not_rebased']",
         "and d['current_complete_flops_exact'] is False and d['operation_counts_reported_not_rebased']\n        and d['failed40_comparison'] == failed40_comparison(trial)",1),
        ("        result['initialization_flops'] = INIT\n        result['initialization_accounting'] = 'observed fixed source-owned startup, not cold-minus-repeat'",
         "        result['initialization_flops'] = read(result['diagnostic_records'][0]['path'])['startup']['observed_startup_region_flops']\n        result['initialization_accounting'] = 'observed actual unique lazy bindings plus fixed prefix, not cold-minus-repeat'",1),
        ("'primary35_capture','qualification_research_flops')\",1)",
         "'primary35_capture','qualification_research_flops','continuation_history','binding_source_price')\",1)",1),
        ("    main,main_proof = splice(main,edits,'qualified main')",
         "    edits += [(\"            result['full_model_calls'] = call+1\",\n        \"            startup_check(startup,trial,context['qualified_cold_startup'])\\n            result['full_model_calls'] = call+1\",1),\n        (\"        result.update(passed=False,completed=False,error=traceback.format_exc())\",\n         \"        result.update(passed=False,completed=False,error=traceback.format_exc())\\n        result['failure_closed_cost'] = closed_cost(result)\",1)]\n    main,main_proof = splice(main,edits,'qualified main')",1),
        ("admission=admitted_child,prepare=adapted_prepare,verify_result=verify_result)",
         "admission=admitted_child,prepare=adapted_prepare,verify_result=verify_result,\n        startup_check=startup_check,closed_cost=closed_cost,failed40_comparison=failed40_comparison)",1),
        ('original_capture_sha256=OWNER_SHA,',
         'original_capture_sha256=OWNER_SHA,v2_capture_adapter=V2_SOURCE_PROOF,',1),
    ]
    return old.splice(original,edits,'V2 metadata/remaining17; all V1 arithmetic retained')


def api():
    global _API
    if _API is None:
        source,proof = derive_source()
        ns = dict(__name__='_h15_train_v2_inert',__file__=str(Path(__file__).resolve()),
            history_admission=history_admission,layout_price=layout_price,startup_check=startup_check,
            closed_cost=closed_cost,failed40_comparison=failed40_comparison,LAYOUTS=LAYOUTS,V2_SOURCE_PROOF=proof)
        exec(compile(source,'<whole-inverse-h15-train-v2>','exec'),ns)
        _API = ns
    return _API


def owner(): return api()['owner']()
def folder(ordinal): return api()['folder'](ordinal)
def admission(args, tick=None): return api()['admission'](args,tick)
def verify_result(record, tick=None): return api()['verify_result'](record,tick)
def main(): return api()['main']()


if __name__ == '__main__':
    raise SystemExit(main())
