"""Fresh fixed CHECK4; retain paid arithmetic, report history-dependent zero-F views."""
import ast
import hashlib
import json
from pathlib import Path
import time

_V2_HERE = Path(__file__).resolve().parent
_V2_OWNER = _V2_HERE/'check4.py'
_V2_OWNER_SHA = '7b4577d019618ea345765cbd99bc4cc7b107eb6bdde97cd11c5e0e9baeeb4543'
_V2_ORIGINAL = _V2_OWNER.read_text()
if hashlib.sha256(_V2_ORIGINAL.encode()).hexdigest() != _V2_OWNER_SHA:
    raise RuntimeError('Frozen negative CHECK4 source is preserved')
_V2_PACKAGE = _V2_HERE.parent.parent/'data_gate/.venv/lib/python3.11/site-packages/flopscope'
_V2_METADATA_SOURCES = {
    _V2_OWNER:_V2_OWNER_SHA,
    _V2_HERE/'watch_check4.py':'d7f99ac0b97563a064784acf88624a1ccdaa853b7f38bfe0472dda0ac6a6586a',
    _V2_HERE/'CHECK4_PROTOCOL.md':'89c460aec6094c2b147fe005fbe9e244c884c96ef2c4e5142078b6f09bc03577',
    _V2_HERE/'CHECK4_WATCH_SPEC.md':'0eeb11b9888fc349874967c8029aa2a9b8920d39ad185cefaa707693f833bdef',
    _V2_PACKAGE/'_array_ops.py':'e45e927f18c54352503a4951d3d925a5f31f92ecea245e32360825d71a05da6c',
    _V2_PACKAGE/'_ndarray.py':'fa5bbecb48b123ab77f31ced3da877d2e333aa5b403a324c33da591b967d3e74',
    _V2_PACKAGE/'_budget.py':'ad9e78172bd0e8175533d4aae75b67df749d96ecc1ca868135eb3d781e542d01',
    _V2_PACKAGE/'_registry.py':'4e091020d5cb8167271fa6ed76ac43983664e3a3cdd9e67ae76968f3884f6623',
}
_V2_NEGATIVE_SEALS = {
    'check4_v1':('7f43c2d8b4bacf267b75b044ee70532babf8979c1b3c1813cd2febe92e3ab385',
        '63e1977e88768389d7324cdc590f82db1d07c4409fed0edab4b92774ff3bd07c',
        'c1f02631200437a8fc69a101d7726790c4070c0a6239bfa4c02d3173c5c4f67b'),
    'check4_watch_v1':('c65408aeeaee72263e06824648e95060951a96fa888d08d76011f0b7911717df',
        '61371b005c18ee172c7396740f013c2aeb41e539bf03e4e6d25128a6b58d3e65',
        '8644aa34984c6c7acc4da2e1d0871a7832e779770939c5834ac91907a545fc00'),
}


def _v2_failed_attempt(inputs, tick=None):
    """Negative attempted outputs constrain replay; they grant no release authority."""
    additional, records = {}, {}
    for folder,expected in _V2_NEGATIVE_SEALS.items():
        paths = [_V2_HERE/folder/n for n in ('result.json','completion.json','finalization.json')]
        digests = tuple(sha(p,tick) for p in paths)
        require(digests == expected, 'Frozen actual negative CHECK4 seals')
        r,e,f = map(read,paths)
        require(all(x['passed'] is False and x['completed'] is False for x in (r,e,f))
            and e['result_sha256'] == f['result_sha256'] == expected[0]
            and f['completion_sha256'] == expected[1], 'Never rehabilitate the negative v1')
        additional.update({str(p):d for p,d in zip(paths,digests)})
        records[folder] = r
    r,w = records['check4_v1'],records['check4_watch_v1']
    require(r['candidate_sha256'] == COMPOSED_SHA and r['ranks'] == list(SEQUENCE)
        and r['full_model_calls'] == r['retired_call_count'] == w['retired_calls'] == 5
        and len(r['trials']) == len(r['diagnostic_records']) == 5 and len(r['artifacts']) == 35
        and 'Own deterministic complete bills reproduce the positive probe' in r['error']
        and all(w[k] is True for k in ('cleanup_passed','whole_owned_tree_retired','observational_groups_retired')),
        'Actual five attempted calls followed by failed metadata gate and external cleanup')
    for group in ('source_bindings','input_bindings','artifacts'):
        require(isinstance(r[group],dict) and r[group], 'Complete negative-attempt inventory')
        for path,digest in r[group].items():
            require(path not in additional or additional[path] == digest, 'Consistent old inventory')
            additional[path] = digest
    # Existing positive prerequisite paths were already freshly checked in this process.
    fresh = {}
    for path,digest in additional.items():
        require(path not in inputs or inputs[path] == digest, 'Same direct input authority')
        if path not in inputs:
            fresh[path] = digest
    bind(fresh,tick); inputs.update(additional)
    return r,{k:dict(zip(('result_sha256','completion_sha256','finalization_sha256'),v))
        for k,v in _V2_NEGATIVE_SEALS.items()}


def _v2_ledger_metrics(original_metrics):
    """Only the original two lexically post-context metrics() calls use this wrapper."""
    def metrics(ctx, started):
        record = original_metrics(ctx,started)
        began = time.monotonic()
        before = (int(ctx.flops_used),len(ctx.op_log),float(ctx.wall_time_s))
        totals, unstack_shapes = {}, {}
        all_hash,non_unstack_hash,charged_hash = (hashlib.sha256() for _ in range(3))
        zero = charged = unstack = flops = 0
        def shape_value(value):
            if isinstance(value,(tuple,list)):
                return [shape_value(v) for v in value]
            if value is None or isinstance(value,str):
                return value
            return int(value)
        for op in ctx.op_log:
            name,cost = str(op.op_name),int(op.flop_cost)
            require(cost >= 0, 'Nonnegative actually recorded public tariff')
            shapes = shape_value(op.shapes)
            encoded = json.dumps([name,op.subscripts,shapes,cost,op.resolved_dtype],
                separators=(',',':'),allow_nan=False).encode()+b'\n'
            all_hash.update(encoded)
            entry = totals.setdefault(name,dict(operation=name,count=0,flops=0))
            entry['count'] += 1; entry['flops'] += cost; flops += cost
            if cost:
                charged += 1; charged_hash.update(encoded)
            else:
                zero += 1
            if name == 'unstack':
                require(cost == 0, 'Every recorded unstack is officially zero-F')
                unstack += 1
                key = json.dumps(shapes,separators=(',',':'))
                item = unstack_shapes.setdefault(key,dict(shapes=shapes,count=0,flops=0))
                item['count'] += 1
            else:
                non_unstack_hash.update(encoded)
        require(before == (int(ctx.flops_used),len(ctx.op_log),float(ctx.wall_time_s))
            and flops == record['flops'] and zero+charged == record['operation_count'],
            'Passive metadata scan cannot change the closed public bill')
        ledger = dict(schema='closed_operation_ledger_v1',after_context_exit=True,
            unchanged_closed_context_bill=True,operation_count=zero+charged,total_flops=flops,
            charged_operations=charged,zero_flop_operations=zero,unstack_operations=unstack,
            unstack_total_flops=0,by_operation=[totals[k] for k in sorted(totals)],
            unstack_shapes=[unstack_shapes[k] for k in sorted(unstack_shapes)],
            all_records_sha256=all_hash.hexdigest(),non_unstack_records_sha256=non_unstack_hash.hexdigest(),
            charged_records_sha256=charged_hash.hexdigest(),
            summary_elapsed_s=time.monotonic()-began,no_array_access_or_candidate_mutation=True)
        require(len(json.dumps(ledger,separators=(',',':'))) < 256*1024, 'Bounded passive ledger metadata')
        record['operation_ledger'] = ledger
        return record
    return metrics


def _v2_validate_ledgers(result):
    """Stdlib arithmetic/metadata only, also callable by the independent supervisor."""
    for record in result['trials']+result['input_preparation']:
        x = record['operation_ledger']
        require(x['schema'] == 'closed_operation_ledger_v1' and x['after_context_exit']
            and x['unchanged_closed_context_bill'] and x['no_array_access_or_candidate_mutation']
            and x['operation_count'] == record['operation_count']
            and x['total_flops'] == record['flops']
            and x['charged_operations']+x['zero_flop_operations'] == record['operation_count']
            and sum(v['count'] for v in x['by_operation']) == record['operation_count']
            and sum(v['flops'] for v in x['by_operation']) == record['flops']
            and sum(v['count'] for v in x['unstack_shapes']) == x['unstack_operations']
            and all(v['flops'] == 0 for v in x['unstack_shapes']) and x['unstack_total_flops'] == 0
            and all(re.fullmatch('[0-9a-f]{64}',x[k]) for k in
                ('all_records_sha256','non_unstack_records_sha256','charged_records_sha256')),
            'Every passive ledger reconciles the unchanged full actual bill')
    return True


def _v2_failed_replay(result, prior, tick=None):
    checks = []
    for call,(a,b) in enumerate(zip(result['trials'],prior['trials'])):
        require(a['call'] == b['call'] == call and a['rank'] == b['rank'] == SEQUENCE[call],
            'Exact five-call negative-attempt replay mapping')
        require(all(a[k] == b[k] for k in ('flops','multiplier','includes_initialization'))
            and a['output_sha256'] == b['output_sha256'] and a['output_bytes_sha256'] == b['output_bytes_sha256']
            and Path(a['output']).read_bytes() == Path(b['output']).read_bytes(),
            'Fresh own complete F and output bytes equal the saved attempt; old result stays negative')
        da = read(result['diagnostic_records'][call]['path'])
        db = read(prior['diagnostic_records'][call]['path'])
        require(all(da[k] == db[k] for k in ('domains','startup','dispatches')),
            'All five actual16/47/startup/dispatch packets replay exactly')
        require(read(a['input_bill'])['flops'] == read(b['input_bill'])['flops'] == 16777216,
            'Own canonical input-copy billing retained')
        checks.append(dict(call=call,rank=a['rank'],output_byte_exact=True,complete_flops_exact=True,
            domains_exact=True,startup_exact=True,dispatches_exact=True,
            old_operation_count=b['operation_count'],new_operation_count=a['operation_count'],
            operation_count_delta=a['operation_count']-b['operation_count'],old_receipt_passed=False))
        if tick is not None:
            tick()
    require(len(checks) == 5 and _v2_validate_ledgers(result), 'All fresh model/input ledgers complete')
    return checks


_V2_MAIN_EDITS = [
    ("        prior_probe,result['prerequisite_seals'] = prerequisites(result['input_bindings'],tick)",
     "        prior_probe,result['prerequisite_seals'] = prerequisites(result['input_bindings'],tick)\n"
     "        prior_failed,result['failed_attempt_seals'] = _v2_failed_attempt(result['input_bindings'],tick)\n"
     "        result.update(prior_failed_attempt_preserved=True,v2_source_inverse_proof=V2_SOURCE_INVERSE_PROOF)"),
    ("finish_call=finish_call,retire_call=retire_call)",
     "finish_call=finish_call,retire_call=retire_call,metrics=_v2_ledger_metrics(legacy.metrics))"),
    ("        result['unaffected_prefix_checks'] = prefix_checks",
     "        result['unaffected_prefix_checks'] = prefix_checks\n"
     "        result['failed_attempt_parity_checks'] = _v2_failed_replay(result,prior_failed)\n"
     "        result.update(all5_failed_attempt_outputs_flops_domains_dispatches_exact=True,\n"
     "            operation_ledgers_verified=True,operation_counts_reported_not_rebased=True,\n"
     "            complete_bill_equality_fields=['flops','multiplier','includes_initialization'])"),
]
_V2_EDITS = [
    ("HERE/'check4.py'","HERE/'check4_v2.py'",1),
    ("HERE/'watch_check4.py'","HERE/'watch_check4_v2.py'",1),
    ("HERE/'CHECK4_PROTOCOL.md'","HERE/'CHECK4_V2_PROTOCOL.md'",1),
    ("HERE/'CHECK4_WATCH_SPEC.md'","HERE/'CHECK4_V2_WATCH_SPEC.md'",1),
    ("HERE/'check4_v1'","HERE/'check4_v2'",1),
    ("current_comp_shared_old384_native_check4_v1","current_comp_shared_old384_native_check4_v2",1),
    ("PREREQUISITES = {","EXTRA_SOURCES.update(_V2_METADATA_SOURCES)\n\nPREREQUISITES = {",1),
    ("keys = ('flops','operation_count','multiplier','includes_initialization')",
     "keys = ('flops','multiplier','includes_initialization')",1),
    ("'Own deterministic complete bills reproduce the positive probe'",
     "'Own complete FLOPs/multiplier/initialization reproduce the positive probe'",1),
    ("complete_bills_exact","complete_flops_exact",2),
    ("        complete_flops_exact=True,domains_exact=True,dispatches_exact=True)",
     "        complete_flops_exact=True,domains_exact=True,dispatches_exact=True,\n"
     "        operation_count_history=dict(prior=b['operation_count'],current=a['operation_count'],\n"
     "            delta=a['operation_count']-b['operation_count'],prior_sequence=[37,37],\n"
     "            current_sequence=list(SEQUENCE),equality_required=False))",1),
    ("_derived = _original","_changes.extend(_V2_MAIN_EDITS)\n_derived = _original",1),
]
_V2_DERIVED = _V2_ORIGINAL
for _v2_before,_v2_after,_v2_count in _V2_EDITS:
    if _V2_DERIVED.count(_v2_before) != _v2_count:
        raise RuntimeError('Unique v2 metadata adaptation: '+_v2_before[:100])
    _V2_DERIVED = _V2_DERIVED.replace(_v2_before,_v2_after)
_V2_INVERSE = _V2_DERIVED
for _v2_before,_v2_after,_v2_count in reversed(_V2_EDITS):
    if _V2_INVERSE.count(_v2_after) != _v2_count:
        raise RuntimeError('Unique v2 metadata inverse: '+_v2_after[:100])
    _V2_INVERSE = _V2_INVERSE.replace(_v2_after,_v2_before)
if _V2_INVERSE != _V2_ORIGINAL:
    raise RuntimeError('Entire frozen negative-v1 source restored byte-exact')
ast.parse(_V2_DERIVED)
V2_SOURCE_INVERSE_PROOF = dict(owner=str(_V2_OWNER),owner_sha256=_V2_OWNER_SHA,
    whole_source_inverse_verified=True,derived_sha256=hashlib.sha256(_V2_DERIVED.encode()).hexdigest(),
    candidate_and_paid_loop_unchanged=True,passive_post_context_ledger_only=True,
    old_negative_receipts_preserved=True,splices=len(_V2_EDITS))
exec(compile(_V2_DERIVED,str(_V2_OWNER)+':fresh-v2-accounting-metadata','exec'),globals())
