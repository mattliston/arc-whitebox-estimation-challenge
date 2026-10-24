"""Inert projected TRAIN20 / fresh CHECK5 capture, never a training worker.

The original capture and projection qualification sources remain immutable.
Only explicit --run can execute the derived, qualified paid loop.
"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
CAPTURE = HERE/'capture_worker.py'
CAPTURE_SHA = '44d750e051c7af0590600ade0dc653fe9cc58da67a077ca2bbde16acdbf1a6ba'
QUALIFIER = HERE/'projection_worker_v2.py'
QUALIFIER_SHA = 'cbdf5fe33d7c36b3e0f5a424d9589d3e2866c595f6d0a1f38449df6db84b9ee2'
ADAPTER_SHA = '3bcefbe9503e8457cc8be2d9dcf3d23effd5b0217bd44dbe9c3a3fd955d29cf0'
ADAPTER_PROTOCOL_SHA = '8730fc7b1fc11bcd8f9571667710034cca0251da9ba992caa8263d948ba7871d'
CANDIDATE_SHA = 'c50070d4f1dfd47cf7148dbee4c0ca7119c42f114ff9df05bc5fe4806d8d4f24'
ADDITIVE_PINS = {
    CAPTURE:CAPTURE_SHA,
    HERE/'capture_watch.py':'31192d37422d81922cc06ddd17d22b8f7d3d2655cff29ea6fd93fc63d0c796dd',
    HERE/'CAPTURE_PROTOCOL.md':'df89746753d30b055f44abd587a9f79ae93735068b12b4702de6bcdedd3ad50c',
    HERE/'CAPTURE_WATCH_SPEC.md':'cf05c859c9eaba6ca38c9aac9e350f44bee43e6a46641fd36487fb64f61e6ac1',
    QUALIFIER:QUALIFIER_SHA,
    HERE/'projection_watch_v2.py':'c59d73a71278a1935b1c92645de158e671cacbdd912ad3a61775bf7fd25e5cb4',
    HERE/'PROJECTION_V2_HARNESS_REPAIR.md':'6ba99981a71ef0f2a0d59961258e419b75e989713f4a36b2ef624f3335fbd670',
    HERE/'projection_worker.py':'d6e50c6ea61e9318864727a5ae9fd8fcb74efabea708391a8a358d6b7a078bd4',
    HERE/'projection_watch.py':'5c5b73152bd926aec8590c4bac74f15fb6f442e53669e8dd05300582e014309c',
    HERE/'projection_adapter.py':ADAPTER_SHA,
    HERE/'PROJECTION_PROTOCOL.md':ADAPTER_PROTOCOL_SHA,
    HERE/'PROJECTION_WATCH_SPEC.md':'0da33a0935f596b694279552f7f8099ffa202d8b1ee29c7ca601e5e7852f9358',
}
TRAIN = (35,36,38,40,41,43,44,45,46,47,49,50,51,52,53,54,55,56,57,58)
CHECK = (37,39,42,48,37)
CAPS = {'train':2460.,'check':660.}
_OWNERS = {}
_HELPERS = None
_HELPER_PROOF = None


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def owner(path, digest):
    """Pinned stdlib-only source provider; never an estimator import."""
    key = str(path)
    if key not in _OWNERS:
        require(not path.is_symlink() and hashlib.sha256(path.read_bytes()).hexdigest() == digest,
                'Immutable source provider '+key)
        spec = importlib.util.spec_from_file_location('projected_capture_'+path.stem,path)
        value = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(value)
        _OWNERS[key] = value
    return _OWNERS[key]


def rewrite(original, edits):
    """Per-occurrence positional inverse, safe against overlapping basenames."""
    changed, history = original, []
    for old,new,count in edits:
        require(old and changed.count(old) == count, 'Exact projected-capture source splice '+old[:90])
        positions, cursor = [], 0
        while True:
            at = changed.find(old,cursor)
            if at < 0:
                break
            positions.append(at); cursor = at+len(old)
        pieces, cursor, shift = [], 0, 0
        for at in positions:
            pieces.extend((changed[cursor:at],new))
            history.append((at+shift,old,new))
            cursor = at+len(old); shift += len(new)-len(old)
        pieces.append(changed[cursor:]); changed = ''.join(pieces)
    inverse = changed
    for at,old,new in reversed(history):
        require(inverse[at:at+len(new)] == new, 'Exact recorded inverse span')
        inverse = inverse[:at]+old+inverse[at+len(new):]
    require(inverse == original, 'Whole original source byte inverse')
    return changed


def selected(source, name):
    nodes = [n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name == name]
    require(len(nodes) == 1, 'Unique exact source-owned function '+name)
    n = nodes[0]
    return ''.join(source.splitlines(keepends=True)[n.lineno-1:n.end_lineno])


def _helpers():
    """Private fresh globals: no caller merge can mutate an old owner's globals."""
    global _HELPERS, _HELPER_PROOF
    if _HELPERS is not None:
        return _HELPERS
    original = CAPTURE.read_text()
    require(hashlib.sha256(original.encode()).hexdigest() == CAPTURE_SHA
        and original.count('def derive_source():') == 1, 'Frozen full original capture owner')
    packet = original.split('def derive_source():',1)[0]
    edits = [
        ("HERE/'capture_worker.py'","HERE/'projection_capture_worker.py'",1),
        ("HERE/'capture_watch.py'","HERE/'projection_capture_watch.py'",1),
        ("HERE/'CAPTURE_PROTOCOL.md'","HERE/'PROJECTION_CAPTURE_PROTOCOL.md'",1),
        ("HERE/'CAPTURE_WATCH_SPEC.md'","HERE/'PROJECTION_CAPTURE_WATCH_SPEC.md'",1),
        ("'current_v29_temporal_acquisition_watch_v1'","'current_v29_output_projection_qualification_watch_v2'",1),
        ("'current_v29_temporal_capture_watch_v1'","'current_v29_temporal_projection_capture_watch_v1'",1),
        ("'current_v29_temporal_capture_v1'","'current_v29_temporal_projection_capture_v1'",1),
        ("'current_v29_temporal_capture_roster_v1'","'current_v29_temporal_projection_capture_roster_v1'",1),
        ("HERE/'train_capture_v1'","HERE/'train_projection_capture_v1'",1),
        ("HERE/'train_capture_watch_v1'","HERE/'train_projection_capture_watch_v1'",1),
        ("HERE/f'{stage}_capture_v1'","HERE/f'{stage}_projection_capture_v1'",1),
        ("HERE/f'{stage}_capture_v1/capture_roster.json'","HERE/f'{stage}_projection_capture_v1/capture_roster.json'",1),
        ("            'complete':folder/f'call_{call:02d}_complete.json',",
         "            'complete':folder/f'call_{call:02d}_complete.json',\n"
         "            'projection':folder/f'call_{call:02d}_projection.json',",1),
        ('Actual per-call eight-artifact inventory','Actual per-call nine-artifact inventory',1),
        ("'host_served_byte_exact'","'raw_host_projected_served_byte_exact'",1),
        ('original_two_call_qualification_passed','projection_two_call_qualification_passed',2),
        ("record['call_artifact_count'] == 8*len(sequence)","record['call_artifact_count'] == 9*len(sequence)",1),
        ('call_artifact_count=8*len(sequence)','call_artifact_count=9*len(sequence)',1),
        ('160/40 exact call-artifact declaration','180/45 exact call-artifact declaration',1),
    ]
    changed = rewrite(packet,edits)
    env = dict(__name__='projected_capture_private_metadata',__file__=str(CAPTURE))
    exec(compile(changed,str(CAPTURE)+':private-metadata-inverse','exec'),env)
    # Old OWNER_PINS were constructed with the OLD adapter constants above.
    # Only the supplier for this new stage changes; historical pins stay exact.
    projection = owner(QUALIFIER,QUALIFIER_SHA)
    scope = projection.execution_scope()
    env['ADAPTER_SHA'],env['ADAPTER_PROTOCOL_SHA'] = ADAPTER_SHA,ADAPTER_PROTOCOL_SHA
    env['OWNER_PINS'] = {**env['OWNER_PINS'],**ADDITIVE_PINS,**scope['PROJECTION_SOURCE_PINS']}
    env['_base_calls_contract'],env['_base_cold_replay'] = env['calls_contract'],env['cold_replay']
    env['_base_verify_expanded'],env['_base_complete_stage'] = env['verify_expanded'],env['complete_stage']
    env.update(calls_contract=calls_contract,cold_replay=cold_replay,qualification=qualification,
        verify_expanded=verify_expanded,complete_stage=complete_stage,derive_source=derive_source)
    _HELPER_PROOF = dict(owner_sha256=CAPTURE_SHA,whole_source_inverse=True,
        original_packet_sha256=hashlib.sha256(packet.encode()).hexdigest(),
        derived_packet_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        untouched_owner_suffix_sha256=hashlib.sha256(original[len(packet):].encode()).hexdigest(),
        private_definition_globals=True,metadata_overrides_owned_by_new_worker=True)
    _HELPERS = env
    return env


def projection_contract(record, folder, sequence, tick=None):
    h = _helpers()
    require(len(record['projection_records']) == len(sequence), 'One actual projection record per call')
    for call,rank in enumerate(sequence):
        p,trial,capture = record['projection_records'][call],record['trials'][call],record['capture_records'][call]
        path = folder/f'call_{call:02d}_projection.json'
        digest = h['sha'](path,tick)
        require(h['read'](path) == p and record['artifacts'][str(path)] == digest
            and trial['projection_statistics'] == str(path) and trial['projection_statistics_sha256'] == digest,
            'Actual projection sidecar joined to closed bill and complete artifact inventory')
        require(p['schema'] == 'current_v29_output_projection_statistics_v1'
            and p['call'] == call and p['rank'] == rank and len(p['layers']) == 16
            and p['host_semantics'] == capture['host_semantics'] == 'raw_physical_unclipped'
            and p['served_semantics'] == 'final_output_maximum_zero'
            and all(p.get(k) is True for k in ('raw_hosts_finite','projected_output_domain_passed',
                'served_equals_projected_raw_host_byte_exact','projection_tariff_qualified',
                'projection_in_complete_bill','statistics_after_context_exit','closed_bill_unchanged_by_statistics'))
            and p['truth_accessed'] is False and p['statistical_score_measured'] is False
            and p['projection_flops'] == trial['projection_flops'] == 16384
            and p['projection_operation_count'] == trial['projection_operation_count'] == 1
            and p['own_complete_flops'] == trial['flops']
            and all(trial.get(k) is True for k in ('raw_hosts_finite','raw_host_projection_byte_exact','projection_tariff_qualified')),
            'Qualified final-only projection, finite raw hosts and own complete tariff')
        require(len(p['projection_operations']) == 1, 'Single paid maximum, never an unpaid output repair')
        op = p['projection_operations'][0]
        require(op['op_name'] == 'maximum' and op['shapes'] == [[16,1024],[]]
            and op['resolved_dtype'] == 'float32' and op['flops'] == 16384
            and 0 < op['cumulative'] <= trial['flops'], 'Exact original FNP projection tariff')
        for layer,row in enumerate(p['layers']):
            require(row['layer'] == layer and row['raw_host']['nonfinite_count'] == 0
                and row['served']['nonfinite_count'] == row['served']['negative_count'] == 0
                and row['served_equals_projected_raw_host_byte_exact'] is True
                and row['raw_host_sha256'] == capture['arrays'][f'layer_{layer:02d}_host']['raw_sha256']
                and row['projected_raw_host_sha256'] == row['served_sha256']
                and h['valid_digest'](row['served_sha256']), 'All16 raw-host/served bytes and domains')
        raw_nonnegative = all(row['raw_host']['negative_count'] == 0 for row in p['layers'])
        require(p['original_unclipped_domain_passed'] is raw_nonnegative
            and trial['original_unclipped_domain_passed'] is raw_nonnegative,
            'Original unclipped-domain outcome reported, never forced positive')


def calls_contract(record, folder, sequence, expanded=False, tick=None):
    _helpers()['_base_calls_contract'](record,folder,sequence,expanded,tick)
    projection_contract(record,folder,sequence,tick)


def negative_pins():
    return dict(owner(QUALIFIER,QUALIFIER_SHA).execution_scope()['NEGATIVE_PINS'])


def qualification(args, sources, inputs, tick=None):
    h = _helpers()
    folder,watched = HERE/'projection_v2',HERE/'projection_watch_v2'
    q,qs = h['chain'](folder,args.qualification_sha256,inputs,300.,tick)
    w,ws = h['chain'](watched,args.qualification_watch_sha256,inputs,300.,tick)
    require(q['schema'] == 'current_v29_output_projection_qualification_v2'
        and q['candidate_sha256'] == CANDIDATE_SHA
        and all(q.get(k) is True for k in ('hashes_unchanged','sources_unchanged','all16_raw_domains_qualified',
            'all16_features_qualified','own_complete_bills','estimator_retired','setup_owned','raw_hosts_preserved',
            'all16_projection_byte_exact','projection_tariff_qualified','original_negative_acquisition_preserved'))
        and q['feature_columns'] == 30 and q['projection_flops'] == [16384,16384]
        and q['production_residual_qualified'] is False and q['downstream_training_authorized'] is False,
        'Actual projected two-call execution qualification, no accuracy or fit permission')
    p = owner(QUALIFIER,QUALIFIER_SHA)
    required = {**p.execution_scope()['PROJECTION_SOURCE_PINS'],QUALIFIER:QUALIFIER_SHA,
        HERE/'projection_watch_v2.py':ADDITIVE_PINS[HERE/'projection_watch_v2.py'],
        HERE/'projection_adapter.py':ADAPTER_SHA,HERE/'PROJECTION_PROTOCOL.md':ADAPTER_PROTOCOL_SHA,
        HERE/'PROJECTION_WATCH_SPEC.md':ADDITIVE_PINS[HERE/'PROJECTION_WATCH_SPEC.md']}
    for path,digest in required.items():
        require(q['source_bindings'].get(str(path)) == digest, 'Exact actual projected qualification source owner')
    for path,digest in negative_pins().items():
        require(q['input_bindings'].get(path) == digest, 'Both immutable failed six-seal histories bound')
        h['add'](inputs,path,digest,tick)
        prior = h['read'](path)
        require(prior.get('passed') is False and prior.get('completed') is False,
                'Unprojected and projection-v1 failures remain FALSE')
    h['watcher_contract'](w,q,qs,(37,37))
    h['inventory'](q,sources,inputs,tick); h['inventory'](w,sources,inputs,tick)
    calls_contract(q,folder,(37,37),tick=tick); h['repeat_contract'](q,None)
    require(q['original_unclipped_domain_passed'] == [x['original_unclipped_domain_passed'] for x in q['projection_records']],
            'Honest original raw-host sign outcome')
    return q,dict(qualification=qs,qualification_watch=ws)


def cold_replay(record, qualified):
    answer = _helpers()['_base_cold_replay'](record,qualified)
    old,new = qualified['projection_records'][0],record['projection_records'][0]
    require(all(old[k] == new[k] for k in ('layers','host_semantics','served_semantics',
        'projection_flops','projection_operation_count','original_unclipped_domain_passed')),
        'Fresh CHECK cold37 repeats the qualified raw hosts, output projection and diagnostics')
    answer.update(raw_host_projection_exact=True,projection_statistics_exact=True,projection_tariff_exact=True)
    return answer


def verify_expanded(record, stage, args, qualified, tick=None):
    h = _helpers()
    h['_base_verify_expanded'](record,stage,args,qualified,tick)
    sequence = TRAIN if stage == 'train' else CHECK
    require(record['candidate_sha256'] == CANDIDATE_SHA
        and all(record.get(k) is True for k in ('raw_hosts_preserved','all16_projection_byte_exact',
            'projection_tariff_qualified','original_negative_acquisition_preserved','failed_projection_v1_preserved'))
        and record['downstream_training_authorized'] is False
        and record['projection_flops'] == [16384]*len(sequence)
        and record['original_unclipped_domain_passed'] == [p['original_unclipped_domain_passed'] for p in record['projection_records']]
        and record['paid_loop_proof']['parent']['post_context_statistics_inverse'] is True
        and record['paid_loop_proof']['parent']['original_strict_served_domain_assertion_unchanged'] is True,
        'Unchanged qualified projected source and all original domain/resource gates')


def complete_stage(result, args, admitted):
    result.update(raw_hosts_preserved=True,all16_projection_byte_exact=True,projection_tariff_qualified=True,
        projection_flops=[p['projection_flops'] for p in result['projection_records']],
        original_unclipped_domain_passed=[p['original_unclipped_domain_passed'] for p in result['projection_records']],
        original_negative_acquisition_preserved=True,failed_projection_v1_preserved=True,
        downstream_training_authorized=False)
    _helpers()['_base_complete_stage'](result,args,admitted)


def derive_source():
    """Full original expanded-worker inverse plus exact qualified projection helpers."""
    original,parent = owner(CAPTURE,CAPTURE_SHA).derive_source()
    projection,qproof = owner(QUALIFIER,QUALIFIER_SHA).derive_source()
    old_paid = selected(original,'paid_loop_source')
    qualified_paid = selected(projection,'paid_loop_source')
    paid = rewrite(qualified_paid,[
        ('def paid_loop_source(owner):','def paid_loop_source(owner, stage):',1),
        ('    return changed, dict(parent=prior,setup_splice_inverse=True,post_context_statistics_inverse=True,',
         '    qualified = dict(parent=prior,setup_splice_inverse=True,post_context_statistics_inverse=True,',1),
        ('        original_strict_served_domain_assertion_unchanged=True)',
         '        original_strict_served_domain_assertion_unchanged=True)\n'
         '    return _expansion.expanded_loop(changed,stage,qualified)',1)])
    helpers = selected(projection,'projection_observe')+'\n\n'+selected(projection,'failure_retire')+'\n\n'
    old_main,new_main = selected(original,'main'),selected(projection,'main')
    old_failure = old_main[old_main.index('    except BaseException:\n'):old_main.index('    result.update(elapsed_s=')]
    new_failure = new_main[new_main.index('    except BaseException:\n'):new_main.index('    result.update(elapsed_s=')]
    old_host = """                require(arrays[f'layer_{layer:02d}_host'].tobytes() == served[layer].tobytes(),
                        'Head-free captured host equals actual served row')
"""
    new_host = """                require(np.maximum(arrays[f'layer_{layer:02d}_host'],np.float32(0)).tobytes()
                        == served[layer].tobytes(), 'Raw physical host projects to actual served row exactly')
"""
    require(new_host in projection, 'Exact already-qualified projected host comparison')
    edits = [
        ("ADAPTER_SHA = 'cab844cd7981d31c7e44f4797ad60490729bdfadb8ce46655cdd5966ae9f36b7'",f"ADAPTER_SHA = '{ADAPTER_SHA}'",1),
        ("ADAPTER_PROTOCOL_SHA = '2a93907e69427a55c9a11eef7bd2143d9f71835d62f5eb4323700c1a8bd545ba'",f"ADAPTER_PROTOCOL_SHA = '{ADAPTER_PROTOCOL_SHA}'",1),
        ("HERE/'adapter.py'","HERE/'projection_adapter.py'",2),
        ("HERE/'PROTOCOL.md': ADAPTER_PROTOCOL_SHA","HERE/'PROJECTION_PROTOCOL.md': ADAPTER_PROTOCOL_SHA",1),
        (old_paid,paid,1),
        ('def main():',helpers+'def main():',1),
        ("HERE/f'{args.stage}_capture_v1'","HERE/f'{args.stage}_projection_capture_v1'",1),
        ("schema='current_v29_temporal_capture_v1'","schema='current_v29_temporal_projection_capture_v1'",1),
        ("schema='current_v29_temporal_capture_roster_v1'","schema='current_v29_temporal_projection_capture_roster_v1'",1),
        ('        capture_records=[],full_model_calls=0,retired_call_count=0,parameter_updates=0,',
         '        capture_records=[],projection_records=[],full_model_calls=0,retired_call_count=0,parameter_updates=0,',1),
        ("    try:\n        bind(result['source_bindings'],tick)",
         "    env=observer=module=None\n    try:\n        bind(result['source_bindings'],tick)",1),
        ("require(tuple(adapter.TRAIN) == _expansion.TRAIN and tuple(adapter.CHECK)+(37,) == _expansion.CHECK, 'Frozen20/4 roster')",
         "require(tuple(adapter._base().TRAIN) == _expansion.TRAIN and tuple(adapter._base().CHECK)+(37,) == _expansion.CHECK, 'Frozen authenticated base20/4 roster')",1),
        (old_host,new_host,1),
        ('feature_F_in_actual_bill=True,host_served_byte_exact=True,all16_raw_domains=True,',
         "feature_F_in_actual_bill=True,raw_host_projected_served_byte_exact=True,\n"
         "                host_semantics='raw_physical_unclipped',all16_raw_domains=True,",1),
        ('            finish_call=finish_call,retire_call=retire_call,cold_create=cold_create)',
         '            finish_call=finish_call,retire_call=retire_call,cold_create=cold_create,\n'
         '            projection_observe=projection_observe,require=require)',1),
        ('        del env,observer,module\n        gc.collect()',
         '        env.clear(); env=observer=module=None\n        gc.collect()',1),
        (old_failure,new_failure,1),
    ]
    changed = rewrite(original,edits)
    _helpers()
    return changed,dict(parent=parent,qualification_source_proof=qproof,whole_source_inverse=True,
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),metadata_helper_proof=_HELPER_PROOF,
        qualified_paid_function_sha256=hashlib.sha256(qualified_paid.encode()).hexdigest(),
        qualified_projection_helpers_sha256=hashlib.sha256(helpers.encode()).hexdigest(),
        qualified_failure_cleanup_sha256=hashlib.sha256(new_failure.encode()).hexdigest(),
        original_fixed_stage_rosters_and_caps_unchanged=True,qualified_projection_arithmetic_unchanged=True,
        qualified_guard_binding_preserved=True,per_call_projection_artifact_preserved=True)


# Public metadata API consumed by the independently owned watcher. Functions
# return values, never the private dictionary in which definitions resolve names.
def environment():
    return _helpers()['environment']()


def configuration(args):
    return _helpers()['configuration'](args)


def worker_args(args):
    return _helpers()['worker_args'](args)


def new_owners(args):
    return _helpers()['new_owners'](args)


def prerequisites(args, sources, inputs, tick=None):
    return _helpers()['prerequisites'](args,sources,inputs,tick)


def fixture_rows(sequence, inputs, tick=None):
    return _helpers()['fixture_rows'](sequence,inputs,tick)


def add(mapping, path, digest, tick=None):
    return _helpers()['add'](mapping,path,digest,tick)


def sha(path, tick=None):
    return _helpers()['sha'](path,tick)


if __name__ == '__main__':
    require('--run' in sys.argv, 'Inert. Explicit --run --stage and reviewed actual hashes required.')
    source,proof = derive_source()
    scope = dict(__name__='__main__',__file__=str(Path(__file__).resolve()),
        _expansion=types.SimpleNamespace(**dict(_helpers())),_expansion_proof=proof)
    exec(compile(source,str(Path(__file__).resolve())+':qualified-projected-capture','exec'),scope)
