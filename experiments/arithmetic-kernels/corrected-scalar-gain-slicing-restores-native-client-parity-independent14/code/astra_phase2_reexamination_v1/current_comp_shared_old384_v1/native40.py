"""Conditional unchanged native40; direct positive CHECK4 boundary, no new fit."""
import ast
import hashlib
from pathlib import Path
import runpy

_N40_HERE = Path(__file__).resolve().parent
_N40_OWNER = _N40_HERE/'check4_v2.py'
_N40_OWNER_SHA = '8d09d9b1f443e229b1641d277312b52c204392648b25445ffab63b82dba16058'
if hashlib.sha256(_N40_OWNER.read_bytes()).hexdigest() != _N40_OWNER_SHA:
    raise RuntimeError('Frozen CHECK4 v2 arithmetic/lifecycle source')
_N40_PARENT = runpy.run_path(str(_N40_OWNER),run_name='inert_native40_parent')
for _n40_key,_n40_value in _N40_PARENT.items():
    if not _n40_key.startswith('__') and not _n40_key.startswith('_N40'):
        globals()[_n40_key] = _n40_value
_N40_RANKS = (37,39,42,48,59,63,66,67,71,73,75,79,93,101,122,130,132,135,139,145,
    146,149,150,151,155,159,163,169,173,175,181,192,198,199,204,219,224,227,233,234)
RANKS,SEQUENCE = _N40_RANKS,_N40_RANKS+(37,)
GATE_KEYS = ('check4_native','check4_native_watch','check4_score','check4_watch')
_N40_SOURCES = {
    _N40_OWNER:_N40_OWNER_SHA,
    _N40_HERE/'watch_check4_v2.py':'982001c397860137a566bca4c6632611e5050724ff09bf97134781948668a194',
    _N40_HERE/'CHECK4_V2_PROTOCOL.md':'e78c5f2e98fcd170817dfdec689fb9cf5853753f8cd18ccabcc3c475097a8d87',
    _N40_HERE/'CHECK4_V2_WATCH_SPEC.md':'dc1e59578a782235817d40b4ff3829b974ec2cb54c2955bd0684a10a0274c26f',
    _N40_HERE/'score_check4_v2.py':'50279253cdd42322fdf0db7e73a29aa91b7807e8fa9e53b97c77e79c3192f076',
    _N40_HERE/'watch_score_check4_v2.py':'708731161669da955f3640bc6e0fb0bc4a1d8631da54233dce4fcd30c0b79faa',
    _N40_HERE/'SCORE_CHECK4_V2_PROTOCOL.md':'2a573943154837389ccc8314cd46fae967ccbb76a0e61ac3343268a29408dd4a',
}
EXTRA_SOURCES = {**EXTRA_SOURCES,**_N40_SOURCES}


def _n40_prerequisites(args, inputs, tick=None):
    """Only direct actual positive native/score/watch seals; never decode their truths."""
    records,seals,union = {},{},{str(p):h for p,h in _N40_SOURCES.items()}
    stages = (('native','check4_v2','check4_native',660,8_000_000_000,False),
        ('native_watch','check4_watch_v2','check4_native_watch',660,8_000_000_000,True),
        ('score','score_check4_v2','check4_score',120,1_000_000_000,False),
        ('score_watch','score_check4_watch_v2','check4_watch',120,1_000_000_000,True))
    for key,folder,arg,limit,rss,watched in stages:
        expected = getattr(args,arg+'_sha256')
        require(re.fullmatch('[0-9a-f]{64}',expected or ''), 'Explicit actual positive CHECK4 boundary')
        paths = [HERE/folder/n for n in ('result.json','completion.json','finalization.json')]
        digests = tuple(sha(p,tick) for p in paths); r,e,f = map(read,paths)
        require(digests[0] == expected and all(x.get('passed') is True and x.get('completed') is True for x in (r,e,f))
            and e['result_sha256'] == f['result_sha256'] == expected and f['completion_sha256'] == digests[1]
            and 0 <= r['elapsed_s'] <= f['elapsed_s'] < limit and r['hashes_unchanged'] and r['sources_unchanged'],
            'Actual completed positive three-seal chain in its original scope')
        if watched:
            require(all(r[k] is True for k in ('whole_owned_tree_retired','cleanup_passed',
                'observational_groups_retired','actual_completed_samples','foreign_numerical_workers_checked'))
                and r['numerical_imports'] is False and r['maximum_swap_growth_bytes'] == 0
                and 0 < r['maximum_interval_s'] <= 1 and 0 < r['maximum_combined_rss_bytes'] < rss,
                'Original independently sampled resource and process-retirement gate')
        else:
            require(0 < r['peak_rss_bytes'] < rss, 'Original actual worker resource bound')
        for group in ('source_bindings','input_bindings','artifacts'):
            require(isinstance(r[group],dict) and r[group], 'Complete direct non-target inventory')
            for path,digest in r[group].items():
                require(path not in union or union[path] == digest, 'Consistent direct prerequisite inventory')
                union[path] = digest
        # truth_bindings are deliberately not visited or copied.
        union.update({str(p):h for p,h in zip(paths,digests)})
        records[key] = r
        seals[key] = dict(zip(('result_sha256','completion_sha256','finalization_sha256'),digests))
    for child,watch in (('native','native_watch'),('score','score_watch')):
        require(all(records[watch]['child_'+k] == v for k,v in seals[child].items()),
            'Actual independent watch owns all three exact producer seals')
    r,w,s,sw = (records[k] for k in ('native','native_watch','score','score_watch'))
    require(r['schema'] == 'current_comp_shared_old384_native_check4_v2'
        and w['schema'] == 'current_comp_shared_old384_native_check4_watch_v2'
        and r['candidate_sha256'] == w['candidate_sha256'] == COMPOSED_SHA
        and r['ranks'] == w['ranks'] == [37,39,42,48,37]
        and r['full_model_calls'] == r['retired_call_count'] == w['retired_calls'] == 5
        and len(r['trials']) == len(r['input_preparation']) == len(r['diagnostic_records']) == 5
        and len(r['artifacts']) == 35 and r['initialization_flops'] == 1944970298
        and r['truth_accessed'] is False and r['score_measured'] is False and r['parameter_updates'] == 0
        and all(r.get(k) is True for k in ('all16_domains_qualified','all47_selectors_qualified','own_complete_bills',
            'repeat_byte_exact','probe37_output_byte_exact','probe37_complete_flops_exact','probe37_domains_exact',
            'probe37_dispatches_exact','operation_ledgers_verified','operation_counts_reported_not_rebased'))
        and r['composition_proof'] == w['composition_proof'] and _v2_validate_ledgers(r),
        'Fresh positive complete fixed CHECK4 acquisition, not the negative earlier attempt')
    native_paths = tuple(_N40_SOURCES)[:4]
    require(all(r['source_bindings'].get(str(p)) == w['source_bindings'].get(str(p)) == _N40_SOURCES[p]
        for p in native_paths), 'Exact reviewed CHECK4 native owners')
    require(s['schema'] == 'current_comp_shared_old384_score_check4_v2'
        and sw['schema'] == 'current_comp_shared_old384_score_check4_watch_v2'
        and s['ranks'] == [37,39,42,48] and len(s['records']) == 4 and s['repeat_excluded']
        and s['full_model_calls'] == s['parameter_updates'] == 0 and s['no_model_imported']
        and s['lineage']['probe'] == seals['native'] and s['lineage']['probe_watch'] == seals['native_watch']
        and all(s.get(k) is True for k in ('integrity_passed','information_promising','scientific_gate_passed',
            'complete_probe_acquisition_reverified','all_predictions_verified_before_truth',
            'exact_APUSH_score_replay','own_complete_bills'))
        and sw['information_promising'] is True and s['comparison']['adjusted_ratio'] <= .97
        and s['comparison']['adjusted_wins'] >= 3 and s['comparison']['maximum_adjusted_ratio'] <= 1.10,
        'Actual prospective CHECK4 own-adjusted .97/3-of-4/1.10 scientific gate')
    require(all(s['source_bindings'].get(str(p)) == _N40_SOURCES[p] for p in _N40_SOURCES),
        'Exact reviewed scorer and acquired native source boundary')
    fresh = {}
    for path,digest in union.items():
        require(path not in inputs or inputs[path] == digest, 'Consistent new native40 inputs')
        if path not in inputs:
            fresh[path] = digest
    bind(fresh,tick); inputs.update(union)
    return r,seals


def _n40_store_rows(inputs, tick=None):
    """Use actual canonical source pins in the unchanged positive APUSH native40."""
    path = APUSH/'native40_terminal_v1/result.json'
    require(sha(path,tick) == CONTROL_SHA, 'Exact full current APUSH physical roster boundary')
    old = read(path)
    require(old['passed'] and old['completed'] and old['candidate_sha256'] == PARENT_SHA
        and old['ranks'] == list(RANKS) and [x['rank'] for x in old['trials']] == list(SEQUENCE),
        'Original fixed40 plus repeat37; no adaptive roster or new inputs')
    result = {}
    for call,rank in enumerate(RANKS):
        row = STORE/f'rows/observer_fit/{rank:04d}/row_manifest.json'
        weights = row.parent/'weights.npy'
        row_sha,weight_sha = old['source_bindings'][str(row)],old['source_bindings'][str(weights)]
        require(sha(row,tick) == row_sha, 'Canonical row manifest from the original40 owner')
        trial = old['trials'][call]; bill = Path(trial['input_bill'])
        require(sha(bill,tick) == old['artifacts'][str(bill)]
            and read(bill)['weight_file_sha256'] == weight_sha, 'Same actual physical weight file')
        result[rank] = row_sha,weight_sha
        inputs[str(row)] = row_sha
    return result


def _n40_parity(result, prior, tick=None):
    """Same-history first4 exact; final37 exact physics with honest cache-history counts."""
    checks = []
    for call,old_call in ((0,0),(1,1),(2,2),(3,3),(40,4)):
        a,b = result['trials'][call],prior['trials'][old_call]
        require(a['rank'] == b['rank'] == SEQUENCE[call]
            and all(a[k] == b[k] for k in ('flops','multiplier','includes_initialization'))
            and a['output_sha256'] == b['output_sha256'] and a['output_bytes_sha256'] == b['output_bytes_sha256']
            and Path(a['output']).read_bytes() == Path(b['output']).read_bytes(),
            'Own full F and exact saved output replay of the qualified same candidate')
        da = read(result['diagnostic_records'][call]['path'])
        db = read(prior['diagnostic_records'][old_call]['path'])
        require(da['domains'] == db['domains'] and da['startup'] == db['startup'],
            'Exact all16/all47 and startup semantics on identical physical inputs')
        def physical(items):
            return [{k:v for k,v in item.items() if k != 'operation_count'} for item in items]
        require(physical(da['dispatches']) == physical(db['dispatches']),
            'Every actual dispatch shape, complete F, seven child tariffs and arena invariant exact')
        if call < 4:
            require(da['dispatches'] == db['dispatches'] and a['operation_count'] == b['operation_count'],
                'Exact same-history first4 dispatch and operation-count replay')
            require({k:v for k,v in a['operation_ledger'].items() if k != 'summary_elapsed_s'} ==
                {k:v for k,v in b['operation_ledger'].items() if k != 'summary_elapsed_s'},
                'Exact same-history first4 passive typed-operation ledgers')
        checks.append(dict(call=call,check4_call=old_call,rank=a['rank'],output_byte_exact=True,
            complete_flops_exact=True,domains_exact=True,startup_exact=True,physical_dispatch_exact=True,
            same_history=(call < 4),old_operation_count=b['operation_count'],
            new_operation_count=a['operation_count'],operation_count_delta=a['operation_count']-b['operation_count'],
            dispatch_operation_counts=dict(prior=[x['operation_count'] for x in db['dispatches']],
                current=[x['operation_count'] for x in da['dispatches']]),
            operation_count_equality_required=(call < 4)))
        if tick is not None:
            tick()
    require(_v2_validate_ledgers(result), 'All82 closed input/model ledgers reconcile their own bills')
    return checks


_N40_ORIGINAL = _N40_PARENT['_derived']
_N40_OLD_PARITY = """        result['failed_attempt_parity_checks'] = _v2_failed_replay(result,prior_failed)
        result.update(all5_failed_attempt_outputs_flops_domains_dispatches_exact=True,
            operation_ledgers_verified=True,operation_counts_reported_not_rebased=True,
            complete_bill_equality_fields=['flops','multiplier','includes_initialization'])
        result['probe37_parity_checks'] = [probe37_check(np,result,prior_probe,call) for call in (0,4)]
        result.update(probe37_output_byte_exact=True,probe37_complete_flops_exact=True,
            probe37_domains_exact=True,probe37_dispatches_exact=True)"""
_N40_EDITS = [
    ("BUDGET, CAP, RSS = 2**41, 660., 8_000_000_000","BUDGET, CAP, RSS = 2**41, 3000., 8_000_000_000",1),
    ("KEYS = ('source','watch','protocol','watch_spec','compose','fnp','design')",
     "KEYS = ('source','watch','protocol','watch_spec','compose','fnp','design')+GATE_KEYS",1),
    ("HERE/'check4_v2.py'","HERE/'native40.py'",1),
    ("HERE/'watch_check4_v2.py'","HERE/'watch_native40.py'",1),
    ("HERE/'CHECK4_V2_PROTOCOL.md'","HERE/'NATIVE40_PROTOCOL.md'",1),
    ("HERE/'CHECK4_V2_WATCH_SPEC.md'","HERE/'NATIVE40_WATCH_SPEC.md'",1),
    ("HERE/'check4_v2'","HERE/'native40_v1'",1),
    ("current_comp_shared_old384_native_check4_v2","current_comp_shared_old384_native40_v1",1),
    ("check4_source_inverse_proof=SOURCE_INVERSE_PROOF","native40_source_inverse_proof=NATIVE40_SOURCE_INVERSE_PROOF",1),
    ("Whole native stage660s/process8GB","Whole native stage3000s/process8GB",1),
    ("        prior_probe,result['prerequisite_seals'] = prerequisites(result['input_bindings'],tick)\n"
     "        prior_failed,result['failed_attempt_seals'] = _v2_failed_attempt(result['input_bindings'],tick)\n"
     "        result.update(prior_failed_attempt_preserved=True,v2_source_inverse_proof=V2_SOURCE_INVERSE_PROOF)\n"
     "        result['probe_prerequisite_qualified'] = True",
     "        prior_check4,result['check4_prerequisite_seals'] = _n40_prerequisites(args,result['input_bindings'],tick)\n"
     "        result['check4_prerequisite_qualified'] = True",1),
    ("        for rank,(row_sha,weight_sha) in ROWS.items():",
     "        for rank,(row_sha,weight_sha) in _n40_store_rows(result['input_bindings'],tick).items():",1),
    ("    for index, rank, expected in ((0,37,461231998786),(1,39,482950877279),(2,42,445415645868),\n"
     "            (3,48,456127497800),(40,37,459287028488)):",
     "    require(record['ranks'] == list(RANKS) and len(record['trials']) == 41, 'Exact original full40 control')\n"
     "    for index,rank in enumerate(SEQUENCE):\n        expected = record['trials'][index]['flops']",1),
    ("if call == 4:  # Only the final37 is a repeated network.",
     "if call == 40:  # Only the final37 is a repeated network.",1),
    ("require(len(result['trials']) == len(result['diagnostic_records']) == 5, 'All five complete fixed calls')",
     "require(len(result['trials']) == len(result['diagnostic_records']) == 41, 'All41 complete fixed calls')",1),
    (_N40_OLD_PARITY,
     "        result['check4_parity_checks'] = _n40_parity(result,prior_check4)\n"
     "        result.update(check4_first4_exact=True,check4_repeat37_physics_exact=True,\n"
     "            operation_ledgers_verified=True,operation_counts_reported_not_rebased=True,\n"
     "            repeat_operation_counts_history_aware=True)",1),
    ("result['trials'][0]['flops']-result['trials'][4]['flops']",
     "result['trials'][0]['flops']-result['trials'][40]['flops']",1),
    ("result['full_model_calls'] = result['retired_call_count'] = 5",
     "result['full_model_calls'] = result['retired_call_count'] = 41",1),
]
_N40_DERIVED = _N40_ORIGINAL
for _n40_before,_n40_after,_n40_count in _N40_EDITS:
    if _N40_DERIVED.count(_n40_before) != _n40_count:
        raise RuntimeError('Unique native40 metadata span '+_n40_before[:100])
    _N40_DERIVED = _N40_DERIVED.replace(_n40_before,_n40_after)
_N40_INVERSE = _N40_DERIVED
for _n40_before,_n40_after,_n40_count in reversed(_N40_EDITS):
    if _N40_INVERSE.count(_n40_after) != _n40_count:
        raise RuntimeError('Unique native40 inverse span '+_n40_after[:100])
    _N40_INVERSE = _N40_INVERSE.replace(_n40_after,_n40_before)
if _N40_INVERSE != _N40_ORIGINAL:
    raise RuntimeError('Entire frozen CHECK4 v2 derived source restored byte-exact')
ast.parse(_N40_DERIVED)
NATIVE40_SOURCE_INVERSE_PROOF = dict(owner=str(_N40_OWNER),owner_sha256=_N40_OWNER_SHA,
    whole_source_inverse_verified=True,derived_sha256=hashlib.sha256(_N40_DERIVED.encode()).hexdigest(),
    candidate_paid_loop_observers_and_closed_ledger_unchanged=True,
    fixed_roster=list(SEQUENCE),whole_cap_s=3000.,same_history_first4_exact=True,
    repeat_operation_count_history_aware=True,positive_CHECK4_objective_required=True)
exec(compile(_N40_DERIVED,str(_N40_OWNER)+':conditional-native40-metadata','exec'),globals())
