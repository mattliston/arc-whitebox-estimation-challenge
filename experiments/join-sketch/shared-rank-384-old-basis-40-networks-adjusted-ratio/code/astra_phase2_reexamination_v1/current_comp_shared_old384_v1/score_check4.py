"""Inert inverse adapter of the successful saved-output scorer; fixed CHECK4."""
import ast
import hashlib
import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BASE = _HERE/'score_probe.py'
_BASE_SHA = 'a279e9bf196c50ef3157ade817946f5f6a5f75521040f0ce03820649a8c1e7cb'
_RANKS = (37,39,42,48)
_PROBE_SCORE = 'd00066a38fc9c6012282c80a3a3d7f9a97f420b137f67e1daae5473589b88dd0'
_PROBE_WATCH = 'ea4a6847a04d82b159d3dbcea51d801e80b84628f1d11eebc22be44322de29f5'


def _require(ok,message):
    if not ok:
        raise RuntimeError(message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _adapt(original,edits):
    changed = original
    for before,after,count in edits:
        _require(changed.count(before) == count,'Unique CHECK4 score edit '+repr((before[:100],count,changed.count(before))))
        changed = changed.replace(before,after)
    inverse = changed
    for before,after,count in reversed(edits):
        _require(inverse.count(after) == count,'Unique CHECK4 score inverse '+after[:100])
        inverse = inverse.replace(after,before)
    _require(inverse == original,'Whole completed scorer byte inverse')
    ast.parse(changed)
    return changed


def _check4_prerequisite(admit):
    """Direct passed development gate, never recomputed or relabeled."""
    score,ss = admit.chain(HERE/'score_probe_v1',_PROBE_SCORE,120,1_000_000_000)
    watched,ws = admit.chain(HERE/'score_probe_watch_v1',_PROBE_WATCH,120,1_000_000_000,watch=True)
    linked(watched,ss)
    require(score['information_promising'] and score['scientific_gate_passed']
        and score['comparison']['adjusted_ratio'] <= .95 and score['exact_APUSH_score_replay']
        and score['schema'] == 'current_comp_shared_old384_score_probe_v1'
        and score['lineage']['probe']['result_sha256'] ==
            '016e81e7bd0750f0e99671ac4def7deb02f591a2a292d07185eee25b7e2c591f',
        'Actual previously passed rank37 objective decision')
    return score,dict(score=ss,watch=ws)


def derive_source():
    _require(_sha(_BASE) == _BASE_SHA,'Frozen successful saved-output scorer')
    original = _BASE.read_text()
    old_score_path = _HERE.parent/'current_comp_aligned_push_d6_v1/score40_v1/result.json'
    _require(_sha(old_score_path) == 'b9667c17e474f99e394481169f9081d5e3320ca74255e2e3e411b7eaf9995279',
             'Actual original40 score metadata only')
    old_score = json.loads(old_score_path.read_text())
    target_bindings = {str(_HERE.parent.parent/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{r:04d}.npz'):
        old_score['records'][i]['target_sha256'] for i,r in enumerate(_RANKS)}
    _require([r['rank'] for r in old_score['records'][:4]] == list(_RANKS), 'Unchanged original CHECK4 roster')
    extra = ('RANKS = '+repr(_RANKS)+'\nTRUTHS = {Path(p):h for p,h in '+repr(target_bindings)+'.items()}\n')
    target_start = original.index("    result['truth_accessed'] = True\n")
    target_end = original.index('    for path,digest in {**admit.pins,**result[\'artifacts\']}.items():\n',target_start)
    old_target = original[target_start:target_end]
    new_target = '''    result['truth_accessed'] = True
    records = []
    for call,rank in enumerate(RANKS):
        path = TRUTH.parent/f'truthpost_rank{rank:04d}.npz'
        expected = TRUTHS[path]
        require(old['truth_bindings'][str(path)] == expected, 'Original per-rank truth binding')
        require(sha(path,tick,truth=True) == expected, 'Only fixed CHECK4 target after all outputs seal')
        result['truth_bindings'][str(path)] = expected
        with np.load(path,allow_pickle=False) as archive:
            original = archive['mc_post']; dtype = str(original.dtype); target = original.astype(np.float64)
        require(original.dtype == np.float32 and target.shape == (16,1024) and np.isfinite(target).all(),
            'Original stored mc_post, promoted to FP64 without requantization')
        row = dict(call=call,rank=rank,target_sha256=expected,target_original_dtype=dtype,per_role={})
        for role in ('APUSH','shared'):
            trial = r['controls'][call] if role == 'APUSH' else r['trials'][call]
            scope = dict(np=np,arrays=arrays,role=role,call=call,target=target,trial=trial,BUDGET=BUDGET,require=require,row=row)
            exec(env['arithmetic'],scope)
        require(row['per_role']['APUSH'] == old['records'][call]['per_role']['terminal'],
                'Exact entire original APUSH per-rank score replay')
        records.append(row)
    require(records[0] == prior['record'], 'Exact previously measured rank37 cold raw/cost/adjusted row')
    summary = {role:{key:sum(row['per_role'][role][key] for row in records)/4
        for key in ('raw_mse','flops','multiplier','adjusted')} for role in ('APUSH','shared')}
    ratios = [row['per_role']['shared']['adjusted']/row['per_role']['APUSH']['adjusted'] for row in records]
    a,b = summary['shared'],summary['APUSH']
    comparison = dict(raw_ratio=a['raw_mse']/b['raw_mse'],complete_flops_ratio=a['flops']/b['flops'],
        adjusted_ratio=a['adjusted']/b['adjusted'],adjusted_improved=a['adjusted'] < b['adjusted'],
        adjusted_wins=sum(x < 1 for x in ratios),maximum_adjusted_ratio=max(ratios),per_rank_adjusted_ratios=ratios)
    promising = comparison['adjusted_ratio'] <= .97 and comparison['adjusted_wins'] >= 3 and max(ratios) <= 1.10
'''
    edits = [
        ("SCHEMA = 'current_comp_shared_old384_score_probe_v1'","SCHEMA = 'current_comp_shared_old384_score_check4_v1'",1),
        ('PINS = {\n','PINS = {\n    _BASE:_BASE_SHA,\n',1),
        ('TRUTH_SHA = \'48fe26d81da4b12cfa07f9ef310fe8ccbbd28558cbe61d0290ac53293fd04f0d\'\n',
         'TRUTH_SHA = \'48fe26d81da4b12cfa07f9ef310fe8ccbbd28558cbe61d0290ac53293fd04f0d\'\n'+extra,1),
        ('(truth and path == TRUTH)','(truth and path in TRUTHS)',1),
        ("HERE/'probe_v1'","HERE/'check4_v1'",3),
        ("HERE/'probe_watch_v1'","HERE/'check4_watch_v1'",1),
        ("args.probe_sha256,300,8_000_000_000","args.probe_sha256,660,8_000_000_000",1),
        ("args.probe_watch_sha256,300,8_000_000_000","args.probe_watch_sha256,660,8_000_000_000",1),
        ("'current_comp_shared_old384_native37_v1'","'current_comp_shared_old384_native_check4_v1'",1),
        ("'current_comp_shared_old384_native37_watch_v1'","'current_comp_shared_old384_native_check4_watch_v1'",1),
        ("r['ranks'] == w['ranks'] == [37,37]","r['ranks'] == w['ranks'] == [37,39,42,48,37]",1),
        ("r['full_model_calls'] == r['retired_call_count'] == w['retired_calls'] == 2",
         "r['full_model_calls'] == r['retired_call_count'] == w['retired_calls'] == 5",1),
        ("HERE/'probe.py'","HERE/'check4.py'",2),
        ("HERE/'watch_probe.py'","HERE/'watch_check4.py'",2),
        ("HERE/'PROBE_PROTOCOL.md'","HERE/'CHECK4_PROTOCOL.md'",2),
        ("HERE/'PROBE_WATCH_SPEC.md'","HERE/'CHECK4_WATCH_SPEC.md'",2),
        ("len(r['artifacts']) == 14 and len(r['trials']) == len(r['input_preparation']) == len(r['diagnostic_records']) == 2",
         "len(r['artifacts']) == 35 and len(r['trials']) == len(r['input_preparation']) == len(r['diagnostic_records']) == 5",1),
        ("'All14 actual artifacts and two complete acquisition records'","'All35 actual artifacts and five complete acquisition records'",1),
        ("t['call'] == call and t['rank'] == 37","t['call'] == call and t['rank'] == (*RANKS,37)[call]",1),
        ("p['call'] == call and p['rank'] == 37","p['call'] == call and p['rank'] == (*RANKS,37)[call]",1),
        ("p['weight_file_sha256'] == WEIGHT",
         "p['weight_file_sha256'] == read(r['controls'][call]['input_bill'])['weight_file_sha256']",1),
        ("packet['rank'] == 37","packet['rank'] == (*RANKS,37)[call]",1),
        ("m['rank'] == 37","m['rank'] == (*RANKS,37)[call]",1),
        ("dict(call=call,rank=37,path=str(mp),sha256=sha(mp))","dict(call=call,rank=(*RANKS,37)[call],path=str(mp),sha256=sha(mp))",1),
        ("domains[0] == domains[1]","domains[0] == domains[-1]",1),
        ("r['trials'][0]['flops']-r['trials'][1]['flops']","r['trials'][0]['flops']-r['trials'][-1]['flops']",1),
        ("HERE/'score_probe.py'","HERE/'score_check4.py'",1),
        ("HERE/'watch_score_probe.py'","HERE/'watch_score_check4.py'",1),
        ("HERE/'SCORE_PROBE_PROTOCOL.md'","HERE/'SCORE_CHECK4_PROTOCOL.md'",1),
        ("    env,proof = helpers(admit); result['helper_source_proof'] = proof\n",
         "    env,proof = helpers(admit); result['helper_source_proof'] = proof\n"
         "    prior,prerequisite = _check4_prerequisite(admit)\n"
         "    result['development_prerequisite'] = prerequisite\n"
         "    result['score_adapter_proof'] = _CHECK4_PROOF\n",1),
        ("[base['trials'][i] for i in (0,40)]","[base['trials'][i] for i in (0,1,2,3,40)]",1),
        ("trials[1]['output_sha256']","trials[-1]['output_sha256']",1),
        ("trials[1]['output_bytes_sha256']","trials[-1]['output_bytes_sha256']",1),
        ("Path(trials[1]['output']).read_bytes()","Path(trials[-1]['output']).read_bytes()",1),
        ("[env['trial_output'](np,APUSH/'native40_terminal_v1',r['controls'][0],admit.pins)]",
         "[env['trial_output'](np,APUSH/'native40_terminal_v1',t,admit.pins) for t in r['controls'][:4]]",1),
        ("[env['trial_output'](np,HERE/'check4_v1',r['trials'][0],admit.pins)]",
         "[env['trial_output'](np,HERE/'check4_v1',t,admit.pins) for t in r['trials'][:4]]",1),
        ("arrays['shared'][0][:5].tobytes() == arrays['APUSH'][0][:5].tobytes()",
         "all(arrays['shared'][i][:5].tobytes() == arrays['APUSH'][i][:5].tobytes() for i in range(4))",1),
        ("sum(v[0].nbytes for v in arrays.values()) < 200_000",
         "sum(x.nbytes for v in arrays.values() for x in v) < 1_000_000",1),
        ("'Two distinct saved input arrays below0.2MB'","'Eight distinct saved arrays below1MB'",1),
        ('ranks=[37]','ranks=list(RANKS)',2),
        ('all_four_outputs_verified','all_ten_outputs_verified',2),
        (old_target,new_target,1),
        ("    require(sha(TRUTH,tick,truth=True) == TRUTH_SHA, 'Fixed truth unchanged after scoring')",
         "    require(all(sha(p,tick,truth=True) == h for p,h in TRUTHS.items()), 'All four original truths unchanged')",1),
        ('record=row,comparison=comparison','records=records,summary=summary,comparison=comparison',1),
        ("'One cold rank37 final raw*max(.1,own completeF/2^41); repeat excluded'",
         "'Mean of four per-net final raw*max(.1,own completeF/2^41); repeat excluded'",1),
        ("HERE/'score_probe_v1'","HERE/'score_check4_v1'",1),
        ("'information_promising','record','comparison','error'","'information_promising','summary','comparison','error'",1),
    ]
    changed = _adapt(original,edits)
    return changed,dict(parent_sha256=_BASE_SHA,whole_source_inverse=True,
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),ranks=list(_RANKS),
        unchanged_per_network_FP64_arithmetic=True,mean_of_products_not_product_of_means=True,
        raw_tuning=False,prior_rank37_exact_replay=True,prospective_mean_ratio=.97,
        minimum_adjusted_wins=3,maximum_individual_ratio=1.10)


_derived,_CHECK4_PROOF = derive_source()
exec(compile(_derived,str(_BASE)+':fixed-check4-inverse','exec'),globals())
