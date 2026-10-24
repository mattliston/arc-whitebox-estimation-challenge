"""Fixed CHECK4 metadata adapter; the qualified native paid loop is unchanged."""
import ast
import hashlib
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_OWNER = _HERE/'probe.py'
_OWNER_SHA = '6193396f83150e7e95f3c16ad5095eeb8d4c62da17d69d5bd7dcb08fad9a743b'
_original = _OWNER.read_text()
if hashlib.sha256(_original.encode()).hexdigest() != _OWNER_SHA:
    raise RuntimeError('Exact successful two-call native source owner')

RANKS = (37,39,42,48)
SEQUENCE = RANKS+(37,)
COMPOSED_SHA = '4d98fef2c11c571efe1eca945018bb9fe9ece1454d16b4abde2eb52d71e755c3'
EXTRA_SOURCES = {
    _OWNER:_OWNER_SHA,
    _HERE/'watch_probe.py':'7ade1c9025a2932e7929498062f894b72a3817d50027e7c5d1cfbf22a1f0fd18',
    _HERE/'PROBE_PROTOCOL.md':'43e0c94e729a7004d2d51fb6424d4122e7d6f168a1f241095ff33abac2ad04c2',
    _HERE/'PROBE_WATCH_SPEC.md':'743fc63fcb7ff5a7517fcb4e4311ee586fbd8ec72ed36004e2883810298cdc3f',
    _HERE/'score_probe.py':'a279e9bf196c50ef3157ade817946f5f6a5f75521040f0ce03820649a8c1e7cb',
    _HERE/'watch_score_probe.py':'ce125eeb2bf774f9647b925018ba3ea1bf4e7b0a4e18277d6fab2d1c09d7d6a5',
    _HERE/'SCORE_PROBE_PROTOCOL.md':'8b2aa405fe83056eb018ddd4f0eaad01e84f5457d4d2144e4d6e9a672fddf570',
    _HERE/'EXPANSION_DECISION_20260912.md':'f8b37def60271d1a1bf17dd9008a076ae144cd3058ebb8e13c1c8799e012cadc',
}
PREREQUISITES = {
    'probe_v1':('016e81e7bd0750f0e99671ac4def7deb02f591a2a292d07185eee25b7e2c591f',
        'd0fffc724a84d330b7db629cb59cb5a321fd36e91f787b533688746c855aba02',
        '0b0e98f17218eb23260f95a8fba4dddfca2b706ae4aa328cc437641c29c29d51'),
    'probe_watch_v1':('735a6efcd8206dd77a9eaa3a940d79cc321232a2993cb25643dbf079b9be7b42',
        '95754bb3d25dfb2088f8812d7ce2ebe418b0510a241f774de6132d93f6630d9a',
        '56fe2953128820874809e4a11f2798a3ac13be669d8cbdd1558c81f7498c7368'),
    'score_probe_v1':('d00066a38fc9c6012282c80a3a3d7f9a97f420b137f67e1daae5473589b88dd0',
        'c5f5fe56d376558154ee0aa16cc47789ec8054768f5570a2409e1848a32a69ce',
        '10c6ccc33527499885662a1c8c2638a4839d7995047c5174a38fade253adf79d'),
    'score_probe_watch_v1':('ea4a6847a04d82b159d3dbcea51d801e80b84628f1d11eebc22be44322de29f5',
        '7958912d3cb50a1de35840d2387df545b590dcd1b1280ed459cef24fff8fc477',
        '701b1bdc06927cab4242bd05f51680c36c4db5eb069cb5bd9eb99f241b5b00c1'),
}
ROWS = {
    37:('900e3cc6929bcc81ea301479708fe8bdfb642447f6c8bb1a10981bffacc15571',
        '967d989cf0ddf4770f199d50c97b26c70b1129d2d739929e82c8e229b7fd43d4'),
    39:('500b56411928de133ee4266eb2378ed6323516981241d45c9639a9a314782f58',
        '17cc551bc05f0cbd81209c79d7527e2f89ed9ad7058a035f88feb03a6250f3cf'),
    42:('2ad08923a0e02ff227c64b6c4368e5bd73381686437f524b36ec8453872375ca',
        'a45c24f5f9b1520adfcd8845263a35afc4dca5803bafb9156dbc02dbfac7487a'),
    48:('5b2a928210cc6db6b9f64dcf52c179bbec0fe2b5c348f53820a78f0d8676e40e',
        '1b1417c2872f8823d466783ba8e1d7d2a12fb30fc4cfe92df473eca73d526842'),
}


def prerequisites(inputs, tick=None):
    """Direct actual positive boundaries only; never import/run their numerical code."""
    union = {str(p):h for p,h in EXTRA_SOURCES.items()}
    records = {}
    def merge(mapping):
        for p,h in mapping.items():
            require(p not in union or union[p] == h, 'Consistent direct prerequisite inventory')
            union[p] = h
    for folder,expected in PREREQUISITES.items():
        paths = [HERE/folder/n for n in ('result.json','completion.json','finalization.json')]
        digests = [sha(p,tick) for p in paths]
        require(tuple(digests) == expected, 'Pinned complete successful prerequisite '+folder)
        r,e,f = map(read,paths); limit = 300 if folder.startswith('probe') else 120
        require(all(x.get('passed') is True and x.get('completed') is True for x in (r,e,f))
            and e['result_sha256'] == f['result_sha256'] == expected[0] and f['completion_sha256'] == expected[1]
            and r['hashes_unchanged'] and r['sources_unchanged'] and 0 <= r['elapsed_s'] < limit
            and 0 <= f['elapsed_s'] < limit, 'Actual positive three-seal eligibility')
        merge({str(p):h for p,h in zip(paths,digests)})
        # No truth_bindings traversal, target hashing, scorer invocation or older admission.
        for key in ('source_bindings','input_bindings','artifacts'):
            require(isinstance(r[key],dict) and r[key], 'Complete direct prerequisite inventory')
            merge(r[key])
        if 'watch' in folder:
            require(all(r[k] is True for k in ('whole_owned_tree_retired','cleanup_passed',
                'observational_groups_retired','actual_completed_samples','foreign_numerical_workers_checked'))
                and r['numerical_imports'] is False and r['maximum_swap_growth_bytes'] == 0
                and 0 < r['maximum_interval_s'] <= 1
                and 0 < r['maximum_combined_rss_bytes'] < (8_000_000_000 if folder == 'probe_watch_v1' else 1_000_000_000),
                'Original passed resource/retirement scope, no authority expansion')
        else:
            require(0 < r['peak_rss_bytes'] < (8_000_000_000 if folder == 'probe_v1' else 1_000_000_000),
                'Actual original worker peak, alongside independent combined-tree authority')
        records[folder] = r
    for worker,watch in (('probe_v1','probe_watch_v1'),('score_probe_v1','score_probe_watch_v1')):
        require(tuple(records[watch]['child_'+k] for k in ('result_sha256','completion_sha256','finalization_sha256'))
            == PREREQUISITES[worker], 'Exact separately watched worker seals')
    p,s = records['probe_v1'],records['score_probe_v1']
    require(p['candidate_sha256'] == COMPOSED_SHA and p['full_model_calls'] == p['retired_call_count'] == 2
        and p['ranks'] == [37,37] and p['repeat_byte_exact'] and p['all16_domains_qualified']
        and p['all47_selectors_qualified'] and p['own_complete_bills'] and p['truth_accessed'] is False
        and p['score_measured'] is False and p['initialization_flops'] == 1944970298,
        'Exact successful fixed candidate and native37/repeat37')
    require(s['information_promising'] is True and s['scientific_gate_passed'] is True
        and s['comparison']['adjusted_ratio'] <= .95 and s['exact_APUSH_score_replay']
        and s['complete_probe_acquisition_reverified'] and s['all_predictions_verified_before_truth']
        and s['full_model_calls'] == s['parameter_updates'] == 0 and s['ranks'] == [37] and s['repeat_excluded']
        and records['score_probe_watch_v1']['information_promising'] is True
        and tuple(s['lineage']['probe'][k] for k in ('result_sha256','completion_sha256','finalization_sha256'))
            == PREREQUISITES['probe_v1'], 'Fixed prospective .95 gate passed; no score or truth rerun')
    bind(union,tick)
    for path,digest in union.items():
        require(path not in inputs or inputs[path] == digest, 'Consistent new input authority')
        inputs[path] = digest
    return p,{name:dict(zip(('result_sha256','completion_sha256','finalization_sha256'),hashes))
        for name,hashes in PREREQUISITES.items()}


def probe37_check(np, result, prior, call):
    """Only cold37 and final repeat37 map to the original two-call subject."""
    old_call = 0 if call == 0 else 1
    a,b = result['trials'][call],prior['trials'][old_call]
    require(a['rank'] == b['rank'] == 37 and call in (0,4), 'Exact old/new37 mapping')
    keys = ('flops','operation_count','multiplier','includes_initialization')
    require(all(a[k] == b[k] for k in keys), 'Own deterministic complete bills reproduce the positive probe')
    require(a['output_sha256'] == b['output_sha256'] and a['output_bytes_sha256'] == b['output_bytes_sha256']
        and Path(a['output']).read_bytes() == Path(b['output']).read_bytes(), 'Exact cold/repeat37 saved bytes')
    da = read(result['diagnostic_records'][call]['path']); db = read(prior['diagnostic_records'][old_call]['path'])
    require(da['domains'] == db['domains'] and da['dispatches'] == db['dispatches'],
        'Exact all16/all47 and actual-shape dispatch packets on37 only')
    require(read(a['input_bill'])['flops'] == read(b['input_bill'])['flops'] == 16777216,
        'Own separate canonical input bill unchanged')
    return dict(call=call,probe_call=old_call,rank=37,output_byte_exact=True,
        complete_bills_exact=True,domains_exact=True,dispatches_exact=True)


_changes = [
    ("BUDGET, CAP, RSS = 2**41, 300., 8_000_000_000", "BUDGET, CAP, RSS = 2**41, 660., 8_000_000_000"),
    ("HERE/'probe.py'", "HERE/'check4.py'"),
    ("HERE/'watch_probe.py'", "HERE/'watch_check4.py'"),
    ("HERE/'PROBE_PROTOCOL.md'", "HERE/'CHECK4_PROTOCOL.md'"),
    ("HERE/'PROBE_WATCH_SPEC.md'", "HERE/'CHECK4_WATCH_SPEC.md'"),
    ("LEGACY:LEGACY_SHA,OBSERVER:OBSERVER_SHA}", "LEGACY:LEGACY_SHA,OBSERVER:OBSERVER_SHA,**EXTRA_SOURCES}"),
    ("for index, expected in ((0,461231998786),(40,459287028488)):",
     "for index, rank, expected in ((0,37,461231998786),(1,39,482950877279),(2,42,445415645868),\n"
     "            (3,48,456127497800),(40,37,459287028488)):"),
    ("require(trial['rank'] == 37 and trial['call'] == index", "require(trial['rank'] == rank and trial['call'] == index"),
    ("FIXTURE[ROW/'weights.npy']", "FIXTURE[STORE/f'rows/observer_fit/{rank:04d}/weights.npy']"),
    ("args.out = HERE/'probe_v1'", "args.out = HERE/'check4_v1'"),
    ("schema='current_comp_shared_old384_native37_v1'", "schema='current_comp_shared_old384_native_check4_v1'"),
    ("retired_call_count=0,production_residual_qualified=False,parameter_updates=0)",
     "retired_call_count=0,production_residual_qualified=False,parameter_updates=0,\n"
     "        check4_source_inverse_proof=SOURCE_INVERSE_PROOF)"),
    ("Whole native stage300s/process8GB", "Whole native stage660s/process8GB"),
    ("        bind(result['source_bindings'],tick)\n        composition = load",
     "        bind(result['source_bindings'],tick)\n"
     "        prior_probe,result['prerequisite_seals'] = prerequisites(result['input_bindings'],tick)\n"
     "        result['probe_prerequisite_qualified'] = True\n        composition = load"),
    ("        result['candidate_sha256'] = hashlib.sha256(text.encode()).hexdigest()",
     "        result['candidate_sha256'] = hashlib.sha256(text.encode()).hexdigest()\n"
     "        require(result['candidate_sha256'] == COMPOSED_SHA, 'No candidate/parameter change after positive probe')"),
    ("        result['input_bindings'].update({str(p):v for p,v in FIXTURE.items()})",
     "        for rank,(row_sha,weight_sha) in ROWS.items():\n"
     "            FIXTURE[STORE/f'rows/observer_fit/{rank:04d}/row_manifest.json'] = row_sha\n"
     "            FIXTURE[STORE/f'rows/observer_fit/{rank:04d}/weights.npy'] = weight_sha\n"
     "        result['input_bindings'].update({str(p):v for p,v in FIXTURE.items()})"),
    ("        row = read(ROW/'row_manifest.json')\n        rows = {37:dict(weights=str(ROW/'weights.npy'),mlp_seed=int(row['mlp_seed']))}\n"
     "        result['ranks'] = [37,37]",
     "        rows = {rank:dict(weights=str(STORE/f'rows/observer_fit/{rank:04d}/weights.npy'),\n"
     "            mlp_seed=int(read(STORE/f'rows/observer_fit/{rank:04d}/row_manifest.json')['mlp_seed'])) for rank in RANKS}\n"
     "        result['ranks'] = list(SEQUENCE)"),
    ("            if result['diagnostic_records']:", "            if call == 4:  # Only the final37 is a repeated network."),
    ("rows=rows,ranks=(37,),result=result", "rows=rows,ranks=RANKS,result=result"),
    ("require(len(result['trials']) == len(result['diagnostic_records']) == 2, 'Both complete fixed calls')",
     "require(len(result['trials']) == len(result['diagnostic_records']) == 5, 'All five complete fixed calls')"),
    ("        result['unaffected_prefix_checks'] = prefix_checks",
     "        result['unaffected_prefix_checks'] = prefix_checks\n"
     "        result['probe37_parity_checks'] = [probe37_check(np,result,prior_probe,call) for call in (0,4)]\n"
     "        result.update(probe37_output_byte_exact=True,probe37_complete_bills_exact=True,\n"
     "            probe37_domains_exact=True,probe37_dispatches_exact=True)"),
    ("result['trials'][0]['flops']-result['trials'][1]['flops']", "result['trials'][0]['flops']-result['trials'][4]['flops']"),
    ("require(result['initialization_flops'] >= 1944970298, 'Original startup is fully owned, no subtraction')",
     "require(result['initialization_flops'] == 1944970298, 'Exact original startup fully owned, no subtraction')"),
    ("result['full_model_calls'] = result['retired_call_count'] = 2", "result['full_model_calls'] = result['retired_call_count'] = 5"),
]
_derived = _original
for _before,_after in _changes:
    if _derived.count(_before) != 1:
        raise RuntimeError('Unique CHECK4 source adaptation: '+_before[:100])
    _derived = _derived.replace(_before,_after)
_inverse = _derived
for _before,_after in reversed(_changes):
    if _inverse.count(_after) != 1:
        raise RuntimeError('Unique CHECK4 source inverse: '+_after[:100])
    _inverse = _inverse.replace(_after,_before)
if _inverse != _original:
    raise RuntimeError('Whole successful probe source restored byte-exact')
ast.parse(_derived)
SOURCE_INVERSE_PROOF = dict(owner=str(_OWNER),owner_sha256=_OWNER_SHA,whole_source_inverse_verified=True,
    derived_sha256=hashlib.sha256(_derived.encode()).hexdigest(),paid_loop_function_unchanged=True,
    observer_functions_unchanged=True,roster_and_positive_boundary_metadata_only=True,splices=len(_changes))
exec(compile(_derived,str(_OWNER)+':fixed-CHECK4-metadata-adapter','exec'),globals())
