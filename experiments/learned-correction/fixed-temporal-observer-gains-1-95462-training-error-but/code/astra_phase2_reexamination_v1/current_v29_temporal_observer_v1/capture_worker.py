"""Inert, explicit-inverse TRAIN20 or fresh-process CHECK5 v29 capture.

No numerical package is imported by this module. Original qualification,
adapter, model/training design and failed historical families are unchanged.
"""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
BASE = HERE/'worker.py'
BASE_SHA = '551644da1dd6a0d7fe193df03894e889e8e6589bb839c65da64bf2917acb271a'
BASE_WATCH_SHA = 'f7783cd003fdca3bd625ffa62733fbb12655704998cb2547e00dcee736b8e0b6'
ADAPTER_SHA = 'cab844cd7981d31c7e44f4797ad60490729bdfadb8ce46655cdd5966ae9f36b7'
ADAPTER_PROTOCOL_SHA = '2a93907e69427a55c9a11eef7bd2143d9f71835d62f5eb4323700c1a8bd545ba'
OWNER_PINS = {
    BASE: BASE_SHA, HERE/'watch.py': BASE_WATCH_SHA,
    HERE/'ACQUISITION_PROTOCOL.md': '4ec87b4e7ce8e8a3859cb772ef16a131886912b45b577417414dce1e8261f7b5',
    HERE/'WATCH_SPEC.md': 'ae85478753619afcb62d82eb0d6e4d23290b3eafc8f4df34750f4176feda6619',
    HERE/'adapter.py': ADAPTER_SHA, HERE/'PROTOCOL.md': ADAPTER_PROTOCOL_SHA,
    HERE/'TRAIN_DESIGN.md': '4c9124877e509570e4d6fcfb2d487dc33154267610a0295c775c6f1e1f60332f',
    HERE/'ACQUISITION_EXPANSION_DESIGN.md': 'c1873dc6d5b763505b4a5b74c519c9e6352a27440c79fd3135ec500ac437aa85',
}
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
STORE_SHA = '9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50'
TRAIN = (35,36,38,40,41,43,44,45,46,47,49,50,51,52,53,54,55,56,57,58)
CHECK = (37,39,42,48,37)
CAPS = {'train':2460., 'check':660.}
RSS, BUDGET, FEATURE_LAYER_CEILING = 8_000_000_000, 2**41, 71_303_168
TIMINGS = ('wall_s','context_wall_s','flopscope_backend_s','flopscope_overhead_s','native_residual_s')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def valid_digest(value):
    return isinstance(value,str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def sha(path, tick=None):
    p = Path(path)
    require(not p.is_symlink(), 'No symlink metadata or array owner')
    p = p.resolve()
    require(p.is_relative_to(P2) and 'truthpost' not in p.parts and not
        (p.suffix in ('.npy','.npz') and 'truth' in p.name.lower()), 'Target-blind byte hashing only')
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''):
            h.update(block)
            if tick is not None:
                tick()
    return h.hexdigest()


def read(path):
    path = Path(path)
    require(not path.is_symlink() and path.stat().st_size < 16*1024**2, 'Bounded regular JSON metadata')
    return json.loads(path.read_text())


def add(mapping, path, digest, tick=None):
    path = str(Path(path).resolve())
    require(valid_digest(digest) and mapping.get(path,digest) == digest and sha(path,tick) == digest,
            'Exact immutable source/input/artifact '+path)
    mapping[path] = digest


def inventory(record, sources, inputs, tick=None):
    for group,destination in (('source_bindings',sources),('input_bindings',inputs),('artifacts',inputs)):
        values = record[group]
        require(isinstance(values,dict) and values, 'Complete target-blind '+group)
        for path,digest in values.items():
            add(destination,path,digest,tick)


def chain(folder, expected, inputs, cap, tick=None):
    paths = [folder/name for name in ('result.json','completion.json','finalization.json')]
    digests = [sha(p,tick) for p in paths]
    require(valid_digest(expected) and digests[0] == expected, 'Explicit actual prerequisite result')
    record,end,final = [read(p) for p in paths]
    require(all(x.get('passed') is True and x.get('completed') is True for x in (record,end,final))
        and end['result_sha256'] == final['result_sha256'] == digests[0]
        and final['completion_sha256'] == digests[1]
        and 0 <= record['elapsed_s'] <= final['elapsed_s'] < cap, 'Actual positive bounded three-seal chain')
    for path,digest in zip(paths,digests):
        add(inputs,path,digest,tick)
    return record,dict(zip(('result_sha256','completion_sha256','finalization_sha256'),digests))


def configuration(args):
    require(args.stage in CAPS and valid_digest(args.qualification_sha256)
        and valid_digest(args.qualification_watch_sha256), 'Stage and actual two-call prerequisite required')
    if args.stage == 'check':
        require(valid_digest(args.train_sha256) and valid_digest(args.train_watch_sha256),
                'Fresh CHECK requires the actual completed TRAIN worker and watch')
    else:
        require(args.train_sha256 is None and args.train_watch_sha256 is None,
                'TRAIN has no prior TRAIN receipt or resume option')
    return CAPS[args.stage], TRAIN if args.stage == 'train' else CHECK


def new_owners(args):
    require(args.adapter_sha256 == ADAPTER_SHA and args.adapter_protocol_sha256 == ADAPTER_PROTOCOL_SHA,
            'Frozen adapter and feature protocol')
    values = {HERE/'capture_worker.py':args.source_sha256,HERE/'capture_watch.py':args.watch_sha256,
        HERE/'CAPTURE_PROTOCOL.md':args.protocol_sha256,HERE/'CAPTURE_WATCH_SPEC.md':args.watch_spec_sha256,
        **OWNER_PINS}
    require(all(valid_digest(v) for v in values.values()), 'Explicit reviewed expanded source owners')
    return {str(p):v for p,v in values.items()}


def environment():
    """The exact original OFFPY/package/environment admission, still stdlib only."""
    require(sha(BASE) == BASE_SHA, 'Frozen environment provider')
    spec = importlib.util.spec_from_file_location('v29_capture_environment_owner',BASE)
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    owner.environment()


def worker_args(args):
    """Normalize the supervisor's source/worker/spec roles, without imports."""
    return types.SimpleNamespace(stage=args.stage,source_sha256=args.worker_sha256,
        watch_sha256=args.source_sha256,protocol_sha256=args.protocol_sha256,
        watch_spec_sha256=args.spec_sha256,adapter_sha256=args.adapter_sha256,
        adapter_protocol_sha256=args.adapter_protocol_sha256,
        qualification_sha256=args.qualification_sha256,
        qualification_watch_sha256=args.qualification_watch_sha256,
        train_sha256=args.train_sha256,train_watch_sha256=args.train_watch_sha256)


def fixture_rows(sequence, inputs, tick=None):
    add(inputs,STORE/'manifest.json',STORE_SHA,tick)
    manifest = read(STORE/'manifest.json')['roles']['observer_fit']['row_manifests']
    rows, fixtures = {},{}
    for rank in dict.fromkeys(sequence):
        matches = [x for x in manifest if x['role_rank'] == rank]
        require(len(matches) == 1, 'Unique canonical observer_fit rank')
        item = matches[0]
        path = STORE/item['path']
        require(path == STORE/f'rows/observer_fit/{rank:04d}/row_manifest.json', 'Canonical row path')
        add(inputs,path,item['sha256'],tick)
        row = read(path); wp = path.parent/'weights.npy'; data = row['files']['weights.npy']
        require(data['dtype'] == '<f4' and data['shape'] == [16,1024,1024]
            and data['bytes'] == 67_108_992 and valid_digest(data['array_sha256'])
            and isinstance(row['mlp_seed'],int), 'Exact existing FP32 weight metadata')
        add(inputs,wp,data['sha256'],tick)
        rows[rank] = dict(weights=str(wp),mlp_seed=row['mlp_seed'])
        fixtures[str(rank)] = dict(rank=rank,row_manifest=str(path),row_manifest_sha256=item['sha256'],
            path=str(wp),sha256=data['sha256'],raw_sha256=data['array_sha256'],dtype=data['dtype'],
            shape=data['shape'],bytes=data['bytes'],mlp_seed=row['mlp_seed'])
    return rows,fixtures


def watcher_contract(watch, worker, seals, sequence, stage=None):
    cap = 300. if stage is None else CAPS[stage]
    schema = 'current_v29_temporal_acquisition_watch_v1' if stage is None else 'current_v29_temporal_capture_watch_v1'
    require(watch['schema'] == schema and (stage is None or watch['stage'] == stage)
        and watch['ranks'] == list(sequence) and watch['retired_calls'] == len(sequence)
        and [(x['call'],x['rank']) for x in watch['markers']] == list(enumerate(sequence))
        and all(watch.get(k) is True for k in ('passed','completed','hashes_unchanged','sources_unchanged',
            'child_execution_completed','whole_owned_tree_retired','cleanup_passed',
            'observational_groups_retired','actual_completed_samples','foreign_numerical_workers_checked'))
        and [watch['child_'+k] for k in seals] == list(seals.values())
        and watch['candidate_sha256'] == worker['candidate_sha256']
        and watch['composition_proof'] == worker['composition_proof']
        and watch['whole_cap_s'] == cap and watch['progress_cap_s'] == 120
        and watch['combined_rss_limit_bytes'] == RSS
        and 0 < watch['maximum_combined_rss_bytes'] < RSS
        and 0 < watch['maximum_interval_s'] <= 1 and watch['maximum_swap_growth_bytes'] == 0
        and watch['observation_samples'] == len(watch['completed_sample_intervals_s']) > 0
        and all(0 < x <= 1 for x in watch['completed_sample_intervals_s'])
        and watch['numerical_imports'] is False and watch['truth_accessed'] is False
        and watch['score_measured'] is False, 'Actual retired, bounded, target-blind supervisor')
    if stage is not None:
        require(watch['child_pid'] == worker['worker_pid'] and watch['expanded_capture_verified'] is True
            and watch['prerequisite_seals'] == worker['prerequisite_seals']
            and watch['fixture_bindings'] == worker['fixture_bindings'], 'Fresh process and complete stage lineage')


def calls_contract(record, folder, sequence, expanded=False, tick=None):
    """Stdlib metadata verification; captured arrays are hashed, never decoded."""
    n = len(sequence)
    require(record['ranks'] == list(sequence)
        and record['full_model_calls'] == record['retired_call_count'] == n
        and len(record['trials']) == len(record['capture_records']) == len(record['input_preparation']) == n
        and record['setup_calls'] == 1 and record['setup_seed'] == record['setup']['seed'] == 0
        and record['setup']['inside_cold_participant_bill'] and record['setup']['constructor_included']
        and 0 <= record['setup']['wall_s'] < 5 and 0 <= record['setup']['flops'] <= record['trials'][0]['flops']
        and record['parameter_updates'] == 0 and record['truth_accessed'] is False
        and record['score_measured'] is False and record['head_attached'] is False
        and record['feedback_enabled'] is False, 'Fixed head-free complete native roster and owned setup')
    expected_artifacts = set()
    for call,rank in enumerate(sequence):
        trial,capture,prepared = record['trials'][call],record['capture_records'][call],record['input_preparation'][call]
        require(trial['call'] == capture['call'] == prepared['call'] == call
            and trial['rank'] == capture['rank'] == prepared['rank'] == rank
            and trial['includes_initialization'] is (call == 0), 'Exact call/rank/cold identities')
        files = {
            'input_bill':folder/f'call_{call:02d}_input_bill.json',
            'input_reset':folder/f'call_{call:02d}_input_reset.json',
            'bill':folder/f'call_{call:02d}_bill.json',
            'lifecycle':folder/f'call_{call:02d}_lifecycle.json',
            'output':folder/f'call_{call:02d}_rank{rank}.npy',
            'capture':folder/f'call_{call:02d}_capture.npz',
            'diagnostics':folder/f'call_{call:02d}_diagnostics.json',
            'complete':folder/f'call_{call:02d}_complete.json',
        }
        for path in files.values():
            expected_artifacts.add(str(path))
            require(sha(path,tick) == record['artifacts'][str(path)], 'Actual per-call eight-artifact inventory')
        require(read(files['bill']) == trial and read(files['input_bill']) == prepared
            and read(files['diagnostics']) == capture
            and trial['output'] == str(files['output']) and trial['input_bill'] == str(files['input_bill'])
            and trial['input_reset_receipt'] == str(files['input_reset'])
            and trial['lifecycle_receipt'] == str(files['lifecycle'])
            and capture['path'] == str(files['capture'])
            and trial['output_sha256'] == record['artifacts'][str(files['output'])]
            and capture['sha256'] == record['artifacts'][str(files['capture'])]
            and trial['input_bill_sha256'] == record['artifacts'][str(files['input_bill'])]
            and trial['input_reset_sha256'] == record['artifacts'][str(files['input_reset'])]
            and valid_digest(trial['output_bytes_sha256']), 'Saved exact own outputs/bills/capture metadata')
        for metrics in (trial,prepared):
            require(0 < metrics['flops'] <= BUDGET and metrics['wall_s'] < 120
                and 0 < metrics['peak_rss_bytes'] < RSS and metrics['operation_count'] > 0
                and all(isinstance(metrics[k],(int,float)) and math.isfinite(metrics[k]) and metrics[k] >= 0
                        for k in TIMINGS), 'Unchanged original per-context resource gates')
        require(trial['multiplier'] == max(.1,trial['flops']/BUDGET)
            and prepared['flops'] == 16_777_216 and prepared['charged_to_participant'] is False
            and all(prepared[k] is True for k in ('exact_bytes','official_fnp_type',
                'independent_weight_buffers','context_closed')), 'Separate original paid input convention')
        for label,bill in (('input_reset','input_bill'),('lifecycle','bill')):
            life = read(files[label])
            own = prepared if label == 'input_reset' else trial
            require(life['call'] == call and life['rank'] == rank and life['saved_bill'] == str(files[bill])
                and life['saved_bill_sha256'] == record['artifacts'][str(files[bill])]
                and life['reset_api'] == 'flopscope.budget_reset' and life['candidate_state_reset'] is False
                and all(life[k] is True for k in ('after_context_exit','context_released','complete',
                    'closed_context_bill_unchanged_by_reset'))
                and life['closed_metrics']['flops'] == own['flops']
                and life['closed_metrics']['operation_count'] == own['operation_count'], 'Actual public reset/release')
        complete = read(files['complete'])
        require(complete['call'] == call and complete['rank'] == rank
            and all(complete[k] is True for k in ('passed','completed','context_retired'))
            and complete['output_sha256'] == trial['output_sha256']
            and complete['capture_sha256'] == capture['sha256'], 'Actual retired complete marker')
        for label in ('bill','lifecycle','diagnostics'):
            key = 'diagnostic_sha256' if label == 'diagnostics' else label+'_sha256'
            require(complete[key] == record['artifacts'][str(files[label])], 'Retired marker joins actual records')
        require(len(capture['arrays']) == 64 and len(capture['domains']) == 16
            and len(capture['layer_feature_flops']) == len(capture['layer_feature_operations']) == 16
            and capture['feature_flops'] == sum(capture['layer_feature_flops'])
            and capture['feature_operations'] == sum(capture['layer_feature_operations'])
            and 0 < capture['feature_flops'] < trial['flops']
            and all(0 < f <= FEATURE_LAYER_CEILING for f in capture['layer_feature_flops'])
            and all(capture[k] is True for k in ('feature_F_in_actual_bill','host_served_byte_exact','all16_raw_domains'))
            and capture['head_attached'] is False and capture['feedback_enabled'] is False,
            'Complete actual paid feature totals and raw domains')
        for layer in range(16):
            for key in ('features','host','sigma','prevariance'):
                item = capture['arrays'][f'layer_{layer:02d}_{key}']
                require(item['dtype'] == 'float32' and item['shape'] == ([1024,30] if key == 'features' else [1024])
                    and valid_digest(item['raw_sha256']), 'Exact64 FP32 array metadata')
            domain = capture['domains'][layer]
            require(domain['layer'] == layer and domain['finite'] is True
                and math.isfinite(domain['minimum']) and domain['minimum'] >= 1e-10
                and domain['below_floor_count'] == 0
                and domain['raw_sha256'] == capture['arrays'][f'layer_{layer:02d}_prevariance']['raw_sha256'],
                'Actual unfloored physical domain binding')
        if expanded:
            fixture = record['fixture_bindings'][str(rank)]
            require(prepared['weight_file_sha256'] == fixture['sha256'], 'Paid copies own the declared canonical input')
    if expanded:
        expected_artifacts.update(str(folder/name) for name in ('started.json','capture_roster.json'))
        for name in ('started.json','capture_roster.json'):
            require(sha(folder/name,tick) == record['artifacts'][str(folder/name)], 'Actual two stage metadata artifacts')
    require(set(record['artifacts']) == expected_artifacts, 'Exact call artifacts plus declared two stage metadata files')
    require(record['feature_flops'] == [x['feature_flops'] for x in record['capture_records']], 'Actual feature totals, never subtracted')


def repeat_contract(record, stage):
    applies = stage != 'train'
    if stage is not None:
        require(record['repeat_applicable'] is applies
            and record['repeat_index'] == (4 if applies else None), 'Only CHECK call4 is a repeat')
    if not applies:
        require(record['repeat_byte_exact'] is None and record['feature_repeat_byte_exact'] is None
            and record['cold_minus_repeat_flops'] is None, 'No TRAIN repeat or cold-minus-last claim')
        return
    first,last = record['capture_records'][0],record['capture_records'][-1]
    require(record['repeat_byte_exact'] is True and record['feature_repeat_byte_exact'] is True
        and record['trials'][0]['output_bytes_sha256'] == record['trials'][-1]['output_bytes_sha256']
        and all(first[k] == last[k] for k in ('arrays','domains','layer_feature_flops','layer_feature_operations'))
        and record['cold_minus_repeat_flops'] == record['trials'][0]['flops']-record['trials'][-1]['flops'],
        'Own exact repeat bytes/features/domains/tariffs and observed cost difference')


def qualification(args, sources, inputs, tick=None):
    folder,watched = HERE/'acquisition_v1',HERE/'acquisition_watch_v1'
    q,qs = chain(folder,args.qualification_sha256,inputs,300.,tick)
    w,ws = chain(watched,args.qualification_watch_sha256,inputs,300.,tick)
    require(q['schema'] == 'current_v29_temporal_acquisition_v1'
        and all(q.get(k) is True for k in ('hashes_unchanged','sources_unchanged','all16_raw_domains_qualified',
            'all16_features_qualified','own_complete_bills','estimator_retired','setup_owned'))
        and q['feature_columns'] == 30 and q['production_residual_qualified'] is False,
        'Original actual execution/capture qualification, not accuracy or fit permission')
    for path,digest in OWNER_PINS.items():
        if path.name not in ('TRAIN_DESIGN.md','ACQUISITION_EXPANSION_DESIGN.md'):
            require(q['source_bindings'].get(str(path)) == digest, 'Exact original qualification owner')
    watcher_contract(w,q,qs,(37,37))
    inventory(q,sources,inputs,tick); inventory(w,sources,inputs,tick)
    calls_contract(q,folder,(37,37),tick=tick); repeat_contract(q,None)
    return q,dict(qualification=qs,qualification_watch=ws)


def prerequisites(args, sources, inputs, tick=None):
    configuration(args)
    q,seals = qualification(args,sources,inputs,tick)
    train = None
    if args.stage == 'check':
        train,ts = chain(HERE/'train_capture_v1',args.train_sha256,inputs,CAPS['train'],tick)
        watched,tws = chain(HERE/'train_capture_watch_v1',args.train_watch_sha256,inputs,CAPS['train'],tick)
        require(train['prerequisite_seals'] == seals, 'Same original qualification admitted by TRAIN and CHECK')
        verify_expanded(train,'train',args,q,tick)
        watcher_contract(watched,train,ts,TRAIN,'train')
        inventory(train,sources,inputs,tick); inventory(watched,sources,inputs,tick)
        require(train['worker_pid'] != os.getpid(), 'CHECK runs in a distinct fresh process')
        seals.update(train=ts,train_watch=tws)
    return dict(qualification=q,train=train,seals=seals)


def cold_replay(record, qualified):
    old,new = qualified['capture_records'][0],record['capture_records'][0]
    qt,nt = qualified['trials'][0],record['trials'][0]
    require(qt['rank'] == nt['rank'] == 37 and qt['includes_initialization'] and nt['includes_initialization']
        and qt['output_bytes_sha256'] == nt['output_bytes_sha256']
        and qt['output_sha256'] == nt['output_sha256'] and qt['flops'] == nt['flops']
        and all(old[k] == new[k] for k in ('arrays','domains','layer_feature_flops','layer_feature_operations'))
        and record['input_preparation'][0]['weight_file_sha256'] == qualified['input_preparation'][0]['weight_file_sha256'],
        'Fresh CHECK cold37 exactly replays the qualified source/input/output/capture/cold complete F')
    return dict(qualified_call=0,current_call=0,rank=37,output_byte_exact=True,
        capture_raw_bytes_exact=True,raw_domains_exact=True,feature_tariffs_exact=True,
        cold_complete_flops_exact=True,cold_operation_count_delta=nt['operation_count']-qt['operation_count'],
        own_repeat_vs_original_repeat_flops_delta=record['trials'][-1]['flops']-qualified['trials'][-1]['flops'],
        own_repeat_vs_original_repeat_operation_count_delta=record['trials'][-1]['operation_count']-qualified['trials'][-1]['operation_count'],
        warm_history_operation_counts_reported_not_equated=True)


def verify_expanded(record, stage, args, qualified, tick=None):
    sequence = TRAIN if stage == 'train' else CHECK
    require(record['schema'] == 'current_v29_temporal_capture_v1' and record['stage'] == stage
        and all(record.get(k) is True for k in ('passed','completed','hashes_unchanged','sources_unchanged',
            'all16_raw_domains_qualified','all16_features_qualified','own_complete_bills','estimator_retired',
            'setup_owned','original_two_call_qualification_passed','fixed_sequence_completed','capture_lineage_complete'))
        and record['feature_columns'] == 30 and record['candidate_sha256'] == qualified['candidate_sha256']
        and record['composition_proof'] == qualified['composition_proof']
        and record['whole_cap_s'] == CAPS[stage] and 0 <= record['elapsed_s'] < CAPS[stage]
        and 0 < record['peak_rss_bytes'] < RSS, 'Exact completed stage, source/configuration and resource scope')
    for path,digest in new_owners(args).items():
        require(record['source_bindings'].get(path) == digest, 'Same reviewed expanded source owner')
    source,proof = derive_source()
    require(record['capture_source_proof'] == proof, 'Exact whole worker source inverse')
    del source
    paid = record['paid_loop_proof']
    require(paid['whole_stage_loop_inverse'] and paid['stage'] == stage and paid['ranks'] == list(sequence)
        and paid['appended_repeat_removed'] and paid['all_other_paid_loop_bytes_unchanged']
        and paid['repeat_index'] == (4 if stage == 'check' else None)
        and paid['parent']['setup_splice_inverse'] and valid_digest(paid['derived_sha256']),
        'Actual stage paid-loop source proof and unchanged owned setup')
    calls_contract(record,HERE/f'{stage}_capture_v1',sequence,expanded=True,tick=tick)
    repeat_contract(record,stage)
    roster = read(HERE/f'{stage}_capture_v1/capture_roster.json')
    require(roster == dict(schema='current_v29_temporal_capture_roster_v1',stage=stage,ranks=list(sequence),
        call_count=len(sequence),cold_rank=sequence[0],repeat_index=4 if stage == 'check' else None,
        setup_seed=0,fixture_bindings=record['fixture_bindings'],prerequisite_seals=record['prerequisite_seals'],
        candidate_sha256=record['candidate_sha256'],capture_source_proof=proof), 'Exact pre-numerical fixed roster seal')
    require(record['call_artifact_count'] == 8*len(sequence)
        and record['stage_metadata_artifact_count'] == 2, '160/40 exact call-artifact declaration')
    if stage == 'check':
        require(record['train_prior_process_retired'] is True and record['cold_qualification_replay'] == cold_replay(record,qualified),
                'Fresh CHECK replay and TRAIN retirement lineage')
    else:
        require(record['cold_qualification_replay'] is None and record['train_prior_process_retired'] is None,
                'TRAIN has no borrowed cold37/repeat claims')


def complete_stage(result, args, admitted):
    """Only after all numerical objects/pools have passed the original weakref retirement."""
    sequence = TRAIN if args.stage == 'train' else CHECK
    result.update(repeat_applicable=args.stage == 'check',repeat_index=4 if args.stage == 'check' else None,
        cold_minus_repeat_flops=(result['trials'][0]['flops']-result['trials'][-1]['flops']) if args.stage == 'check' else None,
        feature_flops=[c['feature_flops'] for c in result['capture_records']],
        cold_qualification_replay=cold_replay(result,admitted['qualification']) if args.stage == 'check' else None,
        train_prior_process_retired=True if args.stage == 'check' else None,
        original_two_call_qualification_passed=True,fixed_sequence_completed=True,capture_lineage_complete=True,
        call_artifact_count=8*len(sequence),stage_metadata_artifact_count=2)
    calls_contract(result,args.out,sequence,expanded=True)


def rewrite(original, edits):
    changed = original
    for old,new in edits:
        require(changed.count(old) == 1, 'Unique expanded capture source edit '+old[:80])
        changed = changed.replace(old,new)
    inverse = changed
    for old,new in reversed(edits):
        require(inverse.count(new) == 1, 'Unique expanded capture source inverse '+new[:80])
        inverse = inverse.replace(new,old)
    require(inverse == original, 'Whole source byte inverse')
    return changed


def expanded_loop(source, stage, prior):
    sequence = TRAIN if stage == 'train' else CHECK
    edits = (
        ('for call,rank in enumerate(tuple(ranks)+(ranks[0],)):',
         'for call,rank in enumerate(tuple(ranks)):'),
        ("if call==len(ranks): assert values.tobytes()==first_bytes,'Repeat identity'",
         "if repeat_index is not None and call==repeat_index: assert values.tobytes()==first_bytes,'Repeat identity'"),
    )
    changed = rewrite(source,edits)
    return changed,dict(parent=prior,whole_stage_loop_inverse=True,stage=stage,ranks=list(sequence),
        appended_repeat_removed=True,repeat_index=4 if stage == 'check' else None,
        all_other_paid_loop_bytes_unchanged=True,derived_sha256=hashlib.sha256(changed.encode()).hexdigest())


def derive_source():
    """Inert original-worker inverse. No source is compiled or imported here."""
    original = BASE.read_text()
    require(hashlib.sha256(original.encode()).hexdigest() == BASE_SHA, 'Frozen two-call worker source')
    old_rows = """        row = owner.read(owner.ROW/'row_manifest.json')
        rows = {37:dict(weights=str(owner.ROW/'weights.npy'),mlp_seed=int(row['mlp_seed']))}
"""
    new_rows = """        require(proof == admitted['qualification']['composition_proof']
            and proof['derived_sha256'] == admitted['qualification']['candidate_sha256'], 'Qualified unchanged physical source')
        rows, result['fixture_bindings'] = _expansion.fixture_rows(SEQUENCE,result['input_bindings'],tick)
        roster = dict(schema='current_v29_temporal_capture_roster_v1',stage=args.stage,ranks=list(SEQUENCE),
            call_count=len(SEQUENCE),cold_rank=SEQUENCE[0],repeat_index=4 if args.stage == 'check' else None,
            setup_seed=0,fixture_bindings=result['fixture_bindings'],prerequisite_seals=result['prerequisite_seals'],
            candidate_sha256=result['candidate_sha256'],capture_source_proof=_expansion_proof)
        write(args.out/'capture_roster.json',roster)
        result['artifacts'][str(args.out/'capture_roster.json')] = sha(args.out/'capture_roster.json')
"""
    old_owner = """    values = {HERE/'worker.py':args.source_sha256, HERE/'watch.py':args.watch_sha256,
        HERE/'ACQUISITION_PROTOCOL.md':args.protocol_sha256, HERE/'WATCH_SPEC.md':args.watch_spec_sha256,
        **SOURCE_PINS}
"""
    new_owner = "    values = {**SOURCE_PINS, **_expansion.new_owners(args)}\n"
    old_loop_return = """    return changed, dict(parent=prior, setup_splice_inverse=True,
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        unchanged_owned_inputs_bills_materialization_resets_release_and_repeat=True)
"""
    new_loop_return = """    qualified = dict(parent=prior, setup_splice_inverse=True,
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        unchanged_owned_inputs_bills_materialization_resets_release_and_repeat=True)
    return _expansion.expanded_loop(changed,stage,qualified)
"""
    edits = (
        (old_owner,new_owner),
        ('def paid_loop_source(owner):','def paid_loop_source(owner, stage):'),
        (old_loop_return,new_loop_return),
        ("    parser.add_argument('--run',action='store_true')", "    parser.add_argument('--run',action='store_true')\n"
         "    parser.add_argument('--stage',choices=('train','check'))\n"
         "    for key in ('qualification','qualification-watch','train','train-watch'):\n"
         "        parser.add_argument('--'+key+'-sha256')"),
        ('    environment(); began = time.monotonic()',
         '    CAP, SEQUENCE = _expansion.configuration(args)\n    environment(); began = time.monotonic()'),
        ("    args.out = HERE/'acquisition_v1'", "    args.out = HERE/f'{args.stage}_capture_v1'"),
        ("schema='current_v29_temporal_acquisition_v1'", "schema='current_v29_temporal_capture_v1',stage=args.stage,\n        worker_pid=os.getpid(),whole_cap_s=CAP,capture_source_proof=_expansion_proof"),
        ("                'Whole300s/worker8GB; watcher owns full-tree resources')",
         "                'Prospective stage whole clock/worker8GB; watcher owns full-tree resources')"),
        ("        owner = load(LOOP_OWNER,'v29_native_loop_owner',LOOP_SHA)",
         "        admitted = _expansion.prerequisites(args,result['source_bindings'],result['input_bindings'],tick)\n"
         "        result['prerequisite_seals'] = admitted['seals']\n"
         "        owner = load(LOOP_OWNER,'v29_native_loop_owner',LOOP_SHA)"),
        ("        result['input_bindings'].update({str(p):h for p,h in owner.FIXTURE.items()})",
         "        require(tuple(adapter.TRAIN) == _expansion.TRAIN and tuple(adapter.CHECK)+(37,) == _expansion.CHECK, 'Frozen20/4 roster')"),
        ('raw_variance_sites=sites,ranks=[37,37])','raw_variance_sites=sites,ranks=list(SEQUENCE))'),
        ("loop, result['paid_loop_proof'] = paid_loop_source(owner)", "loop, result['paid_loop_proof'] = paid_loop_source(owner,args.stage)"),
        (old_rows,new_rows),
        ("        write(args.out/'started.json',result)\n        import numpy as np",
         "        write(args.out/'started.json',result)\n"
         "        result['artifacts'][str(args.out/'started.json')] = sha(args.out/'started.json')\n"
         "        require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules), 'All prerequisites before numerical imports')\n"
         "        import numpy as np"),
        ("            if result['capture_records']:", "            if args.stage == 'check' and call == 4:"),
        ('            ranks=(37,),result=result,module=module,estimator=None,first_bytes=None,observer=observer,',
         "            ranks=SEQUENCE,repeat_index=4 if args.stage == 'check' else None,\n"
         '            result=result,module=module,estimator=None,first_bytes=None,observer=observer,'),
        ("require(len(result['trials']) == len(result['capture_records']) == result['retired_call_count'] == 2,\n"
         "                'Exactly two completed fixed predictions')",
         "require(len(result['trials']) == len(result['capture_records']) == result['retired_call_count'] == len(SEQUENCE),\n"
         "                'Exactly the fixed stage sequence, with no appended TRAIN repeat')"),
        ("        result['cold_minus_repeat_flops'] = result['trials'][0]['flops']-result['trials'][1]['flops']\n"
         "        result['feature_flops'] = [c['feature_flops'] for c in result['capture_records']]",
         "        _expansion.complete_stage(result,args,admitted)"),
        ('            repeat_byte_exact=True,feature_repeat_byte_exact=True,own_complete_bills=True,',
         "            repeat_byte_exact=True if args.stage == 'check' else None,\n"
         "            feature_repeat_byte_exact=True if args.stage == 'check' else None,own_complete_bills=True,"),
    )
    changed = rewrite(original,edits)
    return changed,dict(owner_sha256=BASE_SHA,whole_source_inverse=True,edit_count=len(edits),
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        original_capture_physical_setup_and_numerical_body_unchanged=True,
        stage_roster_repeat_and_metadata_only=True)


if __name__ == '__main__':
    if '--run' not in sys.argv:
        raise SystemExit('Inert. Explicit --run --stage and reviewed source/prerequisite hashes are required.')
    source,proof = derive_source()
    scope = dict(__name__='__main__',__file__=str(Path(__file__).resolve()),
        _expansion=types.SimpleNamespace(**globals()),_expansion_proof=proof)
    exec(compile(source,str(Path(__file__).resolve())+':expanded-qualified-worker','exec'),scope)
