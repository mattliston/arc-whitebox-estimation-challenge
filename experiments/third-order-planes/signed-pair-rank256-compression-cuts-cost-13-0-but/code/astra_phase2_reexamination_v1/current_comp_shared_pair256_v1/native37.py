"""DRAFT: fixed pair256 native37/repeat37; source/metadata only at import."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import resource
import runpy
import struct
import textwrap
import time

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
GAIN = NEW/'current_comp_shared_gain_slice_v1'
SHARED = NEW/'current_comp_shared_old384_v1'
OWNER = GAIN/'native.py'
OWNER_SHA = '90017490bd1f1d5efabe1f32167789ecce948ee339b27c0ea0555f28d08b0587'
LEDGER = SHARED/'check4_v2.py'
LEDGER_SHA = '8d09d9b1f443e229b1641d277312b52c204392648b25445ffab63b82dba16058'
CANDIDATE_SHA = 'd7e9cba805c286bc7e1f81d36d05e4bc8cfb9b992324e61037d3aa6bee5cf6ba'
PARENT_SHA = 'c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea'
SCHEMA = 'current_comp_shared_pair256_native37_v1'
OUTPUT = HERE/'native37_v1'
SEAL_KEYS = ('result_sha256','completion_sha256','finalization_sha256')
CONTROL_NATIVE = ('95d80ab15860bd51865c828a48d09f195642fa6ac6d8f86f9c5e99b4f864d83e',
    'a4896323273ce08d9e8c18a8ee9b2e0a8ada6439bbfbb5d334c530f276eb513c',
    'a113ac65d0360f4021ccc89d10f9bc0165e0be8086fe95afb77ff23e44e56319')
CONTROL_WATCH = ('1bc1135f275609a4ea64bd7414de2cae9881f9396af3b561f6af6e1fe2bc7bfb',
    'ff445d8e23b4b010c03b4137685294ad0bd85b4484f9a2862096c610b229490c',
    'be9293ac03e21ac637bb1dc04af8fb4646639d5cd58c39688e027add935d9541')
KEYS = ('source','watch','protocol','watch_spec','builder','build_protocol','build',
        'build_watch','build_watch_source','build_watch_spec','candidate')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


require(hashlib.sha256(OWNER.read_bytes()).hexdigest() == OWNER_SHA,
        'Frozen qualified OpenBLAS paid-loop owner')
_parent = runpy.run_path(str(OWNER), run_name='_pair256_gain_native_metadata')
sha, read, bind = (_parent[k] for k in ('sha','read','bind'))
OFFPY = _parent['_api']['OFFPY']
STORE = _parent['_api']['STORE']
ROW = STORE/'rows/observer_fit/0037'
FIXTURE = {STORE/'manifest.json':'9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50',
    ROW/'row_manifest.json':'900e3cc6929bcc81ea301479708fe8bdfb642447f6c8bb1a10981bffacc15571',
    ROW/'weights.npy':'967d989cf0ddf4770f199d50c97b26c70b1129d2d739929e82c8e229b7fd43d4'}
PINS = {**_parent['PINS'], OWNER:OWNER_SHA, LEDGER:LEDGER_SHA,
    GAIN/'NATIVE_PROTOCOL.md':'01f8eecb0a42d68c4e36cef7f7029c03803c66283fba8f831d5c49d878848b62',
    GAIN/'native40_v2.py':'93012679ba77f392353a7311b25c3e770451e8a88ef7a6067505832b969629a6',
    GAIN/'watch_native40_v2.py':'a854b402e8112e848c3157c0c9fef59739302ac4309bdee5596585d93dd4bb97',
    GAIN/'NATIVE40_V2_PROTOCOL.md':'ef94ffd39c40b0be0ae8144c6e3e741f79999500c0071ed26f95b2ebf9ea8906',
    GAIN/'NATIVE40_V2_WATCH_SPEC.md':'29e9b3840fdf7920379a70ae712d02f0ceb527e7c2ca771bfb9772abf0bfaaf3'}


def merge(destination, source):
    for path,digest in source.items():
        path = str(path)
        require(path not in destination or destination[path] == digest, 'No conflicting exact owner')
        destination[path] = digest


def source_owners(args):
    mapping = {**PINS, HERE/'native37.py':args.source_sha256,
        HERE/'watch_native37.py':args.watch_sha256, HERE/'NATIVE37_PROTOCOL.md':args.protocol_sha256,
        HERE/'NATIVE37_WATCH_SPEC.md':args.watch_spec_sha256, HERE/'build37.py':args.builder_sha256,
        HERE/'BUILD37_PROTOCOL.md':args.build_protocol_sha256,
        HERE/'watch_build37.py':args.build_watch_source_sha256,
        HERE/'BUILD37_WATCH_SPEC.md':args.build_watch_spec_sha256,
        _parent['_api']['LEGACY']:_parent['_api']['LEGACY_SHA'],
        _parent['_api']['OBSERVER']:_parent['_api']['OBSERVER_SHA']}
    require(all(re.fullmatch('[0-9a-f]{64}',getattr(args,k+'_sha256') or '') for k in KEYS),
            'All eleven actual source/result authorities are explicit')
    return {str(p):d for p,d in mapping.items()}


def seal3(folder, expected_result, inputs, tick=None, exact=None):
    paths = [folder/n for n in ('result.json','completion.json','finalization.json')]
    digests = [sha(p,tick) for p in paths]
    require(digests[0] == expected_result and (exact is None or tuple(digests) == tuple(exact)),
            'Exact actual prerequisite seal hashes')
    r,e,f = map(read,paths)
    require(all(x.get(k) is True for x in (r,e,f) for k in ('passed','completed'))
        and e['result_sha256'] == f['result_sha256'] == digests[0]
        and f['completion_sha256'] == digests[1], 'Three real positive seals and exact links')
    merge(inputs,dict(zip(map(str,paths),digests)))
    return r,e,f,dict(zip(SEAL_KEYS,digests))


def controls(inputs, tick=None):
    folder = GAIN/'native40_v2'
    r,_,f,ns = seal3(folder,CONTROL_NATIVE[0],inputs,tick,CONTROL_NATIVE)
    w,_,wf,ws = seal3(GAIN/'native40_watch_v2',CONTROL_WATCH[0],inputs,tick,CONTROL_WATCH)
    require(r['schema'] == 'current_comp_shared_gain_slice_native40_v2'
        and w['schema'] == 'current_comp_shared_gain_slice_native40_watch_v2'
        and r['candidate_sha256'] == w['candidate_sha256'] == PARENT_SHA
        and r['full_model_calls'] == r['retired_call_count'] == w['retired_calls'] == 41
        and len(r['trials']) == len(r['diagnostic_records']) == len(r['input_preparation']) == 41
        and len(r['artifacts']) == 287 and r['initialization_flops'] == 1944970298
        and all(r[k] is True for k in ('own_complete_bills','all16_domains_qualified','all47_selectors_qualified',
            'repeat_byte_exact','same_production_backend_qualified','operation_ledgers_verified'))
        and tuple(w['child_'+k] for k in SEAL_KEYS) == CONTROL_NATIVE
        and r['truth_accessed'] is False and w['numerical_imports'] is False
        and all(w[k] is True for k in ('whole_owned_tree_retired','cleanup_passed','observational_groups_retired',
            'actual_completed_samples','foreign_numerical_workers_checked','hashes_unchanged','sources_unchanged'))
        and 0 <= r['elapsed_s'] <= f['elapsed_s'] < 3000
        and 0 <= w['elapsed_s'] <= wf['elapsed_s'] < 3000
        and 0 < w['maximum_combined_rss_bytes'] < 8_000_000_000
        and 0 < w['maximum_interval_s'] <= 1 and w['maximum_swap_growth_bytes'] == 0,
        'Direct current c2/OpenBLAS completed native40 boundary, not historical Accelerate')
    selected = []
    for index,flops in ((0,406399683439),(40,404454713141)):
        trial = r['trials'][index]
        require(trial['call'] == index and trial['rank'] == 37 and trial['flops'] == flops
            and trial['includes_initialization'] is (index == 0)
            and trial['multiplier'] == max(.1,flops/2**41), 'Exact current cold/warm control')
        paths = [folder/f'call_{index:02d}_{suffix}.json' for suffix in
                 ('bill','input_bill','input_reset','lifecycle','diagnostics','complete')]+[Path(trial['output'])]
        for path in paths:
            digest = r['artifacts'][str(path)]
            require(sha(path,tick) == digest, 'Selected current control artifact')
            merge(inputs,{str(path):digest})
        require(read(paths[0]) == trial and read(paths[1]) == r['input_preparation'][index]
            and read(paths[1])['weight_file_sha256'] == FIXTURE[ROW/'weights.npy']
            and r['diagnostic_records'][index] == dict(path=str(paths[4]),sha256=inputs[str(paths[4])]),
            'Same canonical37 physical fixture and selected diagnostic')
        selected.append(trial)
    return selected,dict(native=list(ns.values()),watch=list(ws.values()))


def admission(args, tick=None):
    """Stdlib-only, no output or candidate import; actual watch and worker both call it."""
    began = time.monotonic()
    def guard():
        require(time.monotonic()-began < 60 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 2_000_000_000,
                'Target-free source admission60s/self2GB inside the whole watch')
        if tick is not None:
            tick()
    require(not any(k.split('.')[0] in ('numpy','torch','flopscope','threadpoolctl')
                    for k in _parent['_api']['sys'].modules), 'Admission cannot import numerical modules')
    sources,inputs = source_owners(args),{}
    bind(sources,guard)
    builder = _parent['_api']['load'](HERE/'build37.py','pair256_native_build_metadata',args.builder_sha256)
    built = builder.verify_build(args.build_sha256,args.candidate_sha256,args.builder_sha256,
                                 args.build_protocol_sha256,guard)
    merge(sources,built['source_bindings']); merge(inputs,built['input_bindings'])
    candidate = Path(built['candidate'])
    require(candidate == HERE/'build37_v1/estimator.py' and not candidate.is_symlink()
        and sha(candidate,guard) == args.candidate_sha256 == CANDIDATE_SHA,
        'Actual emitted fixed candidate, never an invented future build')
    merge(inputs,{str(candidate):CANDIDATE_SHA})
    br,_,_,bs = seal3(HERE/'build37_v1',args.build_sha256,inputs,guard)
    w,_,wf,ws = seal3(HERE/'build37_watch_v1',args.build_watch_sha256,inputs,guard)
    require(w['schema'] == 'current_comp_shared_pair256_build37_watch_v1' and w['worker_seals'] == bs
        and br['candidate_sha256'] == w['candidate_sha256'] == CANDIDATE_SHA and w['source_build_qualified']
        and all(w[k] is True for k in ('whole_owned_tree_retired','cleanup_passed','observational_groups_retired',
            'hashes_unchanged','sources_unchanged'))
        and w['numerical_imports'] is False and 0 <= w['elapsed_s'] <= wf['elapsed_s'] < 60
        and 0 < w['maximum_combined_rss_bytes'] < 2_000_000_000
        and 0 < w['maximum_interval_s'] <= 1 and w['maximum_swap_growth_bytes'] == 0
        and w['source_bindings'][str(HERE/'watch_build37.py')] == args.build_watch_source_sha256
        and w['source_bindings'][str(HERE/'BUILD37_WATCH_SPEC.md')] == args.build_watch_spec_sha256,
        'Actual independent positive build supervisor, no circular builder authority')
    for group,destination in (('source_bindings',sources),('input_bindings',inputs),('artifacts',inputs)):
        bind(w[group],guard); merge(destination,w[group])
    selected,seals = controls(inputs,guard)
    merge(inputs,FIXTURE); bind(sources,guard); bind(inputs,guard)
    text = candidate.read_text()
    require(hashlib.sha256(text.encode()).hexdigest() == CANDIDATE_SHA, 'Exact admitted UTF8 candidate bytes')
    guard()
    return dict(candidate=str(candidate),candidate_sha256=CANDIDATE_SHA,text=text,
        proof=built['proof'],source_bindings=sources,input_bindings=inputs,controls=selected,control_seals=seals,
        build_seals=bs,build_watch_seals=ws,build_sha256=args.build_sha256,build_watch_sha256=args.build_watch_sha256,
        current_c2_controls_qualified=True,build_watch_qualified=True,tiny_prerequisite_qualified=True,
        authority={k:getattr(args,k+'_sha256') for k in KEYS})


def install_admission(result, args, tick):
    q = admission(args,tick)
    merge(result['source_bindings'],q['source_bindings']); merge(result['input_bindings'],q['input_bindings'])
    result.update({k:v for k,v in q.items() if k not in ('text','proof','source_bindings','input_bindings')})
    result.update(composition_proof=q['proof'],build_proof=q['proof'],native_source_proof=NATIVE_SOURCE_PROOF,
        parent_sha256=PARENT_SHA,parameter_updates=0,raw_error_measured=False,adjusted_score_measured=False,
        automatic_next_stage=False,uploaded=False,submission_authorized=False,
        production_five_qualified=False,current_candidate_native40_qualified=False)
    return q['text'],q['proof']


def payload(path, expected, tick=None):
    """Saved output only, bounded stdlib NPY decoding; no truth or NumPy import."""
    path = Path(path)
    require(not path.is_symlink() and path.stat().st_size <= 65536+512 and sha(path,tick) == expected,
            'Exact small saved output file')
    raw = path.read_bytes()
    require(raw[:8] == b'\x93NUMPY\x01\x00', 'Original NumPy v1 saved-output encoding')
    length = struct.unpack('<H',raw[8:10])[0]
    header = ast.literal_eval(raw[10:10+length].decode('latin1').strip())
    require(header == dict(descr='<f4',fortran_order=False,shape=(16,1024))
        and len(raw) == 10+length+65536, 'Original FP32 all16 shape and byte length')
    data = raw[10+length:]
    require(all(math.isfinite(x) and x >= 0 for (x,) in struct.iter_unpack('<f',data)),
            'All actual served values finite and nonnegative')
    return data


def final_checks(result, tick=None):
    require(len(result['trials']) == len(result['diagnostic_records']) == len(result['input_preparation']) == 2
        and len(result['artifacts']) == 14, 'Exactly both complete calls and14 original artifacts')
    expected = {str(OUTPUT/f'call_{call:02d}_{suffix}.json') for call in range(2)
                for suffix in ('bill','input_bill','input_reset','lifecycle','diagnostics','complete')}
    expected.update(str(OUTPUT/f'call_{call:02d}_rank37.npy') for call in range(2))
    require(set(result['artifacts']) == expected, 'Exactly the original seven filenames per call')
    bind(result['artifacts'],tick)
    _validate_ledgers(result)
    first = None; first_d = None; prefixes = []
    for call,(trial,control) in enumerate(zip(result['trials'],result['controls'],strict=True)):
        require(trial['output'] == str(OUTPUT/f'call_{call:02d}_rank37.npy')
            and result['diagnostic_records'][call] == dict(path=str(OUTPUT/f'call_{call:02d}_diagnostics.json'),
                sha256=result['artifacts'][str(OUTPUT/f'call_{call:02d}_diagnostics.json')]),
            'Same exact own output and diagnostic paths')
        actual = payload(trial['output'],trial['output_sha256'],tick)
        old = payload(control['output'],control['output_sha256'],tick)
        require(actual[:5*1024*4] == old[:5*1024*4], 'Unaffected first five layers exactly match current c2')
        require(hashlib.sha256(actual).hexdigest() == trial['output_bytes_sha256'], 'Saved output raw hash')
        diag = read(result['diagnostic_records'][call]['path'])
        domain,startup = diag['domains'],diag['startup']
        require(domain['variance_count'] == 16 and domain['selector_count'] == 47
            and domain['added_participant_array_operations'] == 0
            and startup['original_arena_bytes'] == 1168900096 and startup['no_second_backing_arena']
            and startup['observed_startup_region_flops'] == (1944970298 if call == 0 else 0)
            and startup['outside_prefix_materialization_flops'] == 0,
            'All original domains/selectors and own nested startup, no added arena')
        for dispatch in diag['dispatches']:
            require(dispatch['same_original_arena'] and dispatch['returned_original_out']
                and dispatch['child_bypasses_dispatcher'] and len(dispatch['children']) == 7,
                'Original observed physical serial D6 ownership')
        if call == 0:
            first,first_d = actual,diag
        else:
            physical = lambda items: [{k:v for k,v in x.items() if k != 'operation_count'} for x in items]
            require(actual == first and domain == first_d['domains']
                and physical(diag['dispatches']) == physical(first_d['dispatches']),
                    'Own repeat bytes/all16/all47/physical dispatch exact; no cross-network mask equality')
        prefixes.append(dict(call=call,layers=5,byte_exact=True))
    require(result['trials'][0]['flops']-result['trials'][1]['flops'] == 1944970298,
            'Own measured cold-minus-repeat initialization, never subtracted from served F')
    return dict(unaffected_prefix_checks=prefixes,current_c2_first5_byte_exact=True,
        current_c2_controls_qualified=True,operation_ledgers_verified=True,
        operation_counts_reported_not_rebased=True,own_repeat_physics_exact=True,
        repeat_operation_count_history=dict(cold=result['trials'][0]['operation_count'],
            repeat=result['trials'][1]['operation_count'],equality_required=False),
        all_native_outputs_sealed_before_comparison=True)


def finish_native(result, tick):
    _parent['_base']['backend_origins'](result,tick)
    result.update(final_checks(result,tick),actual_production_backend_qualified=True)


def verify_native(record, tick=None):
    """Stdlib saved acquisition API; caller authenticates the actual new three seals."""
    require(record['schema'] == SCHEMA and record['passed'] is True and record['completed'] is True
        and record['ranks'] == [37,37] and record['candidate_sha256'] == CANDIDATE_SHA
        and record['full_model_calls'] == record['retired_call_count'] == 2
        and record['native_source_proof'] == NATIVE_SOURCE_PROOF and record['initialization_flops'] == 1944970298
        and record['parameter_updates'] == 0
        and all(record.get(k) is True for k in ('all16_domains_qualified','all47_selectors_qualified',
            'repeat_byte_exact','own_complete_bills','hashes_unchanged','sources_unchanged',
            'actual_production_backend_qualified','current_c2_controls_qualified','build_watch_qualified',
            'tiny_prerequisite_qualified','all_parameter_and_estimator_owners_retired'))
        and all(record.get(k) is False for k in ('truth_accessed','score_measured','raw_error_measured',
            'adjusted_score_measured','automatic_next_stage','uploaded','submission_authorized',
            'production_five_qualified','current_candidate_native40_qualified')),
        'Actual fixed information endpoint; no score, fit, production or release authority')
    require(set(record['authority']) == set(KEYS),'Exactly eleven saved authority fields')
    args = argparse.Namespace(run=True,**{k+'_sha256':v for k,v in record['authority'].items()})
    q = admission(args,tick)
    require(record['controls'] == q['controls'] and record['control_seals'] == q['control_seals']
        and record['build_seals'] == q['build_seals'] and record['build_watch_seals'] == q['build_watch_seals']
        and record['composition_proof'] == record['build_proof'] == q['proof']
        and all(record[g].get(p) == d for g in ('source_bindings','input_bindings') for p,d in q[g].items()),
        'Actual identical build/tiny/watch/control authorities through final verification')
    a = record['backend_attestation']
    require(a['actual_origins_qualified'] and a['production_blas_qualified']
        and a['site'] == str(_parent['_base']['_BP_SITE']) and a['scipy_loaded'] is a['torch_loaded'] is False
        and a['official_prefix'] == str(OFFPY.parent.parent)
        and a['numpy_version'] == '2.4.6' and a['flopscope_version'] == '0.12.0'
        and a['whestbench_version'] == '0.16.0', 'Actual current production-compatible native backend')
    require({'numpy','numpy._core._multiarray_umath','numpy.linalg._linalg','numpy.linalg._umath_linalg',
        'flopscope','flopscope.numpy','threadpoolctl','whestbench.domain','whestbench.scoring','whestbench.seeds'}
        <= set(a['modules']), 'All required original numerical and official modules were observed')
    pools = [p for p in a['threadpools'] if p.get('user_api') == 'blas']
    require(len(pools) == 1 and pools[0]['internal_api'] == 'openblas' and pools[0]['num_threads'] == 1
        and pools[0]['version'] == '0.3.31.188.0' and pools[0]['threading_layer'] == 'pthreads'
        and Path(pools[0]['filepath']).resolve() ==
            (_parent['_base']['_BP_SITE']/'numpy/.dylibs/libscipy_openblas64_.dylib').resolve(),
        'Same exact one-thread production OpenBLAS pool')
    for name,item in a['modules'].items():
        root = _parent['_base']['_BP_OFFSITE' if name.split('.')[0] == 'whestbench' else '_BP_SITE']
        require(Path(item['path']).is_relative_to(root) and record['source_bindings'][item['path']] == item['sha256'],
                'Every actual module belongs to its declared pinned installation')
    for call,trial in enumerate(record['trials']):
        prep = record['input_preparation'][call]; marker = read(OUTPUT/f'call_{call:02d}_complete.json')
        require(trial['call'] == prep['call'] == marker['call'] == call
            and trial['rank'] == prep['rank'] == marker['rank'] == 37
            and trial['includes_initialization'] is (call == 0) and 0 < trial['flops'] <= 2**41
            and trial['multiplier'] == max(.1,trial['flops']/2**41)
            and read(OUTPUT/f'call_{call:02d}_bill.json') == trial and read(trial['input_bill']) == prep
            and prep['flops'] == 16777216 and prep['weight_file_sha256'] == FIXTURE[ROW/'weights.npy']
            and prep['official_fnp_type'] and prep['exact_bytes'] and prep['independent_weight_buffers']
            and prep['charged_to_participant'] is False and marker['passed'] and marker['completed']
            and marker['context_retired'] and marker['output_sha256'] == trial['output_sha256'],
            'Each actual complete paid call and exact separately billed canonical inputs')
        for suffix,key in (('bill','bill_sha256'),('lifecycle','lifecycle_sha256'),('diagnostics','diagnostic_sha256')):
            require(sha(OUTPUT/f'call_{call:02d}_{suffix}.json',tick) == marker[key], 'Actual retired marker artifact links')
        for bill,life in ((trial,read(trial['lifecycle_receipt'])),(prep,read(trial['input_reset_receipt']))):
            require(life['complete'] and life['context_released'] and life['closed_context_bill_unchanged_by_reset']
                and life['after_context_exit'] and life['candidate_state_reset'] is False
                and life['closed_metrics']['flops'] == bill['flops']
                and life['closed_metrics']['operation_count'] == bill['operation_count']
                and 0 < bill['wall_s'] < 120 and 0 <= bill['context_wall_s'] < 120
                and bill['peak_rss_bytes'] < 8_000_000_000
                and all(math.isfinite(bill[k]) and bill[k] >= 0 for k in
                    ('wall_s','context_wall_s','flopscope_backend_s','flopscope_overhead_s','native_residual_s')),
                'Original after-save reset/context retirement and actual120s/8GB bounds')
    checked = final_checks(record,tick)
    require(all(record[k] == v for k,v in checked.items()), 'Saved final flags equal actual metadata/payload checks')
    bind(record['source_bindings'],tick); bind(record['input_bindings'],tick)
    return checked


def _functions(path,digest,names):
    raw = path.read_text(); require(hashlib.sha256(raw.encode()).hexdigest() == digest,'Exact helper source owner')
    tree = ast.parse(raw); lines = raw.splitlines(keepends=True); pieces = []
    for name in names:
        nodes = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == name]
        require(len(nodes) == 1,'Unique helper function')
        pieces.append(''.join(lines[nodes[0].lineno-1:nodes[0].end_lineno]))
    return '\n'.join(pieces)


_ledger_source = _functions(LEDGER,LEDGER_SHA,('_v2_ledger_metrics','_v2_validate_ledgers'))
_ledger_api = dict(require=require,json=json,hashlib=hashlib,time=time,re=re)
exec(compile(_ledger_source,'<exact-original-post-context-ledger>','exec'),_ledger_api)
_ledger_metrics,_validate_ledgers = (_ledger_api[k] for k in ('_v2_ledger_metrics','_v2_validate_ledgers'))


def derive_source():
    original,prior = _parent['derive_source']()
    block = """        composition = load(HERE/'build.py','gain_slice_builder',args.builder_sha256)
        result['source_bindings'].update({str(p):v for p,v in composition.authenticate().items()})
        text, proof = composition.derive_source()
        result['composition_proof'] = proof
        result['candidate_sha256'] = hashlib.sha256(text.encode()).hexdigest()
        gain_candidate(result,text,args,composition,tick)
"""
    edits = (
        ("KEYS = ('source','watch','protocol','watch_spec','builder','build_protocol','build','candidate')",
         'KEYS = '+repr(KEYS)),
        ('return gain_source_owners(args)','return pair_source_owners(args)'),
        ("args.out = HERE/'native_v1'","args.out = HERE/'native37_v1'"),
        ("'current_comp_shared_gain_slice_native_v1'",repr(SCHEMA)),
        ("ROW = STORE/'rows/observer_fit/0035'","ROW = STORE/'rows/observer_fit/0037'"),
        ("'5067b577620cb6d5dc97c61b62f10e8fb3f98880b99fe9320f1ef8a8ca567809'",repr(FIXTURE[ROW/'row_manifest.json'])),
        ("'14563b956b4e326aee081263077c0dd1dcb86037a0b7870dc58a5e33865f47f2'",repr(FIXTURE[ROW/'weights.npy'])),
        (block,'        text,proof = install_admission(result,args,tick)\n'),
        ('return gain_controls(inputs,tick)','return pair_controls(inputs,tick)'),
        ("rows = {35:dict(weights=str(ROW/'weights.npy'),mlp_seed=int(row['mlp_seed']))}",
         "rows = {37:dict(weights=str(ROW/'weights.npy'),mlp_seed=int(row['mlp_seed']))}"),
        ("result['ranks'] = [35,35]","result['ranks'] = [37,37]"),
        ('rows=rows,ranks=(35,),result=result','rows=rows,ranks=(37,),result=result'),
        ('finish_call=finish_call,retire_call=retire_call)',
         'finish_call=finish_call,retire_call=retire_call,metrics=_ledger_metrics(legacy.metrics))'),
        ('gain_compare(np,result,tick)','finish_native(result,tick)'),
        ('        del env,observer,module\n        gc.collect()',
         "        owner_refs = [weakref.ref(env['estimator']),weakref.ref(observer),weakref.ref(module)]\n"
         '        del env,observer,module\n        gc.collect()\n'
         "        require(all(ref() is None for ref in owner_refs),'All estimator/module/observer owners retired')\n"
         "        result['all_parameter_and_estimator_owners_retired'] = True"),
    )
    changed,spans = original,[]
    for before,after in edits:
        require(changed.count(before) == 1,'Unique fixed37 metadata/ledger splice: '+before[:90])
        pos=changed.index(before);spans.append((pos,before,after));changed=changed[:pos]+after+changed[pos+len(before):]
    inverse=changed
    for pos,before,after in reversed(spans):
        require(inverse[pos:pos+len(after)] == after,'Exact positional full native inverse')
        inverse=inverse[:pos]+before+inverse[pos+len(after):]
    require(inverse == original,'Entire qualified current OpenBLAS native body restored')
    return changed,dict(owner_sha256=OWNER_SHA,parent=prior,whole_source_inverse_verified=True,
        derived_source_sha256=hashlib.sha256(changed.encode()).hexdigest(),splices=len(spans),
        original_paid_loop_and_all16_all47_observers_unchanged=True,
        post_context_ledger_owner_sha256=LEDGER_SHA,
        extracted_ledger_sha256=hashlib.sha256(_ledger_source.encode()).hexdigest(),
        no_participant_arithmetic_added_by_harness=True,fixed_current_c2_first5_boundary=True)


def command_args(**authorities):
    require(set(authorities) == set(KEYS),'Exactly eleven worker authorities')
    return ['--run',*(x for k in KEYS for x in ('--'+k.replace('_','-')+'-sha256',authorities[k]))]


def parse_child_command(command):
    require(command[:4] == [str(OFFPY),'-B','-u',str(HERE/'native37.py')], 'Actual sole worker command prefix')
    main_node = next(n for n in ast.parse(_SOURCE).body if isinstance(n,ast.FunctionDef) and n.name == 'main')
    body=main_node.body; lines=_SOURCE.splitlines(keepends=True)
    require(isinstance(body[3],ast.Assign) and body[3].targets[0].id == 'args','Original actual parser prefix')
    original=textwrap.dedent(''.join(lines[body[0].lineno-1:body[3].end_lineno]))
    require(original.count('parser.parse_args()') == 1,'Exact original native parser')
    changed=original.replace('parser.parse_args()','parser.parse_args(argv)')
    scope=dict(_api,argv=list(command[4:]))
    exec(compile(changed,'<actual-native37-parser-only>','exec'),scope)
    args=scope['args']
    require(args.run and set(vars(args)) == {'run',*(k+'_sha256' for k in KEYS)},'Actual exact worker namespace')
    return args


_SOURCE,NATIVE_SOURCE_PROOF = derive_source()
_api = dict(_parent['_api'],__file__=str(Path(__file__).resolve()),__name__='_pair256_native37_effective',
    pair_source_owners=source_owners,pair_controls=controls,install_admission=install_admission,
    finish_native=finish_native,_ledger_metrics=_ledger_metrics)
exec(compile(_SOURCE,str(Path(__file__).resolve())+':qualified-native-inverse','exec'),_api)
main = _api['main']
environment = _api['environment']
paid_loop_source = _api['paid_loop_source']


if __name__ == '__main__':
    raise SystemExit(main())
