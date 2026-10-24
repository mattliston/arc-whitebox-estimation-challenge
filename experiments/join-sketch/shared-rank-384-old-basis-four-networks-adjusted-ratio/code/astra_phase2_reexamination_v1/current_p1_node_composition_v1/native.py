"""Frozen NODE/P1 zero or terminal acquisition; inert without --run.

The qualified BD paid loop is restored by inverse source substitution. Small
existing bytes and public counters are observed; no head replay or targets.
"""
import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
BD, TL = NEW/'birth_direction_fit_v1', NEW/'current_d50_terminal_linear_v2'
P1 = NEW/'current_d50_native_objective_p1_v1'
RC = NEW/'current_d50_response10_candidate_v1/native.py'
RC_SHA = 'cccd8b50d18e6d093a48a3e3675aa20cbe2ef31cb45c2f167f153ddc842fe09c'
CENSUS = NEW/'current_d50_response10_census_v1/watch.py'
CENSUS_SHA = '7f868ba0250f142822b2aa36503d1908b21849c47e6e7406320f979e5d93bbcf'
PROTOCOL_SHA = '5aa1fefd2e7b60ce77814f39efc4dcea3bcd2cc7ed372d4841da4eb8f973696c'
BUILDER_SHA = 'f3a3c95b3a7e99a40d8951800fb9d6ec0c0252f0bd4193c3cc2941e6712e61bf'
BUILD_SHA = '78f7adbb09ce5749a442c0b4d171110627a0881a478fece4e19f4f5467ec26d6'
BUILD_END_SHA = 'd005b2245db255169bc5807349279a52271d451c57b9fa23c1c760e9b1a93306'
CANDIDATES = {
    'zero':'7bf60ce0270a6a932f7f4dcc8d136159fbf9ebde7e88e53e26dbd57754105608',
    'terminal':'2e13d40a21e62691155918719f8ec088f581d495ae17800c5503c3b56504918a'}
P1_SHA = '0deb237350e753220b575796c0278f86884b8461bfeec4ab53eae9c15522c435'
NATIVE_SHA = '8a1b888be28a7ba6d7f8e020bb9b1f2367786e189d952364710b34517ca2b35b'
OBSERVER_SHA = '744f8c46bfdbe0198eb138c1fcacc5a0b90ca6db6eb08ad7d928847c34021abe'
COMMON_SHA = '52467763b2131ad325869ea8e1fffb214ca2a89ab7bd24b3150a3350f5ac3f2a'
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
STORE_SHA = '9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50'
REFERENCES = {
    'zero':(P1/'panel_v1/terminal',
        'c8fa1cacfa98d0f45d38235283789cff67c59f32f72fa80a92dd385ed59f8c70',
        '35dd7c03bb69c0581d28d1437df4c9c856ee646e9ffa14a81085fb6bc0dd5697'),
    'terminal':(P1/'native2_terminal_v1',
        '7a1f37d80a7ed32850b10047408d85eed050220b4a6271dde90bdd21c07062e3',
        'b73a957bf78e45da0db2741cea0b33a2cd98317726edaaa0954eaece054e5957')}
VALID = (37,39,42,48,59,63,66,67,71,73,75,79,93,101,122,130,132,135,
         139,145,146,149,150,151,155,159,163,169,173,175,181,192,198,199,
         204,219,224,227,233,234)
FIXED = {'zero':(35,100),'terminal':VALID}
SCHEMA = 'current_p1_node_composition_native_v1'
WATCH_SCHEMA = 'current_p1_node_composition_native_watch_v1'
BUDGET, RSS_LIMIT = 2**41, 8_000_000_000
BLAS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
        'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
SMALL = None


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    path = Path(path).resolve()
    require(path.is_relative_to(P2) and 'truthpost' not in path.parts and not
            (path.suffix in ('.npy','.npz') and 'truth' in path.name.lower()), 'No target-array access')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')


def bind(mapping):
    for path,digest in mapping.items():
        require(Path(path).is_absolute() and re.fullmatch('[0-9a-f]{64}',digest or '')
                and sha(path) == digest, 'Immutable bound file: '+str(path))


def load(path, name, expected):
    require(sha(path) == expected, 'Reviewed source changed: '+str(path))
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def legacy():
    return load(BD/'native_v2.py','node_p1_original_native',NATIVE_SHA)


def sealed(folder, digest, end_digest=None):
    pins = {str(folder/'result.json'):digest,str(folder/'completion.json'):
            end_digest or sha(folder/'completion.json')}
    bind(pins)
    record,end = read(folder/'result.json'),read(folder/'completion.json')
    require(all(record.get(k) is True for k in ('passed','completed','hashes_unchanged'))
            and end['passed'] is True and end['completed'] is True
            and end['result_sha256'] == digest, 'Completed immutable receipt')
    return record,end,pins


def reference(role):
    folder,digest,end_digest = REFERENCES[role]
    record,end,pins = sealed(folder,digest,end_digest)
    require(record['candidate_sha256'] == P1_SHA and record['truth_accessed'] is False
            and record['own_complete_bills'] and record['all16_domains_qualified']
            and record['all47_selectors_qualified'], 'Qualified P1 comparator, no truths')
    bind(record['artifacts']); pins.update(record['artifacts'])
    require(record['source_bindings'][str(STORE/'manifest.json')] == STORE_SHA,
            'Same canonical input manifest')
    refs = {}
    for trial,prepared in zip(record['trials'],record['input_preparation']):
        require(trial['rank'] == prepared['rank'] and trial['call'] == prepared['call'], 'Reference input/call pair')
        if trial['rank'] in FIXED[role] and trial['rank'] not in refs:
            require(record['artifacts'][trial['output']] == trial['output_sha256'], 'Reference output seal')
            refs[trial['rank']] = dict(trial,weight_file_sha256=prepared['weight_file_sha256'])
    require(set(refs) == set(FIXED[role]), 'Every fixed reference rank')
    return refs,pins


def authenticate(role, build_sha, candidate_sha, builder_sha, native_sha, watcher_sha,
                 zero_result_sha=None, zero_watch_sha=None):
    """Replayable source/metadata admission; no numerical imports or targets."""
    require(role in FIXED and build_sha == BUILD_SHA and builder_sha == BUILDER_SHA
            and candidate_sha == CANDIDATES[role], 'Only the frozen literal composition')
    sources = {str(HERE/'native.py'):native_sha,str(HERE/'watch_native.py'):watcher_sha,
        str(HERE/'build.py'):BUILDER_SHA,str(HERE/'PROTOCOL.md'):PROTOCOL_SHA,
        str(RC):RC_SHA,str(CENSUS):CENSUS_SHA,str(BD/'native_v2.py'):NATIVE_SHA,
        str(TL/'native.py'):OBSERVER_SHA,str(TL/'common.py'):COMMON_SHA}
    bind(sources)
    builder = load(HERE/'build.py','node_p1_literal_builder',BUILDER_SHA)
    context = builder.verify_build(BUILD_SHA,CANDIDATES['terminal'],BUILDER_SHA,PROTOCOL_SHA)
    candidate = context['candidates'][role]
    require(Path(candidate['path']) == HERE/'build_v1'/(role+'_estimator.py')
            and candidate['sha256'] == candidate_sha, 'Requested role source')
    sources.update(context['source_bindings'])
    sources.update({v['path']:v['sha256'] for v in context['candidates'].values()})
    inputs = dict(context['input_bindings'])
    require(inputs[str(HERE/'build_v1/completion.json')] == BUILD_END_SHA, 'Original build completion')
    inputs[str(STORE/'manifest.json')] = STORE_SHA
    refs,rpins = reference(role); inputs.update(rpins)
    prerequisites = dict(build_sha256=BUILD_SHA,zero_result_sha256=zero_result_sha,
                         zero_watch_sha256=zero_watch_sha)
    if role == 'terminal':
        require(all(re.fullmatch('[0-9a-f]{64}',x or '') for x in (zero_result_sha,zero_watch_sha)),
                'Completed zero and independent-watch seals required')
        zr,ze,zp = sealed(HERE/'native_zero_v1',zero_result_sha)
        zw,we,wp = sealed(HERE/'native_zero_watch_v1',zero_watch_sha)
        zrefs,zrefs_pins = reference('zero'); inputs.update(zrefs_pins)
        verify_acquisition(zr,ze,zero_result_sha,'zero',native_sha,watcher_sha,zrefs)
        require(zw['schema'] == WATCH_SCHEMA and zw['role'] == 'zero'
                and zw['child_result_sha256'] == zero_result_sha
                and zw['child_completion_sha256'] == zp[str(HERE/'native_zero_v1/completion.json')]
                and zw['whole_owned_tree_retired'] and zw['child_execution_completed']
                and zw['completed_bill_count'] == 3 and zw['build_sha256'] == BUILD_SHA
                and zw['source_bindings'][str(HERE/'watch_native.py')] == watcher_sha
                and zw['source_bindings'][str(HERE/'native.py')] == native_sha
                and zw['maximum_swap_growth_bytes'] == 0
                and zw['maximum_sampled_rss_bytes'] < RSS_LIMIT
                and zw['maximum_sample_interval_s'] <= 1 and zw['elapsed_s'] < 240,
                'Same reviewed supervisor qualified zero before terminal')
        for rec in (zr,zw):
            for key in ('source_bindings','input_bindings','artifacts'):
                bind(rec.get(key,{})); inputs.update(rec.get(key,{}))
        inputs.update(zp); inputs.update(wp)
    else:
        require(zero_result_sha is None and zero_watch_sha is None, 'Zero has no future-stage dependency')
    bind(sources); bind(inputs)
    require(not any(k in sys.modules for k in ('numpy','torch','flopscope')), 'Admission remains stdlib-only')
    return dict(context,source_bindings=sources,input_bindings=inputs,reference_by_rank=refs,
                prerequisites=prerequisites)


def load_diagnostics():
    global SMALL
    prior = sys.modules.get('common')
    sys.modules['common'] = load(TL/'common.py','node_p1_small_common',COMMON_SHA)
    try:
        SMALL = load(TL/'native.py','node_p1_small_observer',OBSERVER_SHA)
    finally:
        if prior is None:
            sys.modules.pop('common',None)
        else:
            sys.modules['common'] = prior


def bound_depths(module):
    return [] if module._SCRATCH is None else [d for d in range(3,7) if module._SCRATCH[d] is not None]


class Observer:
    """Small detached bytes; all coefficient calls counted without overlap."""
    def __init__(self,module):
        self.module,self.ctx = module,None
        names = ('_predict_learned_compact','_predict_binary','_cp3_bind_arena_views',
                 '_bd_coefficients','_bdnh_parameters')
        self.codes = {name:getattr(module,name).__code__ for name in names}
        require(all(code.co_filename == module.__file__ for code in self.codes.values()), 'Actual frozen code owners')
        self.names = {id(getattr(module,name)):name for name in
            ('_PACKED_PUSH','_PACKED_READ','_PACKED_SQUARE','_PACKED_SQUARE4','_PACKED_SQUARE5','_PACKED_SQUARE6')}
        self.begin()

    def begin(self):
        self.ctx = None
        self.small = SMALL.Observer(self.module)
        self.pending,self.bindings,self.materializations = {},[],[]
        self.prefix = None
        self.depths_before = bound_depths(self.module)

    def __call__(self,frame,event,answer):
        self.small(frame,event,answer)
        code = frame.f_code
        if code is self.codes['_predict_learned_compact'] and event == 'call':
            require(self.prefix is None, 'One learned-predict prefix')
            self.prefix = dict(start=int(self.ctx.flops_used),P_empty_at_entry=self.module._GNN_PARAMS is None,
                cstar_empty_at_entry=self.module._BD_COEFFICIENTS is None,
                A_empty_at_entry=self.module._BDNH_PARAMETERS is None)
        elif code is self.codes['_predict_binary'] and event == 'call':
            require(self.prefix is not None and 'flops' not in self.prefix, 'One binary entry')
            self.prefix.update(flops=int(self.ctx.flops_used)-self.prefix.pop('start'),
                P_ready_at_binary_entry=self.module._GNN_PARAMS is not None,
                cstar_ready_at_binary_entry=self.module._BD_COEFFICIENTS is not None,
                A_ready_at_binary_entry=self.module._BDNH_PARAMETERS is not None)
        elif code is self.codes['_cp3_bind_arena_views']:
            if event == 'call':
                specs = frame.f_locals['view_specs']; name = self.names.get(id(specs))
                require(name is not None and 'binding' not in self.pending, 'Original nonrecursive layout binding')
                self.pending['binding'] = dict(name=name,start=int(self.ctx.flops_used),
                    expected_flops=sum(stop-start for start,stop,shape in specs),
                    arena_dtype=str(frame.f_locals['arena'].dtype))
            elif event == 'return':
                item = self.pending.pop('binding'); item['flops'] = int(self.ctx.flops_used)-item.pop('start')
                require(item['arena_dtype'] == 'float32' and item['flops'] == item['expected_flops'], 'Owned binding tariff')
                self.bindings.append(item)
        else:
            for name,cache,literal in (('_bd_coefficients','_BD_COEFFICIENTS','_BD_COEFFICIENT_LITERAL'),
                                        ('_bdnh_parameters','_BDNH_PARAMETERS','_BDNH_A_LITERAL')):
                if code is not self.codes[name]:
                    continue
                if event == 'call':
                    require(name not in self.pending, 'Nonrecursive parameter materialization')
                    self.pending[name] = dict(name=name,start=int(self.ctx.flops_used),
                        empty_at_entry=getattr(self.module,cache) is None,
                        inside_prefix=self.prefix is not None and 'flops' not in self.prefix,
                        literal_elements=sum(len(row) for row in getattr(self.module,literal)))
                elif event == 'return':
                    item = self.pending.pop(name); item['flops'] = int(self.ctx.flops_used)-item.pop('start')
                    require(item['flops'] == (2*item['literal_elements'] if item['empty_at_entry'] else 0),
                            'Actual once-only FP32 literal materialization')
                    self.materializations.append(item)

    def finish(self):
        require(self.ctx is None and not self.pending and self.prefix is not None
                and all(self.prefix[k] for k in ('P_ready_at_binary_entry','cstar_ready_at_binary_entry','A_ready_at_binary_entry')),
                'Released context and complete prefix')
        coeff = [x for x in self.materializations if x['name'] == '_bd_coefficients']
        node = [x for x in self.materializations if x['name'] == '_bdnh_parameters']
        require(len(coeff) == 2 and [x['inside_prefix'] for x in coeff] == [True,False]
                and len(node) == 1 and node[0]['inside_prefix'] and node[0]['literal_elements'] == 256,
                'Both actual cstar calls and sole NODE-A call observed')
        outside = sum(x['flops'] for x in self.materializations if not x['inside_prefix'])
        startup = dict(prefix=dict(self.prefix),bindings=list(self.bindings),materializations=list(self.materializations),
            square_depths_bound_before=list(self.depths_before),square_depths_bound_after=bound_depths(self.module),
            outside_prefix_materialization_flops=outside,
            observed_startup_region_flops=self.prefix['flops']+sum(x['flops'] for x in self.bindings)+outside,
            prefix_scope='P, first cstar, NODE A, and original zero-state allocation; nested materializations not added twice',
            NODE_A_materialization_flops=node[0]['flops'],public_counters_only=True,added_participant_array_operations=0)
        return self.small.export_bytes(),startup


def observed_predict(estimator,mlp,observer):
    require(sys.getprofile() is None, 'Do not replace another profiler')
    sys.setprofile(observer)
    try:
        return estimator.predict(mlp,BUDGET)
    finally:
        sys.setprofile(None)


def after_bill(np,observer,trial,result,folder,refs):
    packet,startup = observer.finish()
    domains = SMALL.diagnostics(np,packet)
    domains['selection_cells'] = {key:{k:v[k] for k in ('count','floor','indices_sha256','mask_sha256')}
                                 for key,v in domains['selection_cells'].items()}
    cold = trial['call'] == 0
    require(all(startup['prefix'][k] is cold for k in ('P_empty_at_entry','cstar_empty_at_entry','A_empty_at_entry'))
            and startup['NODE_A_materialization_flops'] == (512 if cold else 0), 'P/cstar/A initialized once')
    ref = refs[trial['rank']]
    require(result['input_preparation'][-1]['weight_file_sha256'] == ref['weight_file_sha256'], 'Same canonical P1 input bytes')
    reference_check = dict(path=ref['output'],sha256=ref['output_sha256'],compared=False,
                           operation_count_compared=False,old_bill_transplanted=False)
    if result['role'] == 'zero':
        require(sha(ref['output']) == ref['output_sha256'], 'Pinned P1 output before decode')
        own,old = np.load(trial['output'],allow_pickle=False),np.load(ref['output'],allow_pickle=False)
        exact = (own.shape == old.shape == (16,1024) and own.dtype == old.dtype == np.float32
                 and np.isfinite(old).all() and (old >= 0).all() and own.tobytes() == old.tobytes()
                 and hashlib.sha256(old.tobytes()).hexdigest() == ref['output_bytes_sha256'])
        reference_check.update(compared=True,all16_byte_exact=bool(exact))
        require(exact, 'Zero NODE must reproduce every P1 output byte')
        del own,old
    path = folder/f"call_{trial['call']:02d}_diagnostics.json"
    write(path,dict(call=trial['call'],rank=trial['rank'],domains=domains,startup=startup,
        P1_reference=reference_check,truth_accessed=False,offline_after_saved_bill_and_context_release=True))
    item = dict(path=str(path),sha256=sha(path))
    result['artifacts'][str(path)] = item['sha256']; result['diagnostic_records'].append(item)


def finish_native(result):
    role = result['role']; sequence = FIXED[role]+(FIXED[role][0],)
    require([t['rank'] for t in result['trials']] == list(sequence)
            and len(result['diagnostic_records']) == len(sequence), 'Whole fixed primary/repeat roster')
    records = [read(x['path']) for x in result['diagnostic_records']]
    require(records[0]['domains'] == records[-1]['domains'] and result['trials'][0]['output_bytes_sha256'] ==
            result['trials'][-1]['output_bytes_sha256'], 'Repeated complete output/domain/order bytes')
    measured = result['trials'][0]['flops']-result['trials'][-1]['flops']
    observed = records[0]['startup']['observed_startup_region_flops']-records[-1]['startup']['observed_startup_region_flops']
    require(measured == observed and observed > 0 and not records[-1]['startup']['bindings'], 'Source-owned cold/repeat difference')
    result.update(passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True,
        initialization_flops=measured,initialization_ownership_passed=True,
        initialization_accounting=dict(cold_minus_repeat_flops=measured,observed_startup_difference_flops=observed,
            per_call_observed_startup_region_flops=[r['startup']['observed_startup_region_flops'] for r in records],
            startup_subtracted_from_primary_bills=False,old_initialization_value_assumed=False,
            nested_prefix_materializations_not_double_counted=True),
        all16_domains_qualified=True,all47_selectors_qualified=True,repeat_output_domain_byte_exact=True,
        own_complete_bills=True,all_outputs_and_bills_sealed_before_scoring=True,
        zero_all16_P1_byte_exact=role == 'zero')


def npy_bytes(path,expected_raw):
    """Stdlib header/byte verification only; no numerical array decoding."""
    with Path(path).open('rb') as stream:
        require(stream.read(6) == b'\x93NUMPY', 'NPY magic')
        version = stream.read(2); require(version in (b'\x01\x00',b'\x02\x00'), 'Known NPY header')
        size = int.from_bytes(stream.read(2 if version[0] == 1 else 4),'little')
        require(0 < size < 65536, 'Bounded NPY header')
        header = ast.literal_eval(stream.read(size).decode('latin1'))
        raw = stream.read()
    require(header == dict(descr='<f4',fortran_order=False,shape=(16,1024)) and len(raw) == 16*1024*4
            and hashlib.sha256(raw).hexdigest() == expected_raw, 'Exact saved FP32 output frame/raw bytes')


def verify_acquisition(record,end,digest,role,native_sha,watcher_sha,refs):
    folder = HERE/f'native_{role}_v1'; sequence = FIXED[role]+(FIXED[role][0],)
    flags = ('passed','completed','hashes_unchanged','sources_unchanged','own_complete_bills',
             'all16_domains_qualified','all47_selectors_qualified','repeat_output_domain_byte_exact',
             'initialization_ownership_passed','source_inverse_verified','paid_arithmetic_and_lifecycle_unchanged')
    require(record['schema'] == SCHEMA and record['role'] == role and all(record.get(k) is True for k in flags)
            and end['passed'] is True and end['completed'] is True and end['result_sha256'] == digest
            and record['truth_accessed'] is False and record['candidate_sha256'] == CANDIDATES[role]
            and record['build_sha256'] == BUILD_SHA and record['builder_sha256'] == BUILDER_SHA,
            'Completed immutable target-free role')
    require(record['source_bindings'][str(HERE/'native.py')] == native_sha
            and record['source_bindings'][str(HERE/'watch_native.py')] == watcher_sha
            and record['source_bindings'][str(HERE/'PROTOCOL.md')] == PROTOCOL_SHA,
            'Same reviewed numerical and external-supervisor owners')
    for group in ('source_bindings','input_bindings','artifacts'):
        bind(record.get(group,{}))
    require(record['ranks'] == list(FIXED[role]) and len(record['trials']) == len(record['input_preparation']) ==
            len(record['diagnostic_records']) == len(sequence) and len(record['artifacts']) == 6*len(sequence),
            'Exactly five original artifacts and one small diagnostic per actual call')
    helper = load(CENSUS,'node_p1_lifecycle_reader',CENSUS_SHA)
    domains,startups = [],[]
    for call,(trial,prepared) in enumerate(zip(record['trials'],record['input_preparation'])):
        require(trial['call'] == prepared['call'] == call and trial['rank'] == prepared['rank'] == sequence[call]
                and trial['includes_initialization'] is (call == 0), 'Fixed input/prediction call ownership')
        for bill in (trial,prepared):
            require(type(bill['flops']) is int and 0 < bill['flops'] <= BUDGET and
                    all(math.isfinite(bill[k]) and bill[k] >= 0 for k in
                        ('wall_s','context_wall_s','flopscope_backend_s','flopscope_overhead_s','native_residual_s'))
                    and bill['wall_s'] < 120 and bill['context_wall_s'] < 120
                    and 0 < bill['peak_rss_bytes'] < RSS_LIMIT, 'Complete bill/resource gates')
        for prefix,bill in (('',trial),('_input',prepared)):
            bp = folder/f'call_{call:02d}{prefix}_bill.json'
            lp = folder/f"call_{call:02d}_{'lifecycle' if not prefix else 'input_reset'}.json"
            require(read(bp) == bill and record['artifacts'][str(bp)] == sha(bp)
                    and record['artifacts'][str(lp)] == sha(lp), 'Own exact bill/reset files')
            helper.lifecycle(dict(bill,lifecycle_receipt=str(lp)),bp,lp)
        require(prepared['flops'] == 16*1024*1024 and prepared['layers'] == 16
                and prepared['charged_to_participant'] is False and all(prepared[k] is True for k in
                    ('official_fnp_type','exact_bytes','independent_weight_buffers','context_closed'))
                and prepared['weight_file_sha256'] == refs[trial['rank']]['weight_file_sha256'],
                'Official independent FP32 weight ownership, separate complete bill')
        output = folder/f"call_{call:02d}_rank{trial['rank']}.npy"
        require(trial['output'] == str(output) and trial['output_sha256'] == record['artifacts'][str(output)] == sha(output),
                'Actual saved prediction file')
        npy_bytes(output,trial['output_bytes_sha256'])
        dp = folder/f'call_{call:02d}_diagnostics.json'; item = record['diagnostic_records'][call]
        require(item == dict(path=str(dp),sha256=sha(dp)) and record['artifacts'][str(dp)] == sha(dp), 'Own diagnostic seal')
        diagnostic = read(dp); domain,startup = diagnostic['domains'],diagnostic['startup']
        require(diagnostic['call'] == call and diagnostic['rank'] == sequence[call]
                and diagnostic['truth_accessed'] is False and diagnostic['offline_after_saved_bill_and_context_release'],
                'Observation only after paid-bill/reset/release')
        keys = {f'{l}:{s}' for l in range(16) for s in range(2 if l == 15 else 3)}
        require(domain['variance_count'] == 16 and domain['selector_count'] == 47
                and set(domain['variance_domain']) == {str(l) for l in range(16)}
                and set(domain['selection_cells']) == keys and all(v['finite'] and math.isfinite(v['minimum'])
                    and v['minimum'] >= 1e-10 and v['below_floor_count'] == 0 for v in domain['variance_domain'].values())
                and all(v['floor'] == .01 and type(v['count']) is int and 0 <= v['count'] <= 1024
                        and v['count'] % 32 == 0 for v in domain['selection_cells'].values()), 'Unchanged .01 domains/selectors')
        require(domain['added_participant_array_operations'] == startup['added_participant_array_operations'] == 0
                and startup['public_counters_only'] and startup['NODE_A_materialization_flops'] == (512 if call == 0 else 0),
                'Passive observer; NODE A paid once')
        if role == 'zero':
            require(diagnostic['P1_reference']['all16_byte_exact'] and trial['output_bytes_sha256'] ==
                    refs[trial['rank']]['output_bytes_sha256'], 'Zero all16 P1 output identity')
        domains.append(domain); startups.append(startup)
    require(domains[0] == domains[-1] and record['trials'][0]['output_bytes_sha256'] ==
            record['trials'][-1]['output_bytes_sha256'], 'Exact repeated bytes and charts')
    measured = record['trials'][0]['flops']-record['trials'][-1]['flops']
    observed = startups[0]['observed_startup_region_flops']-startups[-1]['observed_startup_region_flops']
    account = record['initialization_accounting']
    require(measured > 0 and measured == observed == record['initialization_flops'] == account['cold_minus_repeat_flops']
            == account['observed_startup_difference_flops'] and not account['startup_subtracted_from_primary_bills']
            and not account['old_initialization_value_assumed'] and account['nested_prefix_materializations_not_double_counted'],
            'Actual complete bills, measured nonoverlapping initialization ownership')


def derived_main():
    prior = load(RC,'node_p1_native_source_parent',RC_SHA)
    source,original_sha = prior.derived_main()
    before,after = '    ranks=(35,100)\n','    ranks=FIXED_RANKS\n'
    require(source.count(before) == 1, 'Only fixed role roster differs from reviewed observation adapter')
    changed = source.replace(before,after)
    require(changed.replace(after,before) == source, 'Roster inverse to reviewed BD-loop adapter')
    return changed,original_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true'); parser.add_argument('--role',choices=tuple(FIXED))
    for name in ('source','builder','build','candidate','protocol','watcher','zero-result','zero-watch'):
        parser.add_argument('--'+name+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    require(args.role in FIXED and args.protocol_sha256 == PROTOCOL_SHA, 'Explicit role and reviewed protocol')
    require(sys.flags.optimize == 0 and os.getpriority(os.PRIO_PROCESS,0) >= 10
            and all(os.environ.get(k) == '1' for k in BLAS) and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
            and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0', 'Assertions/nice10/BLAS1/no fallback')
    require(not any(k in sys.modules for k in ('numpy','torch','flopscope')), 'All admission before numerical imports')
    context = authenticate(args.role,args.build_sha256,args.candidate_sha256,args.builder_sha256,
                           args.source_sha256,args.watcher_sha256,args.zero_result_sha256,args.zero_watch_sha256)
    versions = {k:importlib.metadata.version(k) for k in ('numpy','flopscope','whestbench','threadpoolctl')}
    require(versions == dict(numpy='2.4.6',flopscope='0.12.0',whestbench='0.16.0',threadpoolctl='3.6.0'), 'Official native versions')
    load_diagnostics(); old = legacy(); source,base_sha = derived_main()
    require(tuple(old.VALID) == VALID, 'Original fixed validation frame')
    folder = HERE/f'native_{args.role}_v1'; candidate = Path(context['candidates'][args.role]['path'])
    require(not folder.exists() and not folder.is_symlink() and not HERE.is_symlink(), 'Exclusive output, no retry/resume')
    pins = {**context['source_bindings'],**context['input_bindings']}
    namespace = dict(vars(old),__file__=str(Path(__file__).resolve()),PARENT=NEW,CANDIDATE=candidate,
        NATIVE_PINS=pins,FIXED_RANKS=FIXED[args.role],Observer=Observer,observed_predict=observed_predict,
        after_bill=lambda np,observer,trial,result,out:after_bill(np,observer,trial,result,out,context['reference_by_rank']),
        finish_native=finish_native,NATIVE_HEADER=dict(schema=SCHEMA,stage='native',role=args.role,completed=False,
            sources_unchanged=False,build_sha256=BUILD_SHA,builder_sha256=BUILDER_SHA,prerequisites=context['prerequisites'],
            versions=versions,truth_accessed=False,no_fit=True,score_measured=False,old_NODE_gate_still_failed=True,
            native_original_sha256=NATIVE_SHA,native_main_sha256=base_sha,native_adapter_parent_sha256=RC_SHA,
            source_inverse_verified=True,paid_arithmetic_and_lifecycle_unchanged=True,
            original_TL_diagnostics_unchanged=True,response_floor=.01,active_floor=.01,birth_floor=.01,
            own_complete_bills=False,old_initialization_value_assumed=False,auto_launch_next_stage=False))
    os.chdir(NEW); saved = sys.argv
    try:
        sys.argv = [str(__file__),'--source',str(candidate),'--split',
                    'development' if args.role == 'zero' else 'validation','--out',str(folder)]
        exec(compile(source,str(__file__)+':preserved_native','exec'),namespace)
        return namespace['main']()
    finally:
        sys.argv = saved


if __name__ == '__main__':
    raise SystemExit(main())
