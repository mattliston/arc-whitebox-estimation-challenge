"""Target-free35/35: unchanged paid native card with the recorded production BLAS."""
import ast
import hashlib
from pathlib import Path
import sys

_BP_HERE = Path(__file__).resolve().parent
_BP_OWNER = _BP_HERE/'probe.py'
_BP_OWNER_SHA = '6193396f83150e7e95f3c16ad5095eeb8d4c62da17d69d5bd7dcb08fad9a743b'
_BP_NEW, _BP_P2 = _BP_HERE.parent, _BP_HERE.parent.parent
_BP_SITE = _BP_NEW/'openblas_runtime_env_v1/lib/python3.11/site-packages'
_BP_OFFSITE = _BP_P2/'data_gate/.venv/lib/python3.11/site-packages'
_BP_CANDIDATE_SHA = '4d98fef2c11c571efe1eca945018bb9fe9ece1454d16b4abde2eb52d71e755c3'
_BP_RUNTIME_SEALS = (
    'c8d325e35229396bf67f63fd65e92f09f75ed8305ef1e169388d975507318af0',
    'b37e54981a24af7433fd65b8ca90b38fb84d092a910f91c7655542b658169f6d',
    '40b6bde559cbee4960cd8a5202bd12c08e241cfa3d5dcea38d460b43a67bcc48')
_BP_NEGATIVE_SEALS = {
    'fidelity_v2':(
        'fcffdc67d39025333f5cb1c15cd1ee5cbd2c1df42cec01355b93d67ce0c381c9',
        'ec07d8e9b44dad72b36d96a589068e4f248cc7495febfd6cbbc1e5d43af79354',
        '0281a990041ee93dd1925fefbd51a4dde216409606789e5822b156fe4e511a0a'),
    'fidelity_watch_v2':(
        '1dccf7ea43149a8f19d8b5fc79bddb93a9ff1d88bf3a555b42d87159e20100f3',
        '107fa0e8cbe1b3a646e92b15e707b6b762ab0e4742472cabf682777d335f15c0',
        'b6ec1d725247d61c553531585076175aabbefbbb59ac2b9276f89a6281c772e6')}
_BP_PINS = {
    _BP_OWNER:_BP_OWNER_SHA,
    _BP_HERE/'watch_probe.py':'7ade1c9025a2932e7929498062f894b72a3817d50027e7c5d1cfbf22a1f0fd18',
    _BP_HERE/'PROBE_PROTOCOL.md':'43e0c94e729a7004d2d51fb6424d4122e7d6f168a1f241095ff33abac2ad04c2',
    _BP_HERE/'PROBE_WATCH_SPEC.md':'743fc63fcb7ff5a7517fcb4e4311ee586fbd8ec72ed36004e2883810298cdc3f',
    _BP_NEW/'openblas_runtime_environment_v1.json':'fa5bbf942d81a67b91c1304afa20726faf5a26ed0047c3cb60828c053678fb90',
    _BP_NEW/'openblas_runtime_env_v1/pyvenv.cfg':'7513b2eb1c0531598377bcf1677fda972031bf33a852d00863364a4bc33b68bc',
    _BP_SITE/'numpy/__init__.py':'92a46f791e453926d3292af2582b89995a289475f0eaaea71a949823200b838a',
    _BP_SITE/'numpy/__config__.py':'97bedd19442e4714a9147f49cd93357032d21d791d87ca1367f44766db0771e6',
    _BP_SITE/'numpy/_core/_multiarray_umath.cpython-311-darwin.so':'b9a53273d646f3ffbf8307725caa306bd81c7ade0815ae999af98e0cf3e8ebec',
    _BP_SITE/'numpy/linalg/_linalg.py':'e47d0e2aed21361291cd6651999a59e6479ba5866abb8eabb1dbea5b0d1618ac',
    _BP_SITE/'numpy/linalg/_umath_linalg.cpython-311-darwin.so':'2dfac87008f01faaaea501c8e316f59f8dc083366f6f8648072cb842d5da7f4b',
    _BP_SITE/'numpy/.dylibs/libscipy_openblas64_.dylib':'bd3ad3014a7977f69921ceb2f5d7b0a8fb4b96dfb0653f87bd3bfba5af806d78',
    _BP_SITE/'flopscope/__init__.py':'cd3d8b325339717390cee951c5cbeee58c9521aa0e8d642ae62f63c39a2ccdcb',
    _BP_SITE/'flopscope/numpy/__init__.py':'33edacfc6ca01866d4f558e3a43f8fadfd6a2c0efdbc4e4712acc1d839ad4e2f',
    _BP_SITE/'flopscope/_budget.py':'ad9e78172bd0e8175533d4aae75b67df749d96ecc1ca868135eb3d781e542d01',
    _BP_SITE/'flopscope/_array_ops.py':'e45e927f18c54352503a4951d3d925a5f31f92ecea245e32360825d71a05da6c',
    _BP_SITE/'flopscope/_ndarray.py':'fa5bbecb48b123ab77f31ced3da877d2e333aa5b403a324c33da591b967d3e74',
    _BP_SITE/'flopscope/_registry.py':'4e091020d5cb8167271fa6ed76ac43983664e3a3cdd9e67ae76968f3884f6623',
    _BP_SITE/'flopscope/_dtype_billing.py':'d6f4df8b2fcf162f41afde2dae23a66d894eba62f173e57d89051e5873569cb8',
    _BP_SITE/'flopscope/numpy/linalg/_decompositions.py':'0cd6f08ac3fe3cef97d506be6713d0aa85f6ced74e8fd35016a420bfe6d8d942',
    _BP_SITE/'flopscope/_einsum.py':'0f25b8f5709c5a96404f20c38afc9b705568ab2654ab6c5fc81f3de292b73284',
    _BP_SITE/'threadpoolctl.py':'12fb9526b6a74d2e686b7ec148dc165c3587999b14fa86984aa180e10802400b'}


def backend_select():
    """Worker-only explicit package selection; no install, env mutation or import."""
    require(not any(k.split('.')[0] in ('numpy','scipy','flopscope','torch','threadpoolctl')
                    for k in sys.modules), 'Select backend before every numerical import')
    require(Path(sys.prefix).resolve() == OFFPY.parent.parent.resolve() and sys.version_info[:3] == (3,11,5)
        and not os.environ.get('PYTHONPATH') and not os.environ.get('FLOPSCOPE_SERVER_URL'),
        'Original OFFPY prefix and explicit local-only backend selection')
    bind({str(p):v for p,v in _BP_PINS.items()})
    require(str(_BP_SITE) not in sys.path and _BP_SITE.is_dir(), 'One fresh explicit site prepend')
    sys.path.insert(0,str(_BP_SITE))
    for name,root in (('numpy',_BP_SITE),('flopscope',_BP_SITE),('threadpoolctl',_BP_SITE),
                      ('whestbench',_BP_OFFSITE)):
        origin = importlib.util.find_spec(name).origin
        require(origin is not None and Path(origin).resolve().is_relative_to(root.resolve()),
                'Declared pre-import origin: '+name)
    require(importlib.util.find_spec('scipy') is None, 'No unowned SciPy numerical backend')


def backend_candidate(result,text,tick):
    path = HERE/'build_v1/estimator.py'
    require(result['candidate_sha256'] == sha(path,tick) == _BP_CANDIDATE_SHA
        and path.read_bytes() == text.encode(), 'Built production source equals the unchanged exact composition')
    result['input_bindings'][str(path)] = _BP_CANDIDATE_SHA
    result.update(candidate=str(path),backend_probe_source_proof=BACKEND_SOURCE_PROOF,
        estimator_source_unchanged=True,backend_selection='explicit production OpenBLAS site prepend',
        old_fidelity_still_failed=True,production_five_qualified=False,native_fidelity_qualified=False,
        fresh_same_backend_objective_required_before_bank=True,automatic_next_stage=False,
        uploaded=False,submission_authorized=False)


def backend_controls(inputs,tick=None):
    """Direct actual production authority; no ancestor or prior scorer rerun."""
    names = ('result.json','completion.json','finalization.json')
    paths = [HERE/'runtime_v1'/n for n in names]
    ds = tuple(sha(p,tick) for p in paths)
    require(ds == _BP_RUNTIME_SEALS, 'The actual unchanged five-call production chain')
    r,e,f = map(read,paths)
    require(all(x.get('passed') is True and x.get('completed') is True for x in (r,e,f))
        and e['result_sha256'] == f['result_sha256'] == ds[0] and f['completion_sha256'] == ds[1]
        and r['schema'] == 'current_comp_shared_old384_runtime_v1'
        and r['candidate_sha256'] == _BP_CANDIDATE_SHA and r['full_model_calls'] == 5
        and r['initialization_flops'] == 1944970298 and r['native_fidelity_qualified'] is False
        and r['production_five_qualified'] is False and r['truth_arrays_accessed'] is False
        and 0 <= r['elapsed_s'] < 600 and 0 <= f['elapsed_s'] < 600
        and 0 < r['maximum_combined_rss_bytes'] < RSS and 0 < r['maximum_interval_s'] <= 1
        and r['maximum_swap_growth_bytes'] == 0 and all(r[k] is True for k in
            ('all5_complete','all5_production_timing_passed','repeat_byte_exact','own_complete_bills',
             'hashes_unchanged','sources_unchanged','whole_owned_tree_retired','cleanup_passed',
             'observational_groups_retired','actual_completed_samples')),
        'Positive timing/retirement is not prior fidelity authority')
    inputs.update({str(p):d for p,d in zip(paths,ds)})
    for group in ('source_bindings','artifacts'):
        bind(r[group],tick)
        for path,digest in r[group].items():
            require(path not in inputs or inputs[path] == digest, 'Consistent direct runtime evidence')
            inputs[path] = digest
    row = read(ROW/'row_manifest.json')
    require(row['files']['weights.npy']['sha256'] == FIXTURE[ROW/'weights.npy']
        and r['fixture_bindings']['35'] == dict(path=str(ROW/'weights.npy'),sha256=FIXTURE[ROW/'weights.npy']),
        'Exact physical production35 fixture')
    records = r['observed_records']
    require([x['rank'] for x in records] == [35,100,500,3016500,35], 'Original five-call history')
    for index,x in enumerate(records):
        require(x['index'] == index and x['output_dtype'] == 'float32' and x['output_shape'] == [16,1024]
            and 0 < x['flops'] <= BUDGET and all(math.isfinite(x['timing'][k])
                and x['timing'][k] >= 0 for k in ('wall_time_s','residual_wall_time_s'))
            and x['timing']['wall_time_s'] < 60 and x['timing']['residual_wall_time_s'] < .4,
            'Every original production call keeps its complete bill and timing gates')
    selected = []
    for index in (0,4):
        x = records[index]
        path = NEW/f'current_comp_shared_old384_runtime_v1_{index}_r35.f32'
        require(path.stat().st_size == 65536 and sha(path,tick) == x['output_sha256'] == r['artifacts'][str(path)],
                'Actual all16 production payload')
        selected.append(dict(call=index,rank=35,output=str(path),output_sha256=x['output_sha256'],
            flops=x['flops'],includes_initialization=(index == 0),timing=x['timing']))
    require(selected[0]['output_sha256'] == selected[1]['output_sha256']
        and selected[0]['flops']-selected[1]['flops'] == 1944970298, 'Exact own production repeat/startup')
    negative = {}
    for folder,expected in _BP_NEGATIVE_SEALS.items():
        pp = [HERE/folder/n for n in names]
        dd = tuple(sha(p,tick) for p in pp); rr,ee,ff = map(read,pp)
        require(dd == expected and all(x.get('passed') is False and x.get('completed') is False
            for x in (rr,ee,ff)) and ee['result_sha256'] == ff['result_sha256'] == dd[0]
            and ff['completion_sha256'] == dd[1], 'Original genuine fidelity failure is immutable and negative')
        inputs.update({str(p):d for p,d in zip(pp,dd)}); negative[folder] = list(dd)
    return selected,dict(runtime=list(ds),old_negative_fidelity=negative)


def backend_origins(result,tick=None):
    """No numerical operation: attest actual imported owners, not a version label."""
    expected = {'numpy':_BP_SITE/'numpy/__init__.py',
        'numpy._core._multiarray_umath':_BP_SITE/'numpy/_core/_multiarray_umath.cpython-311-darwin.so',
        'numpy.linalg._linalg':_BP_SITE/'numpy/linalg/_linalg.py',
        'numpy.linalg._umath_linalg':_BP_SITE/'numpy/linalg/_umath_linalg.cpython-311-darwin.so',
        'flopscope':_BP_SITE/'flopscope/__init__.py',
        'flopscope.numpy':_BP_SITE/'flopscope/numpy/__init__.py',
        'threadpoolctl':_BP_SITE/'threadpoolctl.py'}
    for name,path in expected.items():
        if name.startswith('numpy.linalg.') and name not in sys.modules:
            importlib.import_module(name)
        m = sys.modules.get(name)
        require(m is not None and Path(m.__file__).resolve() == path.resolve()
            and sha(path,tick) == _BP_PINS[path], 'Actual pinned production module: '+name)
    require(not any(k.split('.')[0] in ('scipy','torch') for k in sys.modules),
            'No SciPy/Torch alternative numerical owner')
    origins = {}
    for name,m in tuple(sys.modules.items()):
        root = name.split('.')[0]
        if root not in ('numpy','flopscope','threadpoolctl','whestbench'):
            continue
        p = getattr(m,'__file__',None)
        if p is None:
            continue
        p = Path(p).resolve(); base = _BP_OFFSITE if root == 'whestbench' else _BP_SITE
        require(p.is_relative_to(base.resolve()), 'No mixed-install numerical module: '+name)
        digest = sha(p,tick)
        require(str(p) not in result['source_bindings'] or result['source_bindings'][str(p)] == digest,
                'Imported source cannot change an earlier owner pin')
        result['source_bindings'][str(p)] = digest
        origins[name] = dict(path=str(p),sha256=digest)
    prior = result.get('backend_attestation',{})
    old = prior.get('modules',{})
    require(all(origins.get(k) == v for k,v in old.items()), 'All already imported owners stay exact')
    result['backend_attestation'] = dict(prior,site=str(_BP_SITE),official_prefix=str(OFFPY.parent.parent),
        numpy_version=importlib.metadata.version('numpy'),flopscope_version=importlib.metadata.version('flopscope'),
        whestbench_version=importlib.metadata.version('whestbench'),modules=origins,
        scipy_loaded=False,torch_loaded=False,actual_origins_qualified=True)


def backend_threadpool_info(original,result,tick):
    """Called at the original threadpool_info site, before any paid context."""
    def information():
        pools = original()
        blas = [p for p in pools if p.get('user_api') == 'blas']
        require(len(blas) == 1 and blas[0]['internal_api'] == 'openblas'
            and blas[0]['num_threads'] == 1 and blas[0]['version'] == '0.3.31.188.0'
            and blas[0]['threading_layer'] == 'pthreads'
            and Path(blas[0]['filepath']).resolve() == (_BP_SITE/'numpy/.dylibs/libscipy_openblas64_.dylib').resolve(),
            'Exactly the production NumPy OpenBLAS, at one thread; no Accelerate pool')
        backend_origins(result,tick)
        result['backend_attestation'].update(threadpools=pools,production_blas_qualified=True)
        return pools
    return information


def backend_compare(np,result,tick):
    """After BOTH native contexts retire: no truth, fit, score, or gate change."""
    require(result['retired_call_count'] == result['full_model_calls'] == 2
        and len(result['artifacts']) == 14, 'Both native outputs and all14 owned artifacts sealed before comparison')
    bind(result['artifacts'],tick)
    backend_origins(result,tick)
    checks = []
    for call,(trial,control) in enumerate(zip(result['trials'],result['controls'])):
        old = np.frombuffer(Path(control['output']).read_bytes(),dtype='<f4').reshape(16,1024)
        new = np.load(trial['output'],allow_pickle=False)
        require(new.dtype == np.float32 and new.shape == old.shape == (16,1024)
            and np.isfinite(old).all() and np.isfinite(new).all() and (old >= 0).all() and (new >= 0).all(),
            'All16 production/native values have the same physical FP32 contract')
        delta = new.astype(np.float64)-old.astype(np.float64)
        maxima = np.max(np.abs(delta),axis=1).tolist()
        checks.append(dict(call=call,rank=35,production_call=control['call'],layer_maxabs=maxima,
            all16_maxabs=max(maxima),tolerance=1e-5,within_original_vector_tolerance=max(maxima) <= 1e-5,
            output_byte_exact=new.tobytes() == old.tobytes(),
            prediction_to_prediction_mse=np.mean(delta*delta,axis=1).tolist(),
            native_flops=trial['flops'],production_flops=control['flops'],
            complete_flops_delta=trial['flops']-control['flops'],
            native_operation_count=trial['operation_count'],cross_backend_bill_equality_required=False))
        del old,new,delta
        tick()
    result['production_parity_checks'] = checks
    result['all16_production_vector_parity'] = all(x['within_original_vector_tolerance'] for x in checks)
    require(len(checks) == 2 and result['all16_production_vector_parity'],
            'Original all16 native/client tolerance<=1e-5 for both35 contexts')
    result.update(same_production_backend_qualified=True,target_free_backend_diagnostic_passed=True,
        all_native_outputs_sealed_before_comparison=True,raw_error_measured=False,adjusted_score_measured=False,
        original_mse_ratio_gate_not_evaluated=True,old_fidelity_still_failed=True)


def backend_contract(record,tick=None):
    """Stdlib-only independent supervisor contract; no output/truth decoding."""
    a = record['backend_attestation']
    require(record['candidate_sha256'] == _BP_CANDIDATE_SHA
        and record['ranks'] == [35,35] and record['full_model_calls'] == record['retired_call_count'] == 2
        and record['initialization_flops'] == 1944970298 and len(record['artifacts']) == 14
        and all(record[k] is True for k in ('same_production_backend_qualified',
            'target_free_backend_diagnostic_passed','all16_production_vector_parity',
            'all_native_outputs_sealed_before_comparison','old_fidelity_still_failed',
            'fresh_same_backend_objective_required_before_bank','original_mse_ratio_gate_not_evaluated'))
        and all(record[k] is False for k in ('production_five_qualified','native_fidelity_qualified',
            'truth_accessed','score_measured','raw_error_measured','adjusted_score_measured',
            'automatic_next_stage','uploaded','submission_authorized')),
        'Two-call diagnostic never grants fidelity/score/release authority')
    require(a['site'] == str(_BP_SITE) and a['official_prefix'] == str(OFFPY.parent.parent)
        and a['actual_origins_qualified'] and a['production_blas_qualified']
        and a['numpy_version'] == '2.4.6' and a['flopscope_version'] == '0.12.0'
        and a['whestbench_version'] == '0.16.0' and a['scipy_loaded'] is a['torch_loaded'] is False,
        'Actual imported module and backend attestations')
    require({'numpy','numpy._core._multiarray_umath','numpy.linalg._linalg',
        'numpy.linalg._umath_linalg','flopscope','flopscope.numpy','threadpoolctl',
        'whestbench.domain','whestbench.scoring','whestbench.seeds'} <= set(a['modules']),
        'Every required numerical/official owner actually loaded')
    blas = [p for p in a['threadpools'] if p.get('user_api') == 'blas']
    require(len(blas) == 1 and blas[0]['internal_api'] == 'openblas' and blas[0]['num_threads'] == 1
        and blas[0]['version'] == '0.3.31.188.0' and blas[0]['threading_layer'] == 'pthreads'
        and Path(blas[0]['filepath']).resolve() == (_BP_SITE/'numpy/.dylibs/libscipy_openblas64_.dylib').resolve(),
        'Actual single production BLAS library and thread setting')
    for name,item in a['modules'].items():
        base = _BP_OFFSITE if name.split('.')[0] == 'whestbench' else _BP_SITE
        require(Path(item['path']).resolve().is_relative_to(base.resolve())
            and record['source_bindings'][item['path']] == item['sha256'], 'Complete actual module ownership')
    inputs = {}; selected,seals = backend_controls(inputs,tick)
    require(record['controls'] == selected and record['control_seals'] == seals
        and all(record['input_bindings'].get(p) == d for p,d in inputs.items()),
        'Same actual positive production and negative fidelity boundaries')
    checks = record['production_parity_checks']
    require(len(checks) == 2, 'Both independent native/production comparisons')
    for call,x in enumerate(checks):
        t = record['trials'][call]
        require(x['call'] == t['call'] == call and x['rank'] == t['rank'] == 35
            and x['production_call'] == (0 if call == 0 else 4) and x['tolerance'] == 1e-5
            and len(x['layer_maxabs']) == len(x['prediction_to_prediction_mse']) == 16
            and all(math.isfinite(v) and 0 <= v <= 1e-5 for v in x['layer_maxabs'])
            and all(math.isfinite(v) and v >= 0 for v in x['prediction_to_prediction_mse'])
            and x['all16_maxabs'] == max(x['layer_maxabs']) and x['within_original_vector_tolerance']
            and x['native_flops'] == t['flops'] and x['production_flops'] == selected[call]['flops']
            and x['complete_flops_delta'] == t['flops']-selected[call]['flops']
            and x['native_operation_count'] == t['operation_count']
            and x['cross_backend_bill_equality_required'] is False,
            'Original all16 gate and each separately paid complete cost')
    return True


def derive_source():
    original = _BP_OWNER.read_text()
    if hashlib.sha256(original.encode()).hexdigest() != _BP_OWNER_SHA:
        raise RuntimeError('Immutable original paid native probe owner')
    tree = ast.parse(original)
    node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'controls')
    control_source = ''.join(original.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
    start = original.index('        # The first affected pre-state is layer5.')
    stop = original.index("        result['initialization_flops'] =",start)
    prefix_source = original[start:stop]
    edits = [
        ("HERE/'probe.py'","HERE/'backend_probe.py'"),
        ("HERE/'watch_probe.py'","HERE/'watch_backend_probe.py'"),
        ("HERE/'PROBE_PROTOCOL.md'","HERE/'BACKEND_PROBE_PROTOCOL.md'"),
        ("HERE/'PROBE_WATCH_SPEC.md'","HERE/'BACKEND_PROBE_WATCH_SPEC.md'"),
        ("HERE/'probe_v1'","HERE/'backend_probe_v1'"),
        ("'current_comp_shared_old384_native37_v1'","'current_comp_shared_old384_backend_probe_v1'"),
        ("ROW = STORE/'rows/observer_fit/0037'","ROW = STORE/'rows/observer_fit/0035'"),
        ("'900e3cc6929bcc81ea301479708fe8bdfb642447f6c8bb1a10981bffacc15571'",
         "'5067b577620cb6d5dc97c61b62f10e8fb3f98880b99fe9320f1ef8a8ca567809'"),
        ("'967d989cf0ddf4770f199d50c97b26c70b1129d2d739929e82c8e229b7fd43d4'",
         "'14563b956b4e326aee081263077c0dd1dcb86037a0b7870dc58a5e33865f47f2'"),
        ("    require(all(re.fullmatch('[0-9a-f]{64}',v or '') for v in mapping.values()),",
         "    mapping.update(_BP_PINS)\n    require(all(re.fullmatch('[0-9a-f]{64}',v or '') for v in mapping.values()),"),
        (control_source,"def controls(inputs, tick=None):\n    return backend_controls(inputs,tick)\n"),
        ("    environment(); began = time.monotonic()","    began = time.monotonic(); backend_select(); environment()"),
        ("        result['candidate_sha256'] = hashlib.sha256(text.encode()).hexdigest()",
         "        result['candidate_sha256'] = hashlib.sha256(text.encode()).hexdigest()\n        backend_candidate(result,text,tick)"),
        ("rows = {37:dict(weights=str(ROW/'weights.npy'),mlp_seed=int(row['mlp_seed']))}",
         "rows = {35:dict(weights=str(ROW/'weights.npy'),mlp_seed=int(row['mlp_seed']))}"),
        ("result['ranks'] = [37,37]","result['ranks'] = [35,35]"),
        ("rows=rows,ranks=(37,),result=result","rows=rows,ranks=(35,),result=result"),
        ("threadpool_info=threadpool_info,MLP=MLP,derive_estimator_seed=derive_estimator_seed,",
         "threadpool_info=backend_threadpool_info(threadpool_info,result,tick),MLP=MLP,derive_estimator_seed=derive_estimator_seed,"),
        (prefix_source,"        backend_compare(np,result,tick)\n"),
        ("result['initialization_flops'] >= 1944970298","result['initialization_flops'] == 1944970298")]
    changed,spans = original,[]
    for before,after in edits:
        if changed.count(before) != 1:
            raise RuntimeError('Unique backend metadata splice: '+before[:90])
        pos = changed.index(before); spans.append((pos,before,after))
        changed = changed[:pos]+after+changed[pos+len(before):]
    inverse = changed
    for pos,before,after in reversed(spans):
        if inverse[pos:pos+len(after)] != after:
            raise RuntimeError('Positional backend source inverse')
        inverse = inverse[:pos]+before+inverse[pos+len(after):]
    if inverse != original:
        raise RuntimeError('Whole original probe byte inverse')
    return changed,dict(owner=str(_BP_OWNER),owner_sha256=_BP_OWNER_SHA,whole_source_inverse_verified=True,
        derived_source_sha256=hashlib.sha256(changed.encode()).hexdigest(),splices=len(spans),
        original_paid_loop_and_observers_unchanged=True,estimator_source_unchanged=True,
        old37_parent_prefix_experiment_preserved=True,new_same_candidate_production35_question=True)


_BP_DERIVED, BACKEND_SOURCE_PROOF = derive_source()
exec(compile(_BP_DERIVED,str(_BP_OWNER)+':explicit-production-backend','exec'),globals())
