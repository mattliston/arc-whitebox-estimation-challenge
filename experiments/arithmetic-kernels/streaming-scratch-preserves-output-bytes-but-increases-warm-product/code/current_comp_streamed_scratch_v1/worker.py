"""DRAFT: one new fixed streamed-scratch RPC unit; no numerical import at admission.

The complete qualified compact-scratch context/receipt lifecycle is restored by
a positional inverse. The fixture, geometry, reference and helper bodies below
are NEW; no claim of unchanged old Gram arithmetic or operands is made.
"""
import ast
import hashlib
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parent
OWNER=HERE.parent/'current_comp_compact_scratch_v1/worker.py'
OWNER_SHA='5d961948bc0460b0efc3917cb8695c1a5234ab6f67eb2b3c7ef7f5fe4ab2c087'
OWNER_PINS={
    OWNER:OWNER_SHA,
    HERE/'kernel.py':'4ae8be98f8330deea0228482059a5e193445989582aaf2e6c4e234b353694819',
    HERE/'DESIGN.md':'e8a293e91b9b296b862191dff7747d1a4330f9713e66b95337eb0e881fcce35f',
    HERE/'TIMING_BOUNDARY_V1.md':'cf9d12b26f059763f14ead2e6409fc583da971c798eed174260ee7b1cae165f1',
    OWNER.with_name('client.py'):'1c1c0bc42e589398424af6990d63904bc2b4f206c6c9f7e6309e5dc8ed1259eb',
    OWNER.with_name('watch.py'):'43cc5dbef197d92c4475494a58fd26449dee5ac571c1a577142e70f4e6042088',
    OWNER.with_name('PROTOCOL.md'):'48c4b31c4757dc522e98a804969c130a4e0c77b0fd92ac379b000aa527d61b8f',
    OWNER.with_name('WATCH_SPEC.md'):'e6f4980189a8c9a267599486c8ecd6763831ecbd29a55203fddb1173f5e03c7e'}

HELPER_SOURCE='''def helper_source():
    require(_KERNEL is not None,'Actual kernel SHA admitted first')
    source,canonical_specs,proof=_KERNEL.source_bundle()
    require(_KERNEL.GEOMETRY==(384,384,8192) and _KERNEL.CAPACITY==(512,512,8192)
        and _KERNEL.DEPTH==4 and _KERNEL.WARM_FLOPS==1512549696
        and _KERNEL.REFERENCE_FLOPS==4825546752 and _KERNEL.BASE_BIND_FLOPS==254059520,
        'Only the prospectively fixed physical source geometry and complete tariffs')
    return source,proof
'''

EXERCISE='''def _exercise(fnp,ns,ctx,audit):
    sections,rows,rejected,refs=(audit[k] for k in ('sections','rows','rejected','refs'))
    def own(value):
        if hasattr(value,'shape') and hasattr(value,'dtype'):
            refs.append(weakref.ref(value))
        elif isinstance(value,dict):
            for item in value.values():
                own(item)
        elif isinstance(value,(tuple,list)):
            for item in value:
                own(item)
        return value
    def section(name,function,expected=None):
        before=int(ctx.flops_used)
        audit['active_section']=name
        began=time.perf_counter()
        try:
            value=function()
        except BaseException as error:
            sections.append(dict(name=name,flops=int(ctx.flops_used)-before,
                expected_flops=expected,function_completed=False,error_type=type(error).__name__))
            raise
        elapsed=time.perf_counter()-began
        own(value)
        used=int(ctx.flops_used)-before
        sections.append(dict(name=name,flops=used,expected_flops=expected,
            function_completed=True,tariff_matched=used>=0 and (expected is None or used==expected),
            helper_wall_s=elapsed))
        require(used>=0 and (expected is None or used==expected),'Actual complete section tariff: '+name)
        audit['active_section']=None
        return value
    def finite(value):
        require(bool(fnp.all(fnp.isfinite(value))),'All actual array values finite')
    def serialized(value,shape):
        require(tuple(value.shape)==shape and str(value.dtype)=='float32',
                'Only the declared stored FP32 operand or output')
        snapshot=value.astype(fnp.float32)
        finite(snapshot)
        raw,actual_shape,dtype=snapshot._fetch_data()
        require(tuple(actual_shape)==shape and str(dtype)=='float32'
            and len(raw)==4*math.prod(shape),'Exact serialized stored FP32 payload')
        return raw
    def reference_check(actual,reference):
        widened=actual.astype(fnp.float64)
        finite(widened)
        delta=widened-reference
        norm=fnp.sqrt(fnp.sum(reference*reference))
        denominator=fnp.maximum(norm,fnp.full((1,),1e-30,dtype=fnp.float64))[0]
        relative=fnp.sqrt(fnp.sum(delta*delta))/denominator
        maximum=fnp.max(fnp.abs(delta))
        return dict(reference_norm=float(norm),relative_frobenius=float(relative),maxabs=float(maximum))

    try:
        section('canonical_D4_bindings',lambda:_KERNEL.setup(fnp,ns),254059520)
        def fixture():
            # Two independent draws from one fixed PCG64 stream; BOTH scales
            # sqrt(2/384). All scalar construction and arithmetic is metered.
            scale=fnp.sqrt(fnp.full((1,),2.0,dtype=fnp.float64)/
                           fnp.full((1,),384.0,dtype=fnp.float64))[0]
            rng=fnp.random.default_rng(938248)
            left=rng.normal(0.0,float(scale),(384,384)).astype(fnp.float32)
            right=rng.normal(0.0,float(scale),(384,8192)).astype(fnp.float32)
            del rng,scale
            return left,right
        left,right=section('FNP_rng_and_fixed_operands',fixture)
        raw_left=section('serialize_left',lambda:serialized(left,(384,384)))
        raw_right=section('serialize_right',lambda:serialized(right,(384,8192)))
        outputs,raw_outputs,output_hashes,cache_history,timed_blocks=[],[],[],[],[]
        for index,name in enumerate(ROSTER):
            before_cache=len(ns['_PACKED_FRAME_CACHE'])
            require(before_cache==(0 if index==0 else 1),'Actual canonical first/warm cache state')
            first_ref=len(refs)
            function=(lambda:_KERNEL.canonical_product(fnp,ns,left,right,own)) if name.startswith('canonical') else (
                lambda:_KERNEL.streamed_product(fnp,ns,left,right,own))
            block_before=int(ctx.flops_used)
            block_began=time.perf_counter()
            value=section(name,function,1512549696)
            if name.startswith('stream'):
                require(all(ref() is None or ref() is value for ref in refs[first_ref:]),
                        'Every streamed backing except returned output retired at helper return')
            raw=section(name+'_snapshot',lambda:serialized(value,(384,8192)),9437183)
            digest=hashlib.sha256(raw).hexdigest()
            block_elapsed=time.perf_counter()-block_began
            block_flops=int(ctx.flops_used)-block_before
            require(block_flops==1521986879,'Actual product plus one required paid snapshot')
            after_cache=len(ns['_PACKED_FRAME_CACHE'])
            require(after_cache==1,'Only the canonical frame persists')
            own(ns['_PACKED_FRAME_CACHE'])
            cache_history.append(dict(before=before_cache,after=after_cache,
                streamed_cache_entries=0,streamed_temporaries_retired=name.startswith('stream')))
            outputs.append(value)
            raw_outputs.append(raw)
            output_hashes.append(digest)
            timed_blocks.append(dict(timed_product_snapshot_wall_s=block_elapsed,
                timed_product_snapshot_flops=block_flops))
        def reference_operands():
            a=fnp.frombuffer(raw_left,dtype=fnp.float32).reshape((384,384)).astype(fnp.float64)
            b=fnp.frombuffer(raw_right,dtype=fnp.float32).reshape((384,8192)).astype(fnp.float64)
            finite(a);finite(b)
            return a,b
        a64,b64=section('serialized_operand_FP64_reference_copies',reference_operands)
        reference=section('one_dense_FP64_reference',lambda:a64@b64,4825546752)
        section('reference_finite',lambda:finite(reference))
        for index,(name,value) in enumerate(zip(ROSTER,outputs)):
            diagnostics=section(name+'_reference_check',lambda:reference_check(value,reference))
            rows.append(dict(name=name,dtype='float32',shape=[384,8192],finite=True,
                raw_sha256=output_hashes[index],serialized_bytes=len(raw_outputs[index]),
                product_arithmetic_flops=1512549696,snapshot_flops=9437183,
                **timed_blocks[index],
                frame_cache=cache_history[index],
                input_left_sha256=hashlib.sha256(raw_left).hexdigest(),
                input_right_sha256=hashlib.sha256(raw_right).hexdigest(),**diagnostics))
        require(raw_outputs[0]==raw_outputs[1]==raw_outputs[2]==raw_outputs[3],
                'Actual canonical/streamed byte equality and both helper repeats')
        require(all(r['maxabs']<=5e-5 and r['relative_frobenius']<=2e-5
                    and r['reference_norm']>0 for r in rows),'Fixed stored-operand reference gates')
        own(ns['_STREAM_ARENA']);own(ns['_STREAM_VIEWS']);own(ns['_PACKED_FRAME_CACHE'])
        del outputs,left,right,a64,b64,reference,value,function,raw_outputs,raw,raw_left,raw_right
    finally:
        _KERNEL.clear(ns)
    return rows,sections,rejected,refs
'''

VERIFY='''def verify_call(record,mode,index,tick=None):
    require(mode=='remote' and index==0 and record['schema']==SCHEMA
        and record['mode']==mode and record['call']==index
        and all(record.get(k) is True for k in ('passed','completed','context_retired',
            'all_array_weakrefs_retired','own_complete_bill','fixed_D4_streamed_scratch_only',
            'remote_sections_are_actual_budget_deltas'))
        and record['model_calls']==0 and record['target_accessed'] is False,
        'One genuinely retired paid RPC context, not a model')
    sections=record['sections']
    exact_sections=['canonical_D4_bindings','FNP_rng_and_fixed_operands','serialize_left','serialize_right',
        *(name+suffix for name in ROSTER for suffix in ('','_snapshot')),
        'serialized_operand_FP64_reference_copies','one_dense_FP64_reference','reference_finite',
        *(name+'_reference_check' for name in ROSTER)]
    require([s['name'] for s in sections]==exact_sections
        and len({s['name'] for s in sections})==len(sections)
        and all(s.get('function_completed') and s.get('tariff_matched')
            and type(s['flops']) is int and s['flops']>=0
            and (s['expected_flops'] is None or s['flops']==s['expected_flops']) for s in sections)
        and sum(s['flops'] for s in sections)==record['complete_flops']==record['bill']['flops_used']
        and 0<record['complete_flops']<CALL_BUDGET,'All actual costs, no warm credit against total')
    expected={'canonical_D4_bindings':254059520,'one_dense_FP64_reference':4825546752,
              **{name:1512549696 for name in ROSTER},
              **{name+'_snapshot':9437183 for name in ROSTER}}
    for name,flops in expected.items():
        found=[s for s in sections if s['name']==name]
        require(len(found)==1 and found[0]['flops']==flops,'Fixed setup/core/reference complete tariff')
    rows=record['rows']
    require([r['name'] for r in rows]==list(ROSTER) and not record['rejected_cases'],
            'Exactly four fixed helper calls; no adaptive repeat or extra test')
    for key in ('raw_sha256','input_left_sha256','input_right_sha256'):
        require(len({r[key] for r in rows})==1 and all(re.fullmatch('[0-9a-f]{64}',r[key]) for r in rows),
                'Exact serialized operand/output SHA joins')
    for index,row in enumerate(rows):
        require(row['finite'] and row['dtype']=='float32' and row['shape']==[384,8192]
            and row['serialized_bytes']==12582912 and row['product_arithmetic_flops']==1512549696
            and row['snapshot_flops']==9437183 and row['timed_product_snapshot_flops']==1521986879
            and all(math.isfinite(row[k]) and row[k]>=0 for k in
                    ('maxabs','relative_frobenius','reference_norm','timed_product_snapshot_wall_s'))
            and row['maxabs']<=5e-5 and row['relative_frobenius']<=2e-5
            and row['reference_norm']>0 and row['timed_product_snapshot_wall_s']>0,
            'Actual finite stored-operand reference and direct helper measurement')
        timed_sections=[s for s in sections if s['name'] in (row['name'],row['name']+'_snapshot')]
        require(len(timed_sections)==2
            and sum(s['flops'] for s in timed_sections)==row['timed_product_snapshot_flops']
            and row['timed_product_snapshot_wall_s']>=sum(s['helper_wall_s'] for s in timed_sections),
            'Same product plus required snapshot/fetch/hash boundary for both roles')
        require(row['frame_cache']==dict(before=0 if index==0 else 1,after=1,
                    streamed_cache_entries=0,streamed_temporaries_retired=index in (1,3)),
                'Only canonical first/warm caching; streamed temporaries retire every call')
    bill=record['bill']
    require(all(math.isfinite(bill[k]) and bill[k]>=0 for k in
        ('wall_time_s','residual_wall_time_s','flopscope_backend_time_s','flopscope_overhead_time_s'))
        and bill['wall_time_s']<60. and bill['residual_wall_time_s']<.4,
        'Unchanged actual RPC context60s/residual<.4 gate')
'''

COMPARISON='''def compare(remote,tick=None):
    require(all(remote.get(k) is True for k in ('passed','completed','own_complete_bill',
        'one_context_completed','source_hashes_unchanged','helper_tariffs_exact','byte_exact',
        'helper_repeats_exact','fixed_D4_streamed_scratch_only')) and remote['retired_contexts']==1,
        'Fresh complete one-context worker authority')
    record=call_records(remote,tick)[0]
    rows=record['rows']
    warm=rows[3]['timed_product_snapshot_wall_s']/rows[2]['timed_product_snapshot_wall_s']
    first=rows[1]['timed_product_snapshot_wall_s']/rows[0]['timed_product_snapshot_wall_s']
    require(all(math.isfinite(r) and r>0 for r in (warm,first)),'Two actual fixed component ratios')
    return dict(passed=True,one_context_completed=True,all_independent_references_passed=True,
        helper_tariffs_exact=True,byte_exact=True,helper_repeats_exact=True,
        complete_flops=record['complete_flops'],product_arithmetic_flops=1512549696,
        snapshot_flops=9437183,timed_product_snapshot_flops=1521986879,
        timing_scope='product plus required FP32 copy, finite scan, fetch and SHA256',
        warm_wall_ratio=warm,first_wall_ratio=first,warm_gate_passed=warm<=.85,
        first_gate_passed=first<=.90,component_gate_passed=warm<=.85 and first<=.90,
        timing_condition='stream_warm/canonical_warm <= 0.85 AND stream_first/canonical_first <= 0.90',
        runtime_gain_measured=True,full_estimator_fidelity_established=False,
        full_45s_gate_established=False,raw_value_measured=False,model_calls=0,
        truth_accessed=False,automatic_next_stage=False)
'''


def derive_source():
    for path,digest in OWNER_PINS.items():
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Exact qualified compact-unit lifecycle owner: '+str(path))
    spec=importlib.util.spec_from_file_location('_streamed_compact_unit_owner',OWNER)
    base=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    original,parent_proof=base.derive_source()
    source,positions=original,[]
    def change(before,after,count=1):
        nonlocal source
        if source.count(before)!=count:
            raise RuntimeError('Exact streamed-unit source splice count: '+before[:100])
        for unused in range(count):
            at=source.index(before);positions.append((at,before,after))
            source=source[:at]+after+source[at+len(before):]
    for name,replacement in (('helper_source',HELPER_SOURCE),('_exercise',EXERCISE),
                              ('verify_call',VERIFY),('compare',COMPARISON)):
        nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name]
        if len(nodes)!=1:
            raise RuntimeError('One original unit function: '+name)
        change(ast.get_source_segment(source,nodes[0]),replacement.rstrip())
    change("SCHEMA='current_comp_compact_scratch_unit_v1'","SCHEMA='current_comp_streamed_scratch_unit_v1'")
    change('CALL_BUDGET=10_000_000_000','CALL_BUDGET=14_000_000_000')
    change("ROSTER=('serial_first','compact_first','serial_warm','compact_warm')",
           "ROSTER=('canonical_first','stream_first','canonical_warm','stream_warm')")
    change('fixed_D5_Gram_only','fixed_D4_streamed_scratch_only',2)
    change("namespace=\\'root-batch-gram-unit\\'","namespace=\\'streamed-scratch-unit\\'")
    inverse=source
    for at,before,after in reversed(positions):
        if inverse[at:at+len(after)]!=after:
            raise RuntimeError('Exact positional streamed-unit inverse')
        inverse=inverse[:at]+before+inverse[at+len(after):]
    if inverse!=original:
        raise RuntimeError('Entire qualified compact-unit lifecycle source restored')
    return source,dict(parent=parent_proof,owner_sha256=OWNER_SHA,
        whole_compact_effective_worker_inverse=True,qualified_RPC_lifecycle_retained=True,
        original_Gram_fixture_or_arithmetic_claimed=False,new_fixed_operands=True,
        geometry=[384,384,8192],capacity=[512,512,8192],depth=4,
        binding_flops=254059520,helper_flops=1512549696,reference_flops=4825546752,
        snapshot_flops=9437183,timed_product_snapshot_flops=1521986879,
        timing_boundary_sha256=OWNER_PINS[HERE/'TIMING_BOUNDARY_V1.md'],
        fixed_subtotal_flops=11129805056,hard_context_flops=14000000000,
        prospective_warm_ratio_limit=.85,prospective_first_ratio_limit=.90,
        derived_sha256=hashlib.sha256(source.encode()).hexdigest())


_source,STREAM_SOURCE_PROOF=derive_source()
_api=dict(__file__=str(Path(__file__).resolve()),__name__='_streamed_scratch_worker_inert')
exec(compile(_source,str(Path(__file__).resolve())+':whole-qualified-lifecycle-inverse','exec'),_api)
_api['PINS'].update(OWNER_PINS)
for _name in ('main','admit','owner_map','helper_source','primitive_namespace','derive_main','PINS',
              'numerical_modules','sha','read','write','verify_call','call_records','compare',
              'SCHEMA','CALL_BUDGET','ROSTER'):
    globals()[_name]=_api[_name]
del _source


if __name__=='__main__':
    raise SystemExit(main())
