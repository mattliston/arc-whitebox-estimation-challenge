"""Source fragments for ONE target-free D5 two-RHS real-RPC unit.

No numerical packages, contexts, arrays or files are opened at import.
These reviewed function bodies are installed into the qualified lifecycle.
"""


def owner_map(args):
    global _KERNEL
    _KERNEL=_load(HERE/'helper.py',args.kernel_sha256,'w5_fixed_helper_owner')
    return {**PINS,**_KERNEL.PINS,HERE/'worker.py':args.source_sha256,
        HERE/'client.py':args.client_sha256,HERE/'helper.py':args.kernel_sha256,
        HERE/'PROTOCOL.md':args.protocol_sha256,HERE/'watch.py':args.watcher_sha256,
        HERE/'WATCH_SPEC.md':args.watch_spec_sha256,HERE/'unit_arithmetic.py':UNIT_SOURCE_SHA}


def helper_source():
    require(_KERNEL is not None,'Actual helper SHA admitted first')
    source,proof=_KERNEL.source_bundle()
    require(_KERNEL.PRODUCT_FLOPS==(1160608064,458060764) and
        _KERNEL.REUSE_PRODUCT_FLOPS==(1177280608,421528380) and
        _KERNEL.SETUP_FLOPS==288218624,'One source-priced fixed D5 pair')
    return source,proof


def primitive_namespace(fnp):
    class CountingFNP:
        def __init__(self,backend):
            self.backend=backend
            self.calls=[]
        def __getattr__(self,name):
            function=getattr(self.backend,name)
            if name not in ('add','subtract','stack','copyto','matmul'):
                return function
            def call(*args,**kwargs):
                # A completed primitive record contains no array owners.
                result=function(*args,**kwargs)
                self.calls.append(name)
                return result
            return call
    facade=CountingFNP(fnp)
    ns,proof=_KERNEL.namespace(facade)
    ns['_COUNT_FNP']=facade
    return ns,proof


def _exercise(fnp,ns,ctx,audit):
    sections,rows,rejected,refs=(audit[k] for k in ('sections','rows','rejected','refs'))
    facade=ns['_COUNT_FNP']
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
        require(used>=0 and (expected is None or used==expected),'Exact complete section tariff: '+name)
        audit['active_section']=None
        return value
    def serialized(value,shape):
        require(tuple(value.shape)==shape and str(value.dtype)=='float32','Exact FP32 stored array')
        snapshot=value.astype(fnp.float32)
        require(bool(fnp.all(fnp.isfinite(snapshot))),'All values finite')
        raw,actual_shape,dtype=snapshot._fetch_data()
        require(tuple(actual_shape)==shape and str(dtype)=='float32' and
            len(raw)==4*math.prod(shape),'Full stored FP32 snapshot/fetch')
        return raw
    def reject(name,fn):
        before=int(ctx.flops_used)
        before_calls=len(facade.calls)
        try:
            fn()
        except RuntimeError as error:
            message=str(error)
            error.__traceback__=None
        else:
            raise RuntimeError('Required metadata rejection missing: '+name)
        require(int(ctx.flops_used)==before and len(facade.calls)==before_calls,
                'Rejected metadata performed no arithmetic')
        rejected.append(dict(name=name,flops=0,arithmetic_calls=0,message=message))
    cache=None
    try:
        arena,views=section('original_and_compact_bindings',lambda:_KERNEL.setup(facade,ns),288218624)
        def fixture():
            scale=fnp.sqrt(fnp.full((1,),2.0,dtype=fnp.float64)/
                fnp.full((1,),992.0,dtype=fnp.float64))[0]
            rng=fnp.random.default_rng(938248)
            values=tuple(rng.normal(0.0,float(scale),shape).astype(fnp.float32)
                for shape in ((1024,992),(992,1024),(992,384)))
            del rng,scale
            return values
        left,right0,right1=section('FNP_rng_and_fixed_operands',fixture)
        inputs=(left,right0,right1)
        input_raw=[]
        for name,value,shape in zip(('left','right0','right1'),inputs,
                                   ((1024,992),(992,1024),(992,384))):
            input_raw.append(section('serialize_'+name,lambda:serialized(value,shape),3*math.prod(shape)-1))
        del value
        token,previous=object(),object()
        cache=_KERNEL.begin_layer(token,5,left,previous)
        reject('unknown_role',lambda:_KERNEL.product_d5(left,right0,left,views,cache,
               role='unknown',event=lambda entry:None))
        reject('wrong_layer',lambda:_KERNEL.begin_layer(token,4,left,previous))
        outputs,raw_outputs,events=[],[],[]
        for index,name in enumerate(ROSTER):
            rhs=right0 if index%2==0 else right1
            shape=(1024,int(rhs.shape[1]))
            expected=(1160608064,458060764,1177280608,421528380)[index]
            call_start=len(facade.calls)
            before=int(ctx.flops_used)
            began=time.perf_counter()
            def product():
                result=fnp.empty(shape,dtype=fnp.float32)
                if index<2:
                    return ns['_packed_product_cached'](left,rhs,result,views,5)
                return _KERNEL.product_d5(left,rhs,result,views,cache,
                    role=('covariance_outer','shared_basis_push')[index-2],event=events.append)
            value=section(name,product,expected)
            outputs.append(value)
            primitive_names=facade.calls[call_start:]
            require(len(primitive_names)==(91,91,92,61)[index] and primitive_names.count('matmul')==1,
                    'Actual 182 to153 complete product primitive calls')
            raw=section(name+'_snapshot',lambda:serialized(value,shape),3*math.prod(shape)-1)
            digest=hashlib.sha256(raw).hexdigest()
            elapsed=time.perf_counter()-began
            raw_outputs.append(raw)
            row=dict(name=name,shape=list(shape),dtype='float32',finite=True,
                raw_sha256=digest,serialized_bytes=len(raw),product_flops=expected,
                snapshot_flops=3*math.prod(shape)-1,
                timed_product_snapshot_flops=int(ctx.flops_used)-before,
                timed_product_snapshot_wall_s=elapsed,primitive_calls=primitive_names,
                input_left_sha256=hashlib.sha256(input_raw[0]).hexdigest(),
                input_right_sha256=hashlib.sha256(input_raw[1+index%2]).hexdigest(),
                frame_count=len(ns['_PACKED_FRAME_CACHE']),
                compact_frame_count=len(ns['_CS_CAPACITY_VIEWS']))
            rows.append(row)
            if index==2:
                foreign=_KERNEL.begin_layer(object(),5,left,object())
                reject('foreign_generation',lambda:_KERNEL.product_d5(left,right0,outputs[0],
                    views,foreign,role='covariance_outer',event=events.append))
                _KERNEL.close_layer(foreign)
                del foreign
        require(raw_outputs[0]==raw_outputs[2] and raw_outputs[1]==raw_outputs[3],
                'Both distinct RHS outputs exactly equal the actual original source')
        require([e['kind'] for e in events]==['creator','hit'] and
            [e['copied_elements'] for e in events]==[16672544,0] and
            [e['skipped_left_arithmetic_calls'] for e in events]==[0,30] and
            all(e['generation']==id(token) and e['left_owner']==id(left) and
                e['source_owner']==id(left) and e['previous_owner']==id(previous) and
                e['layer']==5 and e['depth']==5 and e['capacity_bytes']==68841472 and
                e['leaf_shape']==[7,7,7,7,7,32,31] for e in events),'Actual fixed cache provenance')
        refs.extend(cache.array_refs)
        _KERNEL.close_layer(cache)
        gc.collect()
        require(all(ref() is None for ref in cache.array_refs),'Copied cache owners retired')
        reject('use_after_close',lambda:_KERNEL.product_d5(left,right1,outputs[-1],views,cache,
            role='shared_basis_push',event=events.append))
        rows[-1]['cache_events']=events
        rows[-1]['cache_retired']=True
        refs.extend(ns['_W5_ARRAY_REFS'])
        own(ns['_PACKED_FRAME_CACHE'])
        del outputs,inputs,left,right0,right1,rhs,value,product,raw_outputs,raw,input_raw
        del arena,views
    finally:
        if cache is not None:
            _KERNEL.close_layer(cache)
        _KERNEL.clear(ns)
    return rows,sections,rejected,refs


def verify_call(record,mode,index,tick=None):
    require(mode=='remote' and index==0 and record['schema']==SCHEMA and
        record['mode']==mode and record['call']==index and
        all(record.get(k) is True for k in ('passed','completed','context_retired',
            'all_array_weakrefs_retired','own_complete_bill','fixed_D5_weight_pair_only',
            'remote_sections_are_actual_budget_deltas')) and
        record['model_calls']==0 and record['target_accessed'] is False,'One real retired unit context')
    sections=record['sections']
    expected_names=['original_and_compact_bindings','FNP_rng_and_fixed_operands',
        'serialize_left','serialize_right0','serialize_right1',
        *(name+suffix for name in ROSTER for suffix in ('','_snapshot'))]
    require([s['name'] for s in sections]==expected_names and
        all(s['function_completed'] and s['tariff_matched'] and
            (s['expected_flops'] is None or s['flops']==s['expected_flops']) for s in sections) and
        sum(s['flops'] for s in sections)==record['complete_flops']==record['bill']['flops_used'] and
        0<record['complete_flops']<CALL_BUDGET,'Actual complete bill with all sections')
    fixed=sum(s['flops'] for s in sections if s['name']!='FNP_rng_and_fixed_operands')
    require(fixed==3521584817,'All priced setup/products/copy/snapshots, plus actual fixture')
    rows=record['rows']
    require([r['name'] for r in rows]==list(ROSTER) and
        [r['name'] for r in record['rejected_cases']]==
        ['unknown_role','wrong_layer','foreign_generation','use_after_close'] and
        all(r['flops']==0 and r['arithmetic_calls']==0 for r in record['rejected_cases']),
        'Fixed calls and zero-F metadata rejections')
    for j,r in enumerate(rows):
        shape=[1024,1024 if j%2==0 else 384]
        expected=(1160608064,458060764,1177280608,421528380)[j]
        require(r['finite'] and r['dtype']=='float32' and r['shape']==shape and
            r['serialized_bytes']==4*math.prod(shape) and r['product_flops']==expected and
            r['snapshot_flops']==3*math.prod(shape)-1 and
            r['timed_product_snapshot_flops']==expected+r['snapshot_flops'] and
            len(r['primitive_calls'])==(91,91,92,61)[j] and
            r['primitive_calls'].count('matmul')==1 and
            r['frame_count']==2 and r['compact_frame_count']==1 and
            math.isfinite(r['timed_product_snapshot_wall_s']) and
            r['timed_product_snapshot_wall_s']>0,'Actual product/snapshot/call evidence')
        pair=rows[j%2]
        require(all(r[k]==pair[k] and re.fullmatch('[0-9a-f]{64}',r[k])
            for k in ('raw_sha256','input_left_sha256','input_right_sha256')),'Own bytes joined across arms')
    require(rows[-1]['cache_retired'] and [e['kind'] for e in rows[-1]['cache_events']]==['creator','hit'],
            'Cache ownership retired before closed bill')
    bill=record['bill']
    require(all(math.isfinite(bill[k]) and bill[k]>=0 for k in
        ('wall_time_s','residual_wall_time_s','flopscope_backend_time_s','flopscope_overhead_time_s')) and
        bill['wall_time_s']<60 and bill['residual_wall_time_s']<.4,'Actual whole context and residual gates')


def compare(remote,tick=None):
    require(all(remote.get(k) is True for k in ('passed','completed','own_complete_bill',
        'one_context_completed','source_hashes_unchanged','helper_tariffs_exact','byte_exact',
        'two_rhs_parity_exact','fixed_D5_weight_pair_only')) and remote['retired_contexts']==1,
        'One complete current unit authority')
    record=call_records(remote,tick)[0]
    rows=record['rows']
    old=sum(r['timed_product_snapshot_wall_s'] for r in rows[:2])
    new=sum(r['timed_product_snapshot_wall_s'] for r in rows[2:])
    ratio=new/old
    require(math.isfinite(ratio) and ratio>0,'Actual descriptive two-arm timing')
    return dict(passed=True,one_context_completed=True,original_source_reference_parity=True,
        helper_tariffs_exact=True,byte_exact=True,two_rhs_parity_exact=True,
        complete_flops=record['complete_flops'],net_product_flops_saved=19859840,
        original_pair_flops=1618668828,reused_pair_flops=1598808988,
        primitive_calls_before=182,primitive_calls_after=153,
        pair_wall_ratio=ratio,component_gate_passed=True,
        timing_condition='Descriptive only; exact source/bytes/cost qualification',
        runtime_gain_measured=False,helper_timings_measured=True,
        full_estimator_fidelity_established=False,full_45s_gate_established=False,
        raw_value_measured=False,model_calls=0,truth_accessed=False,automatic_next_stage=False)
