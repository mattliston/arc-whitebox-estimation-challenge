"""DRAFT: one paid, target-free real-RPC lazy-factor algebra context.

Only stdlib source admission at import. The original shared helpers and exact
lazy fragment run on the original public dense branch, never a card or arena.
The separately pinned client and independent watcher own any future --run.
"""
import ast
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import weakref

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
OWNER = NEW/'current_comp_shared_pair256_v1/tiny_worker.py'
OWNER_SHA = 'c6afc327c9acb6a9cd6ba3ed66ef293f8a05d9a22e9031b97ffe9cec49d42e1d'
PARENT = NEW/'current_comp_push_frontier_boxes_v1/rpc37_v1/estimator.py'
PARENT_SHA = '378092f76102b83ddcff445adaf27309ae8136d1df6844599dad59b69cd75dfa'
FRAGMENT_SHA = 'e96ee6b660a688f767abd2b8300883e52920eace477e35e7b5edaae759ac6e92'
SCHEMA = 'current_comp_shared_lazy_rotation_tiny_v1'
CALL_BUDGET = 10_000_000_000
RANK, ROWS, RESPONSE, LIVE, PREFIX, JOIN = 384, 1024, 32, 64, 64, 32
REJECTIONS = ('repeat_lazy', 'missed_consolidation', 'columns', 'stale_F',
              'dtype', 'nonfinite_prefix', 'nonfinite_map', 'overflow',
              'huge_map_tiny_prefix', 'tiny_map_huge_prefix')


def _owner():
    raw = OWNER.read_bytes()
    if hashlib.sha256(raw).hexdigest() != OWNER_SHA:
        raise RuntimeError('Exact admitted primitive environment/paid-loop owner')
    ns = dict(__file__=str(OWNER), __name__='_lazy_tiny_qualified_owner')
    exec(compile(raw, str(OWNER), 'exec'), ns)
    return ns


_BASE = _owner()
require, sha, read, write = (_BASE[k] for k in ('require', 'sha', 'read', 'write'))
numerical_modules = _BASE['numerical_modules']
PINS = {p:d for p,d in _BASE['PINS'].items()
        if p.is_relative_to(_BASE['OSITE']) or p.is_relative_to(_BASE['CSITE'])}
PINS.update({OWNER:OWNER_SHA, PARENT:PARENT_SHA, HERE/'fnp.py':FRAGMENT_SHA,
    HERE/'compose.py':'020e7ad0c82227547268017e5402a44720e546e222a87845ac5e06ccc2f677d6',
    HERE/'price.py':'9b11906d79cdf4e79d3700e1f79e1a6c1d34d43e2277bbab0a737b10ced8572d',
    OWNER.with_name('tiny_client.py'):'a8876ee4dffab18e5c468c90a91ef7c96289264ecf5014d95b2feea7cdf6ca10'})


def owner_map(args):
    return {**PINS, HERE/'tiny_worker.py':args.source_sha256,
        HERE/'tiny_client.py':args.client_sha256, HERE/'TINY_PROTOCOL.md':args.protocol_sha256,
        HERE/'watch_tiny.py':args.watcher_sha256, HERE/'TINY_WATCH_SPEC.md':args.watch_spec_sha256,
        HERE/'PROTOCOL.md':args.decision_sha256}


def admit(args, guard=None):
    require(not numerical_modules(), 'No numerical imports before source admission')
    owners = owner_map(args)
    for path,digest in owners.items():
        require(re.fullmatch('[0-9a-f]{64}',digest or '') and sha(path,guard)==digest,
                'Exact lazy unit owner: '+str(path))
    return {str(p):d for p,d in owners.items()}


def helper_source():
    raw, fragment = PARENT.read_bytes(), (HERE/'fnp.py').read_bytes()
    require(hashlib.sha256(raw).hexdigest()==PARENT_SHA
            and hashlib.sha256(fragment).hexdigest()==FRAGMENT_SHA, 'Exact primitive owners')
    tree = ast.parse(raw.decode())
    names = ('_mm','_shared_old384_mm','_shared_old384_finite')
    chunks = []
    for name in names:
        nodes = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name]
        require(len(nodes)==(4 if name=='_mm' else 1), 'Exact last-effective helper owner')
        chunks.append(_BASE['_node_bytes'](raw,nodes[-1]))
    require(chunks[0].startswith(b"def _mm(a, b):\n    if not _WINOGRAD or str(a.dtype) != 'float32' or str(b.dtype) != 'float32':\n        return a @ b\n"),
            'Unchanged original public dense branch')
    text = b'\n\n'.join(chunks)+b'\n\n'+fragment
    proof = dict(parent_sha256=PARENT_SHA, fragment_sha256=FRAGMENT_SHA,
        extracted_source_sha256=hashlib.sha256(text).hexdigest(),
        original_helper_names=list(names), original_helper_bytes_unchanged=True,
        exact_fragment_bytes=True, dense_branch_only=True, actual_rank=384,
        actual_full_reconstruction_rows=1024, packed_kernel_qualified=False,
        deployment_binding_qualified=False, estimator_executed=False,
        model_calls=0, heads_called=False, truth_accessed=False)
    return text.decode(), proof


def primitive_namespace(fnp):
    source, proof = helper_source()
    ns = dict(fnp=fnp, _WINOGRAD=False)
    exec(compile(source, 'lazy-tiny:exact-shared-helpers+dense-branch', 'exec'), ns)
    return ns, proof


def dense(m,k,n):
    return m*n*(2*k-1)


def finite_price(elements):
    return 2*elements-1


def join_price(kind, old, joined, split=None):
    r, h = RANK, finite_price
    if kind=='baseline':
        return dense(r,r,old)+r*(old+joined)+h(r*(old+joined))
    if kind=='lazy':
        return h(r*old)+h(r*joined)+h(r*r)+h(r*old)+h(r*r)+20
    require(kind=='consolidate' and split and sum(split)==old, 'Exact two-group consolidation')
    a,b = split
    return (dense(r,r,r)+h(r*r)+dense(r,r,a)+dense(r,r,b)
            +r*(old+joined)+h(r*(old+joined)))


def consumer_price(columns, split=None):
    n,r,s,v,h = ROWS,RANK,RESPONSE,LIVE,finite_price
    common = h(n*columns)+r*columns+s*columns+dense(s,r,v)
    if split is None:
        return common+dense(n,r,columns)+dense(s,columns,r)
    a,b = split
    require(a+b==columns, 'Exact complete split')
    return (common+dense(n,r,r)+h(n*r)+dense(n,r,a)+dense(n,r,b)+n*columns
            +dense(s,a,r)+dense(s,b,r)+dense(s,r,r)+h(s*r)+s*r)


def _exercise(fnp, ns, ctx):
    """All fixtures, references, diagnostics and rejection arithmetic are paid."""
    sections, rows, rejected, refs = [], [], [], []
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
    def section(name, function, expected=None):
        before = int(ctx.flops_used)
        value = own(function())
        used = int(ctx.flops_used)-before
        require(used>=0 and (expected is None or used==expected), 'Exact actual section tariff: '+name)
        sections.append(dict(name=name,flops=used,expected_flops=expected))
        return value
    def scalar(value, dtype=None):
        return fnp.full((1,1),value,dtype=fnp.float32 if dtype is None else dtype)
    def grid(m,n,a,b,c,modulus,denominator):
        i = fnp.arange(m,dtype=fnp.int32)[:,None]
        j = fnp.arange(n,dtype=fnp.int32)[None,:]
        return (((a*i+b*j+c)%modulus)-(modulus//2)).astype(fnp.float32)/scalar(denominator)
    def coefficients(count):
        positive = fnp.arange(1,count//2+1,dtype=fnp.int32).astype(fnp.float32)/scalar(64)[0]
        return fnp.concatenate([positive,-positive])
    def factors(columns,phase):
        i = fnp.arange(RANK,dtype=fnp.int32)[:,None]
        j = fnp.arange(columns//2,dtype=fnp.int32)[None,:]
        first = (scalar(.125)+(i%17).astype(fnp.float32)/scalar(512)
            +(j+1).astype(fnp.float32)/scalar(1024)
            +((i+3*j+phase)%13).astype(fnp.float32)/scalar(2048))
        return fnp.concatenate([first,first*scalar(.5)],axis=1)
    def fixtures():
        i = fnp.arange(RANK,dtype=fnp.int32)
        diagonal = (i[:,None]==i[None,:]).astype(fnp.float32)
        cycle = (((i+1)%RANK)[:,None]==i[None,:]).astype(fnp.float32)
        swap = ((i+fnp.where(i%2==0,1,-1))[:,None]==i[None,:]).astype(fnp.float32)
        return dict(L=diagonal*scalar(.5)+cycle*scalar(.125),
            N=diagonal*scalar(.375)+swap*scalar(.0625),
            U=grid(ROWS,RANK,5,11,1,29,128)+scalar(.25),
            V=grid(ROWS,RANK,7,13,2,31,128)+scalar(.25),
            F=factors(PREFIX,3),J=factors(JOIN,7),K=factors(JOIN,11),
            a=coefficients(PREFIX), b=coefficients(JOIN), c=coefficients(JOIN))
    values = section('fixtures',fixtures)
    reference_values = section('reference_casts',lambda:{
        key:value.astype(fnp.float64) for key,value in values.items()})
    original_mm = ns['_shared_old384_mm']
    def tracked_mm(left,right):
        return own(original_mm(left,right))
    ns['_shared_old384_mm'] = tracked_mm
    def state(factors):
        return dict(F=factors,columns=int(factors.shape[1]),lazy=None)
    def baseline_join(current,overlap,joining):
        factors = fnp.concatenate([ns['_shared_old384_mm'](overlap,current['F']),joining],axis=1)
        current['columns'] += int(joining.shape[1])
        current['F'] = ns['_shared_old384_finite'](factors,'tiny baseline factors')
        return current
    def lazy_join(current,layer,overlap,joining):
        factors = ns['_slr_join_factors'](layer,overlap,current,joining)
        current['columns'] += int(joining.shape[1])
        ns['_slr_finish_factors'](current,factors)
        return current
    def consume(current,U,lam):
        X = ns['_shared_old384_finite'](ns['_slr_reconstruct'](U,current),'tiny all-row reconstruction')
        inner = ns['_slr_inner'](X[:RESPONSE],lam,current)
        plane = ns['_shared_old384_mm'](inner,U[:LIVE].T)
        return X,plane
    def direct(F,U,lam):
        X = U@F
        plane = ((X[:RESPONSE]*X[:RESPONSE])*lam[None,:])@X[:LIVE].T
        return X,plane
    def compare(name,actual,reference):
        a = actual.astype(fnp.float64)
        difference = a-reference
        error = fnp.abs(difference)
        max_error = fnp.max(error,keepdims=True)
        reference_max = fnp.max(fnp.abs(reference),keepdims=True)
        reference_norm = fnp.sqrt(fnp.sum(reference*reference,keepdims=True))
        relative = fnp.sqrt(fnp.sum(difference*difference,keepdims=True))/fnp.maximum(
            reference_norm,scalar(2**-20,fnp.float64))
        allowed = scalar(2e-5,fnp.float64)+scalar(5e-5,fnp.float64)*reference_max
        require(bool(fnp.all(fnp.isfinite(a))) and bool(fnp.all(fnp.isfinite(reference)))
            and bool(fnp.all(reference_norm>=scalar(2**-12,fnp.float64)))
            and bool(fnp.all(max_error<=allowed))
            and bool(fnp.all(relative<=scalar(2e-4,fnp.float64))),
            'Fixed conditioned FP64 direct-reference tolerance: '+name)
        return dict(name=name,maxabs=float(fnp.sum(max_error)),
                    allowed_maxabs=float(fnp.sum(allowed)),
                    reference_maxabs=float(fnp.sum(reference_max)),
                    relative_frobenius=float(fnp.sum(relative)),
                    reference_norm=float(fnp.sum(reference_norm)),passed=True)
    try:
        for initial,next_layer in ((8,9),(14,15)):
            baseline,lazy = state(values['F']),state(values['F'])
            lam = section(f'L{initial}_coefficients',lambda:fnp.concatenate([values['a'],values['b']]))
            ref_lam = section(f'L{initial}_reference_coefficients',lambda:lam.astype(fnp.float64))
            section(f'L{initial}_baseline_join',lambda:baseline_join(baseline,values['L'],values['J']),
                    join_price('baseline',PREFIX,JOIN))
            section(f'L{initial}_lazy_join',lambda:lazy_join(lazy,initial,values['L'],values['J']),
                    join_price('lazy',PREFIX,JOIN))
            ref_F = section(f'L{initial}_reference_join',lambda:fnp.concatenate(
                [reference_values['L']@reference_values['F'],reference_values['J']],axis=1))
            for layer,Ukey in ((initial,'U'),(next_layer,'V')):
                if layer==9:
                    section('L9_baseline_join',lambda:baseline_join(baseline,values['N'],values['K']),
                            join_price('baseline',PREFIX+JOIN,JOIN))
                    section('L9_consolidate',lambda:lazy_join(lazy,9,values['N'],values['K']),
                            join_price('consolidate',PREFIX+JOIN,JOIN,(PREFIX,JOIN)))
                    ref_F = section('L9_reference_join',lambda:fnp.concatenate(
                        [reference_values['N']@ref_F,reference_values['K']],axis=1))
                    lam = section('L9_coefficients',lambda:fnp.concatenate([lam,values['c']]))
                    ref_lam = section('L9_reference_coefficients',lambda:lam.astype(fnp.float64))
                    for label,current in (('baseline',baseline),('lazy',lazy)):
                        rows.append(section('L9_'+label+'_factor_check',
                            lambda:compare('L9 '+label+' factors',current['F'],ref_F)))
                columns = int(baseline['columns'])
                split = None if lazy['lazy'] is None else (PREFIX,JOIN)
                left = section(f'L{layer}_baseline_consumer',lambda:consume(baseline,values[Ukey],lam),
                               consumer_price(columns))
                right = section(f'L{layer}_lazy_consumer',lambda:consume(lazy,values[Ukey],lam),
                                consumer_price(columns,split))
                reference = section(f'L{layer}_direct_reference',lambda:direct(
                    ref_F,reference_values[Ukey],ref_lam))
                for label,actual in (('baseline',left),('lazy',right)):
                    for index,part in enumerate(('all_rows','signed_plane')):
                        rows.append(section(f'L{layer}_{label}_{part}_check',
                            lambda:compare(f'L{layer} {label} {part}',actual[index],reference[index])))
                del left,right,reference
            del baseline,lazy,lam,ref_lam,ref_F

        def valid_lazy():
            return dict(F=None,columns=PREFIX+JOIN,lazy=dict(layer=8,prefix=PREFIX,joined=JOIN,
                        base=values['F'],join=values['J'],map=values['L']))
        for name in REJECTIONS:
            def reject():
                current = valid_lazy()
                expected = {'repeat_lazy':'no accumulation','missed_consolidation':'immediate next-join',
                    'columns':'ordered factor groups','stale_F':'one lazy owner','dtype':'FP32 shape',
                    'nonfinite_prefix':'nonfinite or non-FP32','nonfinite_map':'nonfinite or non-FP32',
                    'overflow':'virtual rotation overflow envelope',
                    'huge_map_tiny_prefix':'virtual rotation overflow envelope',
                    'tiny_map_huge_prefix':'virtual rotation overflow envelope'}[name]
                try:
                    if name=='repeat_lazy':
                        ns['_slr_join_factors'](10,values['N'],current,values['K'])
                    elif name=='missed_consolidation':
                        ns['_slr_join_factors'](11,values['N'],current,values['K'])
                    else:
                        if name=='columns':
                            current['lazy']['joined'] += 2
                        elif name=='stale_F':
                            current['F'] = values['F']
                        elif name=='dtype':
                            current['lazy']['base'] = own(values['F'].astype(fnp.float64))
                        elif name=='nonfinite_prefix':
                            current['lazy']['base'] = own(fnp.full((RANK,PREFIX),float('nan'),dtype=fnp.float32))
                        elif name=='nonfinite_map':
                            current['lazy']['map'] = own(fnp.full((RANK,RANK),float('inf'),dtype=fnp.float32))
                        elif name=='overflow':
                            current['lazy']['map'] = own(fnp.full((RANK,RANK),float(2**70),dtype=fnp.float32))
                            current['lazy']['base'] = own(fnp.full((RANK,PREFIX),2**20,dtype=fnp.float32))
                        elif name=='huge_map_tiny_prefix':
                            current['lazy']['map'] = own(fnp.full((RANK,RANK),float(2**90),dtype=fnp.float32))
                            current['lazy']['base'] = own(fnp.full((RANK,PREFIX),2**-30,dtype=fnp.float32))
                        elif name=='tiny_map_huge_prefix':
                            current['lazy']['map'] = own(fnp.full((RANK,RANK),2**-30,dtype=fnp.float32))
                            current['lazy']['base'] = own(fnp.full((RANK,PREFIX),float(2**90),dtype=fnp.float32))
                        if name in ('columns','stale_F','dtype'):
                            ns['_slr_layout'](current,current['columns'])
                        else:
                            ns['_slr_finish_factors'](current,None)
                except RuntimeError as error:
                    require(expected in str(error), 'Only the prescribed rejection: '+name)
                    return dict(name=name,expected_error=expected,error=str(error),passed=True)
                raise RuntimeError('Expected rejection did not happen: '+name)
            rejected.append(section('reject_'+name,reject))
        high = section('safe_envelope_inputs',lambda:(
            fnp.full((RANK,RANK),2**35,dtype=fnp.float32),
            fnp.full((RANK,PREFIX),2**36,dtype=fnp.float32)))
        section('safe_envelope',lambda:ns['_slr_virtual_rotation_envelope'](*high),
                finite_price(RANK*RANK)+finite_price(RANK*PREFIX)+20)
        del high
    finally:
        ns['_shared_old384_mm'] = original_mm
    return rows,sections,rejected,refs


def arithmetic(flops,fnp,ns,mode,index,guard):
    require(mode=='remote' and index==0, 'ONE actual RPC unit context')
    with flops.BudgetContext(flop_budget=CALL_BUDGET,quiet=True,namespace='lazy-rotation-tiny') as ctx:
        rows,sections,rejected,refs = _exercise(fnp,ns,ctx)
        gc.collect()
        require(all(ref() is None for ref in refs), 'All tracked numerical owners released before close')
        guard()
    summary = ctx.summary_dict()
    complete = int(summary['flops_used'])
    require(complete==sum(s['flops'] for s in sections) and 0<complete<CALL_BUDGET,
            'Complete input/reference/guard/operator tariff, no hidden subtraction')
    tracked = len(refs)
    del ctx,refs
    gc.collect()
    record = dict(schema=SCHEMA,mode=mode,call=index,passed=True,completed=True,
        context_retired=True,all_array_weakrefs_retired=True,tracked_array_weakrefs=tracked,
        own_complete_bill=True,complete_flops=complete,bill=summary,rows=rows,
        sections=sections,rejected_cases=rejected,dense_branch_only=True,
        remote_sections_are_actual_budget_deltas=True,model_calls=0,target_accessed=False)
    verify_call(record,mode,index)
    return record


def verify_call(record,mode,index,tick=None):
    require(mode=='remote' and index==0 and record['schema']==SCHEMA
        and record['mode']==mode and record['call']==index and all(record.get(k) is True for k in
        ('passed','completed','context_retired','all_array_weakrefs_retired','own_complete_bill',
         'dense_branch_only','remote_sections_are_actual_budget_deltas'))
        and record['model_calls']==0 and record['target_accessed'] is False, 'Actual one-context algebra scope')
    sections = record['sections']
    require(len({x['name'] for x in sections})==len(sections)
        and all(type(x['flops']) is int and x['flops']>=0 and
                (x['expected_flops'] is None or x['flops']==x['expected_flops']) for x in sections)
        and sum(x['flops'] for x in sections)==record['complete_flops']==record['bill']['flops_used']
        and 0<record['complete_flops']<CALL_BUDGET, 'Actual complete and per-component tariff reconciliation')
    require(len(record['rows'])==18 and all(x['passed'] and
        all(math.isfinite(x[k]) and x[k]>=0 for k in
            ('maxabs','allowed_maxabs','reference_maxabs','relative_frobenius','reference_norm'))
        and x['maxabs']<=x['allowed_maxabs'] and x['relative_frobenius']<=2e-4
        and x['reference_norm']>=2**-12 for x in record['rows'])
        and [x['name'] for x in record['rejected_cases']]==list(REJECTIONS)
        and all(x['passed'] and x['expected_error'] in x['error'] for x in record['rejected_cases']),
        'All fixed direct references and expected failures')
    bill = record['bill']
    require(all(math.isfinite(bill[k]) and bill[k]>=0 for k in
        ('wall_time_s','residual_wall_time_s','flopscope_backend_time_s','flopscope_overhead_time_s'))
        and bill['wall_time_s']<60. and bill['residual_wall_time_s']<.4,
        'Unchanged actual RPC60s/residual<.4 context gate')


def call_records(result,tick=None):
    require(result['mode']=='remote' and len(result['calls'])==1, 'ONE freshly retired context')
    path = HERE/'tiny_qualification_v1/remote/call_00.json'
    item = result['calls'][0]
    require(item['path']==str(path) and sha(path,tick)==item['sha256']==result['artifacts'][str(path)],
            'Actual exclusive complete-call receipt')
    record = read(path)
    verify_call(record,'remote',0,tick)
    return [record]


def compare(remote,tick=None):
    require(all(remote.get(k) is True for k in ('passed','completed','own_complete_bill',
            'one_context_completed','source_hashes_unchanged')) and remote['retired_contexts']==1,
            'Fresh positive actual single RPC chain')
    record = call_records(remote,tick)[0]
    return dict(passed=True,one_context_completed=True,all_independent_references_passed=True,
        dense_component_tariffs_exact=True,expected_rejections_passed=True,
        complete_flops=record['complete_flops'],dense_branch_only=True,model_calls=0,
        truth_accessed=False,full_estimator_fidelity_established=False,
        runtime_gain_measured=False,raw_value_measured=False,automatic_next_stage=False)


def derive_main():
    original = _BASE['_node_bytes'](OWNER.read_bytes(),next(n for n in ast.parse(
        OWNER.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')).decode()
    changes = (
        ("args.mode in (('local','server') if forced_mode == 'local' else ('remote',))",
         "args.mode in (('server',) if forced_mode == 'local' else ('remote',))"),
        ("time.monotonic()-began < 270 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 1_000_000_000",
         "time.monotonic()-began < 120 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 8_000_000_000"),
        ("'Tiny mode270s/self1GB including final publication'","'Tiny mode120s/self8GB; independent full-tree cap required'"),
        ("signal.setitimer(signal.ITIMER_REAL,270.)","signal.setitimer(signal.ITIMER_REAL,120.)"),
        ("for index in range(2):","for index in range(1):"),
        ("min(limit,270-(time.monotonic()-began))","min(limit,120-(time.monotonic()-began))"),
        ("signal.setitimer(signal.ITIMER_REAL,270-(time.monotonic()-began))",
         "signal.setitimer(signal.ITIMER_REAL,120-(time.monotonic()-began))"),
        ("        repeat_check(records)","        require(len(records)==1, 'No local/repeated/full-model context')"),
        ("            two_contexts_completed=True,retired_contexts=2,all_array_weakrefs_retired=True,\n"
         "            all_independent_references_passed=True,metric_allowance_passed=True,expected_rejections_passed=True,\n"
         "            original_physical_coordinates_verified=True,repeat_byte_exact=True,",
         "            one_context_completed=True,retired_contexts=1,all_array_weakrefs_retired=True,\n"
         "            all_independent_references_passed=True,dense_component_tariffs_exact=True,\n"
         "            expected_rejections_passed=True,dense_branch_only=True,repeat_claimed=False,"),
    )
    source, positions = original, []
    for before,after in changes:
        require(source.count(before)==1, 'Unique original paid-loop/environment splice')
        at = source.index(before)
        positions.append((at,before,after))
        source = source[:at]+after+source[at+len(before):]
    inverse = source
    for at,before,after in reversed(positions):
        require(inverse[at:at+len(after)]==after, 'Exact paid-loop positional inverse')
        inverse = inverse[:at]+before+inverse[at+len(after):]
    require(inverse==original, 'Whole qualified primitive main restored')
    return source,dict(owner_sha256=OWNER_SHA,whole_main_source_inverse=True,
        original_backend_attestation_retained=True,original_closed_budget_API_retained=True,
        one_RPC_context_only=True,derived_sha256=hashlib.sha256(source.encode()).hexdigest())


def main(forced_mode='local'):
    source,proof = derive_main()
    namespace = dict(_BASE)
    namespace.update({k:v for k,v in globals().items() if k not in ('_BASE','main')})
    namespace['SOURCE_ADAPTER_PROOF'] = proof
    exec(compile(source,str(Path(__file__).resolve())+':exact-main-inverse','exec'),namespace)
    return namespace['main'](forced_mode=forced_mode)


if __name__=='__main__':
    raise SystemExit(main())
