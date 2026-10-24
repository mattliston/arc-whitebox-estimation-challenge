"""Inert saved-entry8 native/Torch OLD-only numerical-fidelity diagnostic.

No launcher, targets, heads, full forward, parameter update or new capture.
Root owns two SERIAL fresh processes: native FNP context then retired process;
only afterwards a CPU Torch/no_grad process. prepare() is stdlib-only in each.
native_phase() runs wholly inside root's owned FNP context. Its return contains
only bytes/plain metadata; root must close/reset/release/seal/reap before using
torch_phase(). Native complete bills are actual; Torch work is source-priced
offline research, never borrowed as a deployment bill.

Attribution, not efficacy: the native-Y cut must reduce BOTH old reconstructed
bank and old response-plane squared discrepancies to <=.25 of ordinary Torch.
Zero denominators are unresolved. No Q/F byte gate (QR coordinates have gauge).
Operational completion does not require this diagnostic attribution predicate.
"""
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import time
import types
import weakref

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
TRACE = NEW / 'current_comp_join_trace_information_v1'
OPERATOR = HERE / 'shared_operator.py'
SCHEMA = 'current_shared384_entry8_diagnostic_v1'
BUNDLE_SCHEMA = 'current_shared384_entry8_native_bundle_v1'
CANDIDATE_SHA = 'c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea'
TOTAL_FLOP_CAP = 100_000_000_000
BUNDLE_BYTE_CAP = 160 * 1024**2
PINS = {
    TRACE / 'probe.py': '17dc7f67e126f78d1bc390ed3c74f8347cd8077bc262406d032677be89c879b3',
    TRACE / 'watch_probe.py': 'bfb12b6801aa0ccef1e064276c602303e9f6d228a4d9b6914c4e05c28716fa75',
    TRACE / 'PROTOCOL.md': 'a46b8ceb091d99241fb9f38140c390dc2e8097f98f2ae17229b4b6092f5c8cea',
    TRACE / 'WATCH_SPEC.md': '98483acbac017f1f97418ccc3cf9ed6030956a423cf28a09d43d7ce13d61652a',
    OPERATOR: '09136ff645869df9d8fae3e2e8c16ff65c777971c308d1ba8c124d35649d2ad2',
    HERE / 'entry8_diagnostic.py': '46600bd3b322f8b3d032ecc1886f110f7c020b00ae9e3f2560debb91a4c58010',
    HERE / 'entry8_worker.py': 'da72d7091ccb4a374972e944d3c6982c70b7e7505e62ba6f225a2b707da7d01e',
    HERE / 'watch_entry8.py': 'b2d511351d2860d456e0357f09364dac9a0a2c90c85c17eb847e2e22e94cbdf0',
    HERE / 'ENTRY8_SELECTION_20260913_UTC.md': '3009ab9de3be0b1d759e7ff8071b1fbe08ba5dbe960d266b495dbbb7b257a17f',
    HERE / 'entry8_native_v1/result.json': 'd83f87a0b7635db6e5e02b6c7cdf9c5803e6b6e9cac506a9a9b5e71411bf40dd',
    HERE / 'entry8_native_v1/completion.json': 'eca48d6490c74cc141f1a67622f9c61ad1a64d4efbce347f432894a5d7dcea85',
    HERE / 'entry8_native_v1/finalization.json': '8d33f00e54b5b800c1e0cc3c0b92c4a8a7f1ee4b204ee58c1ebc492d5cf97e36',
    HERE / 'entry8_watch_v1/result.json': '781437c225cf4a11a602a7b9d7c94875d030c463ca39f8543b08a599e2a20b13',
    HERE / 'entry8_watch_v1/completion.json': '58aaa78a78f337f63fee861b9a8e994b1634f8a46c64f32acb8999dd98f9e291',
    HERE / 'entry8_watch_v1/finalization.json': 'cad55f4aa4fdb30b00ef4209db00446169cee56526e4c3938d040a246d20da0f',
}
STOPS = (2048,4096,6144,8128,9984,11712,13376,14912)
SHAPES = {
    'input.w': (1024,1024), 'input.p': (1024,13376), 'input.lam': (14912,),
    'input.previous': (896,), 'input.Q': (1024,384), 'input.F': (384,6144),
    'input.core': (384,384), 'input.response': (736,), 'input.live': (896,),
    'native.Y': (896,384), 'native.Q': (896,384), 'native.F': (384,8128),
    'native.core': (384,384), 'native.U': (1024,384),
    'native.old_pp': (1024,8128), 'native.old_plane': (736,896),
    'native.old_diag': (1024,), 'native.full_cp': (1024,1024),
    'native.full_diag': (1024,),
}
INT_NAMES = {'input.previous','input.response','input.live'}
DOUBLE_NAMES = {'native.old_diag','native.full_cp','native.full_diag'}
NATIVE_ORDER = ('native.Y','native.Q','native.F','native.core','native.U',
                'native.old_pp','native.old_plane')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''):
            h.update(block)
    return h.hexdigest()


def load(path,name,expected):
    require(sha(path)==expected,'Pinned inert helper before import')
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replace(source,edits):
    original=source; spans=[]
    for before,after in edits:
        require(source.count(before)==1,'Unique entry8 source splice: '+before[:70])
        at=source.index(before); source=source[:at]+after+source[at+len(before):]
        spans.append((at,before,after))
    inverse=source
    for at,before,after in reversed(spans):
        require(inverse[at:at+len(after)]==after,'Exact positional inverse')
        inverse=inverse[:at]+before+inverse[at+len(after):]
    require(inverse==original,'Complete unchanged source inverse')
    return source,dict(original_sha256=digest(original.encode()),
        derived_sha256=digest(source.encode()),splices=len(edits),whole_source_inverse=True)


def derived_sources(j,a,tick=None):
    original=a.source(); native=original; proofs=[]
    edits=[
        ("        Qn, unused_R = fnp.linalg.qr(Y, mode='reduced')",
         "        _entry8_snapshot('native.Y', Y)\n"
         "        Qn, unused_R = fnp.linalg.qr(Y, mode='reduced')"),
        ('        active_basis = Qn',
         "        _entry8_snapshot('native.Q', Qn)\n"
         "        _entry8_snapshot('native.F', state['F'])\n"
         "        _entry8_snapshot('native.core', state['core'])\n"
         '        active_basis = Qn'),
        ("    state['pre_basis'] = U",
         "    _entry8_snapshot('native.U', U)\n"
         "    _entry8_snapshot('native.old_pp', old)\n"
         "    state['pre_basis'] = U")]
    native,p=a.replace_function(native,'_shared_old384_push',edits); proofs.append(p)
    if tick is not None:tick()
    before="    old_plane = _shared_old384_mm(inner, state['pre_basis'][live_indices, :].T)"
    native,p=a.replace_function(native,'_shared_old384_read',[
        (before,before+"\n    _entry8_snapshot('native.old_plane', old_plane)")]); proofs.append(p)
    if tick is not None:tick()
    inherited,owner=j.derived_sources(a,tick)
    source=dict(estimator=native,setup=inherited['setup'],reader=inherited['reader'],
                ledger=inherited['ledger'])
    text=OPERATOR.read_text()
    left,right=a.function_span(text,'push'); push=text.encode()[left:right].decode()
    sketch=('        Y = J @ (weight[:, None]*(J.T @ Omega))\n'
            '        if old_columns:\n'
            '            Y = Y + Qp @ (core @ (Qp.T @ Omega))\n')
    tail=push[push.index('    # Dense explicit newest legs'):]
    push,pproof=replace(push,[
        ('def push(', 'def old_push('),
        ('Q, F, core, cohort_stops, rank=384):',
         'Q, F, core, cohort_stops, rank=384, native_Y=None):'),
        ('    total = int(p_full.shape[1])\n','    total = int(lam.numel())\n'),
        ("    _tensor(torch, p_full, (n, total), w32, 'p_full')\n",
         "    _tensor(torch, p_full, (n, 13376), w32, 'exact inherited entry8 p')\n"),
        (sketch,'        if native_Y is None:\n'+''.join('    '+line for line in sketch.splitlines(keepends=True))+
         '        else:\n            Y = native_Y\n'),
        (tail,'    return old, U, Fnew, corenew, old_columns, Y, basis')])
    left,right=a.function_span(text,'read'); read=text.encode()[left:right].decode()
    tail=read[read.index('    young_left, young_right ='):]
    read,rproof=replace(read,[('def read(', 'def old_read('),(tail,'    return old_plane')])
    source.update(torch_push=push,torch_read=read)
    proof=dict(candidate_sha256=CANDIDATE_SHA,native_functions=proofs,
        native_derived_sha256=digest(native.encode()),native_original_arithmetic_unchanged=True,
        paid_snapshots_are_observational=True,setup_reader_owner=owner,
        Torch_old_push=pproof,Torch_old_read=rproof,skipped_young_both_Torch_cuts=True,
        old_entry_native_coordinates_unchanged=True,no_head_or_predictor_called=True)
    return source,proof


def cost_certificate(base):
    """Conservative original native bound plus complete two OLD-only cuts."""
    c=base['cost_certificate']; n,a,r,R=1024,896,384,8128
    young={'young_inherited','young_newborn_base','young_read'}
    sketch={'Jt_Omega','J_product','Qt_Omega','old_core_sketch','old_Q_sketch'}
    rows=c['native_products']
    native=sum(x['upper_flops'] for x in rows)+c['native_QR_upper_flops']+c['initialization_upper_flops']
    native+=sum(c['pointwise'][k] for k in ('native_full_bank','native_join_core',
        'entry_copy_finite_export','weight_queries_copy_finite','control_copy_finite_export'))
    require(native==51_425_137_664,'Exact qualified native-source subset price')
    elems=sum(math.prod(shape) for shape in SHAPES.values())
    # Each snapshot: <=16/item covers copy, finite bitmap+reduction, native
    # representation export, scalar bookkeeping and dtype-rate2. Extra old
    # diagonal: conversion, two products, lambda weight, sum, finite/copy.
    # All original input restore/CP control costs already remain in native.
    snapshots=16*elems+16*n*R+8*n*n
    qr=8*a*r*r  # Explicit Q, deliberately above 4ar²-4r³/3.
    old=[x for x in rows if x['owner'] not in young]
    def classical(records):
        return sum(2*math.prod(x['shape']) for x in records)
    # Per cut <2.5BF accounts ALL Torch tensor ingress, slicing/copy/cat,
    # finite scans, weights, old diagonal, FP64 differences/norms/max, and
    # representation retirement. This is64 times the complete bundle
    # element inventory plus32 times the largest reconstructed old bank.
    pointwise=64*elems+32*n*R
    require(pointwise<2_500_000_000,'Explicit complete per-cut pointwise bound')
    own=classical(old)+qr+pointwise
    cut=classical([x for x in old if x['owner'] not in sketch])+qr+pointwise
    total=native+snapshots+own+cut
    require(total<TOTAL_FLOP_CAP,'Complete native plus both OLD-only Torch cuts <100BF')
    return dict(schema='current_shared384_entry8_price_v1',native_base_upper=native,
        native_snapshot_and_old_diagonal_upper=snapshots,native_upper=native+snapshots,
        Torch_ownY_upper=own,Torch_nativeY_upper=cut,per_cut_pointwise_upper=pointwise,
        Torch_explicit_QR_upper=qr,complete_planning_upper=total,cap=TOTAL_FLOP_CAP,
        native_actual_bill_required=True,Torch_is_source_priced_not_a_native_bill=True,
        width=n,active=a,rank=r,old_columns=R,all_original_native_young_work_charged=True)


def prepare(reviewed_sha256,tick=None):
    require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules),
            'Metadata admission before numerical imports in each fresh child')
    require(sha(__file__)==reviewed_sha256,'Explicit reviewed diagnostic source')
    j=load(TRACE/'probe.py','entry8_original_trace_source',PINS[TRACE/'probe.py'])
    args=types.SimpleNamespace(**{key:PINS[TRACE/name] for key,name in (
        ('source_sha256','probe.py'),('watcher_sha256','watch_probe.py'),
        ('protocol_sha256','PROTOCOL.md'),('watch_spec_sha256','WATCH_SPEC.md'))})
    base=j.admission(args,tick)
    c,a,unused_h,unused_backend,unused_legacy=j.helpers()
    source,proof=derived_sources(j,a,tick)
    sources=dict(base['source_bindings']); j.merge(sources,PINS)
    sources[str(Path(__file__).resolve())]=reviewed_sha256
    j.bind(sources,tick)
    require(base['manifest']['packets']['entry8']['p']['array']['shape']==[1024,13376],
            'Original p excludes exactly the pending newborn cohort')
    return dict(schema=SCHEMA,source_bindings=sources,input_bindings=base['input_bindings'],
        capture_seals={str(k):list(v) for k,v in base['capture_seals'].items()},
        manifest=base['manifest'],fixture=base['fixture'],
        packet_proof=base['packet_proof'],source=source,source_proof=proof,
        cost_certificate=cost_certificate(base),cohort_stops=list(STOPS),truth_accessed=False)


def remember(refs,value,seen=None):
    seen=set() if seen is None else seen
    if id(value) in seen:return
    seen.add(id(value))
    if isinstance(value,dict):
        for item in value.values():remember(refs,item,seen)
    elif isinstance(value,(tuple,list)):
        for item in value:remember(refs,item,seen)
    # API modules/classes expose shape/dtype too; they are not array owners.
    # The private native module is already tracked by an explicit weakref.
    elif (not isinstance(value,(types.ModuleType,type))
          and hasattr(value,'shape') and hasattr(value,'dtype')
          and (hasattr(value,'tobytes') or
               (hasattr(value,'detach') and hasattr(value,'data_ptr')))):
        try:refs.append(weakref.ref(value))
        except TypeError:pass


def validate_bundle(bundle):
    m,blob=bundle['metadata'],bundle['blob']
    require(isinstance(blob,bytes) and m['schema']==BUNDLE_SCHEMA and m['rank']==35
        and m['layer']==8 and m['candidate_sha256']==CANDIDATE_SHA
        and m['bytes']==len(blob)<BUNDLE_BYTE_CAP and m['sha256']==digest(blob)
        and m['byteorder']==sys.byteorder
        and set(m['arrays'])==set(SHAPES) and m['cohort_stops']==list(STOPS),
        'One exact typed detached same-entry bundle')
    cursor=0
    for name,row in sorted(m['arrays'].items(),key=lambda item:item[1]['offset']):
        dtype='int64' if name in INT_NAMES else 'float64' if name in DOUBLE_NAMES else 'float32'
        size=math.prod(SHAPES[name])*(8 if dtype in ('int64','float64') else 4)
        require(row['shape']==list(SHAPES[name]) and row['dtype']==dtype and row['bytes']==size
            and row['offset']==cursor and row['sha256']==digest(blob[cursor:cursor+size]),
            'Contiguous nonaliasing fixed bundle member '+name)
        cursor+=size
    require(cursor==len(blob) and m['native_control']['cp_pre']['byte_exact']
        and m['native_control']['diagonal']['byte_exact'],'Complete byte-exact native witness')
    return True


def native_phase(context,np,flops,fnp,ctx,refs):
    """Inside root-owned native FNP BudgetContext; return bytes, never arrays."""
    require('torch' not in sys.modules and ctx is not None,'No Torch in native child')
    j=load(TRACE/'probe.py','entry8_native_trace_owner',PINS[TRACE/'probe.py'])
    c,unused_a,unused_h,unused_backend,unused_legacy=j.helpers()
    chunks=[]; arrays={}; cursor=0; callback_order=[]; original_old=None
    def snapshot(name,value):
        nonlocal cursor,original_old
        require(name in SHAPES and name not in arrays and tuple(value.shape)==SHAPES[name],
                'Exact once-only paid snapshot '+name)
        copy=fnp.copy(value)
        require(bool(fnp.all(fnp.isfinite(copy))),'Finite paid native snapshot')
        data=copy.tobytes(); refs.append(weakref.ref(copy))
        arrays[name]=dict(offset=cursor,bytes=len(data),shape=list(copy.shape),
                          dtype=str(copy.dtype),sha256=digest(data))
        chunks.append(data); cursor+=len(data)
        if name in NATIVE_ORDER:callback_order.append(name)
        if name=='native.old_pp':original_old=value
    raw=Path(context['manifest']['packet_file']).read_bytes()
    require(digest(raw)==context['manifest']['packet_sha256'],'Same qualified native entry bytes')
    weights=np.load(context['fixture'],mmap_mode='r',allow_pickle=False)
    module=types.ModuleType('entry8_private_native_c2');refs.append(weakref.ref(module))
    module.__file__='<entry8-exact-native-statements-with-paid-snapshots>'
    module._entry8_snapshot=snapshot
    try:
        for key in ('estimator','setup','reader'):
            exec(compile(context['source'][key],module.__file__+':'+key,'exec'),vars(module))
        require(module._SCRATCH is None,'Fresh exact native setup owner');module._jtrace_setup()
        packets=context['manifest']['packets'];entry=packets['entry8'];pre=packets['pre8']
        restored={k:c.restore(entry[k],raw,np,fnp,refs,'entry8.'+k) for k in j.ENTRY_KEYS}
        response,live=[j.fixed_query(pre[k]['array'],raw,np,fnp,refs) for k in j.QUERY_KEYS]
        w=fnp.array(weights[8],dtype=fnp.float32,copy=True)
        require(bool(fnp.all(fnp.isfinite(w))) and w.tobytes()==weights[8].tobytes(),
                'Paid exact canonical W8')
        inputs=dict(w=w,p=restored['p'],lam=restored['lam'],previous=restored['previous'],
            Q=restored['shared_old']['Q'],F=restored['shared_old']['F'],
            core=restored['shared_old']['core'],response=response,live=live)
        for key,value in inputs.items():snapshot('input.'+key,value)
        pp,cp,diag=module._jtrace_reader(8,1024,w,restored['p'],restored['previous'],
            restored['newborn'],restored['lam'],restored['shared_old'],response,live)
        require(tuple(callback_order)==NATIVE_ORDER,'Exact original native join/read chronology')
        controls={}
        for key,value in (('cp_pre',cp),('diagonal',diag)):
            row=pre[key]['array'];copy=fnp.copy(value);refs.append(weakref.ref(copy))
            require(bool(fnp.all(fnp.isfinite(copy))),'Paid native control finite')
            data=copy.tobytes();lo=row['offset']
            require(digest(data)==row['sha256'] and data==raw[lo:lo+row['bytes']],
                    'Byte-exact full original native pre8 '+key+' before attribution')
            controls[key]=dict(sha256=digest(data),byte_exact=True)
        old64=original_old.astype(fnp.float64)
        old_diag=fnp.sum(module._cp_diagonal_cube(old64)*restored['lam'][:8128].astype(fnp.float64),axis=1)
        snapshot('native.old_diag',old_diag);snapshot('native.full_cp',cp);snapshot('native.full_diag',diag)
        blob=b''.join(chunks)
        result=dict(metadata=dict(schema=BUNDLE_SCHEMA,rank=35,layer=8,candidate_sha256=CANDIDATE_SHA,
            bytes=len(blob),sha256=digest(blob),byteorder=sys.byteorder,arrays=arrays,
            cohort_stops=list(STOPS),capture_seals=context['capture_seals'],
            packet_sha256=context['manifest']['packet_sha256'],
            native_control=controls,source_proof=context['source_proof'],
            cost_certificate=context['cost_certificate'],truth_accessed=False),blob=blob)
        validate_bundle(result)
        return result
    finally:
        remember(refs,vars(module));remember(refs,locals());remember(refs,inputs if 'inputs' in locals() else {})
        module.__dict__.clear();original_old=None;chunks.clear()


def torch_phase(context,bundle,np,torch,refs):
    """Fresh CPU-only child AFTER native process retirement and six-seal join."""
    require('flopscope' not in sys.modules,'No live/imported native runtime in Torch child')
    validate_bundle(bundle)
    require(bundle['metadata']['source_proof']==context['source_proof']
        and bundle['metadata']['cost_certificate']==context['cost_certificate']
        and bundle['metadata']['capture_seals']==context['capture_seals']
        and bundle['metadata']['packet_sha256']==context['manifest']['packet_sha256'],
        'Same source/price/capture chain across the retired process boundary')
    for key in ('cp_pre','diagonal'):
        require(bundle['metadata']['native_control'][key]['sha256']==
            context['manifest']['packets']['pre8'][key]['array']['sha256'],
            'Known original full CP control hash, not a supplied assertion')
    op=load(OPERATOR,'entry8_frozen_shared_operator',PINS[OPERATOR])
    env=dict(vars(op))
    exec(compile(context['source']['torch_push']+'\n\n'+context['source']['torch_read'],
        '<exact-old-only-shared384-proposal>', 'exec'),env)
    arrays={};m=bundle['metadata'];blob=bundle['blob']
    def tensor(name):
        row=m['arrays'][name];lo=row['offset'];data=blob[lo:lo+row['bytes']]
        view=np.frombuffer(data,dtype=row['dtype']).reshape(row['shape'])
        value=torch.tensor(view,dtype=getattr(torch,row['dtype']),device='cpu')
        refs.append(weakref.ref(value))
        require(bool(torch.isfinite(value).all()) and value.numpy().tobytes()==data,
                'Finite exact CPU byte ingress '+name)
        return value
    def errors(value,reference):
        x,y=value.to(torch.float64),reference.to(torch.float64)
        d=x-y;ss=(d*d).sum();scale=(y*y).sum()
        require(bool(torch.isfinite(ss)&torch.isfinite(scale)),'Finite physical comparison')
        return dict(squared_error=float(ss),reference_squared_norm=float(scale),
            maxabs=float(d.abs().max()),RMS=float(torch.sqrt(ss/d.numel())),
            relative_L2=float(torch.sqrt(ss/scale)) if bool(scale>0) else None)
    try:
        with torch.no_grad():
            for name in ('input.w','input.p','input.lam','input.previous','input.Q','input.F',
                         'input.core','input.response','input.live','native.Y','native.old_pp',
                         'native.old_plane','native.old_diag'):
                arrays[name]=tensor(name)
            require(torch.get_num_threads()==1,'Single CPU Torch thread')
            inp=lambda key:arrays['input.'+key]
            outputs={}
            for name,cut in (('own_Y',None),('native_Y',arrays['native.Y'])):
                started=time.monotonic()
                old,U,F,core,count,Y,Q=env['old_push'](torch,8,inp('w'),inp('p'),inp('lam'),
                    inp('previous'),inp('Q'),inp('F'),inp('core'),STOPS,rank=384,native_Y=cut)
                require(count==8128 and old.shape==(1024,8128),'All old prefix, no newborn confinement')
                plane=env['old_read'](torch,old,inp('lam')[:count],inp('response'),inp('live'),U,F,count)
                old64=old.to(torch.float64)
                diagonal=((old64*old64)*old64*inp('lam')[:count].to(torch.float64)).sum(1)
                require(all(bool(torch.isfinite(v).all()) for v in (old,U,F,core,Y,Q,plane,diagonal)),
                        'Finite complete OLD-only cut')
                outputs[name]=dict(old_pp=errors(old,arrays['native.old_pp']),
                    old_plane=errors(plane,arrays['native.old_plane']),
                    old_diag=errors(diagonal,arrays['native.old_diag']),
                    Y=errors(Y,arrays['native.Y']),elapsed_s=time.monotonic()-started,
                    planned_upper_flops=context['cost_certificate']['Torch_'+('ownY' if cut is None else 'nativeY')+'_upper'])
                remember(refs,(old,U,F,core,Y,Q,plane,diagonal,old64))
                del old,U,F,core,Y,Q,plane,diagonal,old64
            ratios={}
            for key in ('old_pp','old_plane'):
                ordinary=outputs['own_Y'][key]['squared_error']
                cut=outputs['native_Y'][key]['squared_error']
                ratios[key]=cut/ordinary if ordinary>0 else None
            resolved=all(v is not None for v in ratios.values())
            support=resolved and all(v<=.25 for v in ratios.values())
            return dict(schema=SCHEMA,passed=True,completed=True,rank=35,layer=8,
                comparisons=outputs,native_Y_over_own_Y_squared_error=ratios,
                attribution_denominators_nonzero=resolved,rangefinder_dominant_support=bool(support),
                outcome='rangefinder_dominant_support' if support else 'attribution_negative' if resolved else 'unresolved_zero_denominator',
                current_CP_reader_byte_exact=True,old_coordinate_gauge_not_scored=True,
                fixed_rank384_age4=True,parameter_updates=0,full_model_calls=0,
                truth_accessed=False,score_measured=False,fit_authorized=False,
                cost_certificate=context['cost_certificate'])
    finally:
        remember(refs,locals());arrays.clear();env.clear()
