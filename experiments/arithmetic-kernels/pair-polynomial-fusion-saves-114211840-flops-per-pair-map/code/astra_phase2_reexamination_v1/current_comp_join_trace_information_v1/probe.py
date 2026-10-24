"""One target-free current-c2 entry8 join/reader trace-information context.

Inert without --run. No nonlinear predictor, moment archive, truth or fit.
The original stored-Q/F tensor and its orthogonal shortcut are distinct.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import sys
import time
import traceback
import types
import weakref

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
JCELL = NEW/'current_comp_joint_cell_attribution_v1'
SHARED = NEW/'current_comp_shared_old384_v1'
CANDIDATE = NEW/'current_comp_shared_gain_slice_v1/build_v1/estimator.py'
CANDIDATE_SHA = 'c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea'
OUTPUT = HERE/'probe_v1'
SCHEMA = 'current_comp_join_trace_information_v1'
CAP, RSS, FLOP_CAP = 120., 8_000_000_000, 200_000_000_000
KEYS = ('source','watcher','protocol','watch-spec')
PINS = {
    JCELL/'consumer.py': 'cb9649482c362a9b13a0ebe3e57cf536ac5657cd970defc18ab09f659967e2bb',
    JCELL/'adapter_v2.py': '6f30900ea7a0e3babf2b339c327475d5d64d16fa6ebfdc2ff74c44167e8b3510',
    SHARED/'probe.py': '6193396f83150e7e95f3c16ad5095eeb8d4c62da17d69d5bd7dcb08fad9a743b',
    SHARED/'backend_probe.py': '36d9068c8dead7f32b53da9c5c5382f9004a8a45c0c47c29d7e1f4b2435fdcc8',
    SHARED/'check4_v2.py': '8d09d9b1f443e229b1641d277312b52c204392648b25445ffab63b82dba16058',
    CANDIDATE: CANDIDATE_SHA,
}
ENTRY_KEYS = ('p','lam','previous','newborn','shared_old')
QUERY_KEYS = ('response_indices','live_indices')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(path, tick=None):
    path = Path(path)
    require(not path.is_symlink(), 'No symlink owner')
    path = path.resolve()
    require(path.is_relative_to(P2) and path.name != 'key.txt'
        and not any(x in path.parts for x in ('truthpost','public_g19_attribution_v1',
                                            'covariance_replica_truth_v1'))
        and not (path.suffix in ('.npy','.npz') and 'truth' in str(path).lower()),
        'Only target-free local bytes; no A/B/final-label/key access')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20), b''):
            h.update(block)
            if tick is not None: tick()
    return h.hexdigest()


def read(path):
    path = Path(path)
    require(not path.is_symlink() and path.stat().st_size < 16*1024**2, 'Bounded JSON only')
    return json.loads(path.read_text())


def write(path, value):
    def safe(x):
        if isinstance(x,float) and not math.isfinite(x): return dict(nonfinite_float=repr(x))
        if isinstance(x,dict): return {str(k):safe(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)): return [safe(v) for v in x]
        return x
    with Path(path).open('x') as stream:
        json.dump(safe(value),stream,sort_keys=True,indent=2,allow_nan=False); stream.write('\n')


def bind(items, tick=None):
    require(isinstance(items,dict), 'Flat byte inventory')
    for path,d in items.items():
        require(re.fullmatch('[0-9a-f]{64}',d or '') and sha(path,tick)==d,
                'Actual immutable owner: '+str(path))


def merge(destination, items):
    for path,d in items.items():
        path = str(path)
        require(path not in destination or destination[path]==d, 'No conflicting owner')
        destination[path] = d


def load(path, name, expected):
    require(sha(path)==expected, 'Source pinned before inert import')
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def helpers():
    # Code-only reuse. In particular, never call the old consumer admission,
    # derive/restart/main/supply routines or its failed scientific receipts.
    c = load(JCELL/'consumer.py','join_trace_packet_owner',PINS[JCELL/'consumer.py'])
    a = load(JCELL/'adapter_v2.py','join_trace_span_owner',PINS[JCELL/'adapter_v2.py'])
    h = load(SHARED/'probe.py','join_trace_lifecycle_api',PINS[SHARED/'probe.py'])
    backend = load(SHARED/'backend_probe.py','join_trace_backend_api',PINS[SHARED/'backend_probe.py'])
    legacy = h.load(h.LEGACY,'join_trace_original_bill_reset',h.LEGACY_SHA)
    return c,a,h,backend,legacy


def derived_sources(a, tick=None):
    """One reversible observer insertion; exact original paid statement spans."""
    original = a.source()
    if tick is not None: tick()
    changed, join_proof = a.replace_function(original,'_shared_old384_push',[
        ('        active_basis = Qn',
         "        _jtrace_capture(Qn, state['F'])\n        active_basis = Qn")])
    require(changed.replace("        _jtrace_capture(Qn, state['F'])\n",'',1)==original,
            'Whole current-c2 source inverse, including every head/literal byte')
    lo,hi = a.function_span(original,'_predict_binary')
    body = original.encode()[lo:hi].decode()
    start,stop = body.index('    if _SCRATCH is None:'),body.index('    previous = fnp.arange(n)')
    setup_span = body[start:stop]
    setup = 'def _jtrace_setup():\n    global _SCRATCH\n'+setup_span
    start,stop = body.index('        pp = _shared_old384_push('),body.index('        star_pre = ')
    reader_span = body[start:stop]
    reader_body = ''.join(line[4:] if line.startswith('    ') else line
                          for line in reader_span.splitlines(keepends=True))
    require(''.join('    '+line for line in reader_body.splitlines(keepends=True))==reader_span,
            'Exact CP statement span, reversible indentation only')
    reader = ('def _jtrace_reader(l,n,w32,p,previous,newborn,lam,shared_old,response_indices,live_indices):\n'
              +reader_body+'    return pp, cp_pre, diagonal\n')
    ledger_text = (SHARED/'check4_v2.py').read_text()
    left,right = a.function_span(ledger_text,'_v2_ledger_metrics')
    ledger = ledger_text.encode()[left:right].decode()
    if tick is not None: tick()
    return dict(estimator=changed,setup=setup,reader=reader,ledger=ledger),dict(
        candidate_sha256=CANDIDATE_SHA,derived_estimator_sha256=digest(changed.encode()),
        whole_source_inverse=True,only_zero_F_borrowed_join_callback=True,join=join_proof,
        setup_original_span_sha256=digest(setup_span.encode()),
        reader_original_span_sha256=digest(reader_span.encode()),
        setup_sha256=digest(setup.encode()),reader_sha256=digest(reader.encode()),
        ledger_sha256=digest(ledger.encode()),actual_original_CP_statements=True,
        no_predictor_or_head_called=True,candidate_materialized=False)


def cost_certificate(manifest):
    """Source upper counts; a measured bill must also fit this bound and200BF."""
    n,a,r,R,b,old,s,v = 1024,896,384,8128,1984,6144,736,896
    entry = manifest['packets']['entry8']; pre = manifest['packets']['pre8']
    require(entry['previous']['array']['shape']==[a]
        and pre['response_indices']['array']['shape']==[s]
        and pre['live_indices']['array']['shape']==[v], 'Exact fixed support/query geometry')
    state = entry['shared_old']['mapping']
    cohorts = [tuple(z['scalar'] for z in row['sequence']) for row in state['cohorts']['sequence']]
    require(cohorts==[(0,2048),(2048,2048),(4096,2048),(6144,1984),
        (8128,1856),(9984,1728),(11712,1664),(13376,1536)]
        and state['columns']=={'scalar':6144}, 'All old prefix through join3, never newborn')
    def mm(m,k,q):
        # Same fixed parent dispatcher bound as qualified JCELL consumer.
        # 96 per operand/output cell dominates packed D6 max64.687/35.964
        # additions/copies; tiling and explicit concatenation are not free.
        if m>1024:
            return sum(mm(min(1024,m-j),k,q) for j in range(0,m,1024))+m*q
        if q>8192:
            return sum(mm(m,k,min(8192,q-j)) for j in range(0,q,8192))+m*q
        if k>8192:
            return sum(mm(m,min(8192,k-j),q) for j in range(0,k,8192))+((k-1)//8192)*m*q
        return 2*m*k*q+96*(m*k+k*q+m*q)
    native_shapes = [
        ('Jt_Omega',(b,a,r)),('J_product',(a,b,r)),('Qt_Omega',(r,a,r)),
        ('old_core_sketch',(r,r,r)),('old_Q_sketch',(a,r,r)),('Fj',(r,a,b)),
        ('joined_core',(r,b,r)),('overlap',(r,a,r)),('rotate_F',(r,r,old)),
        ('core_right',(r,r,r)),('core_left',(r,r,r)),('affine_basis',(n,a,r)),
        ('old_reconstruction',(n,r,R)),('young_inherited',(n,a,13376-R)),
        ('young_newborn_base',(n,a,768)),('old_read_inner',(s,R,r)),
        ('old_read_outer',(s,r,v)),('young_read',(s,14912-R,v))]
    products = [dict(owner=k,shape=list(shape),upper_flops=mm(*shape)) for k,shape in native_shapes]
    diagnostic_shapes = [('dense_QF',(a,r,R)),('unprojected_queries',(n,a,R)),
        ('projected_queries',(n,a,R)),('unprojected_plane',(s,R,v)),
        ('projected_plane',(s,R,v)),('query_Gram',(s,a,v)),('Qt_Q',(r,a,r)),
        ('source_trace',(a,R,1)),('projected_trace',(a,R,1)),
        ('shortcut_F_trace',(r,R,1)),('shortcut_Q_trace',(a,r,1)),
        ('trace_queries',(n,a,1))]
    diagnostics = [dict(owner=k,shape=list(shape),upper_flops=4*math.prod(shape))
                   for k,shape in diagnostic_shapes]
    entry_elements = sum(math.prod(row['shape']) for key in ENTRY_KEYS
                         for unused,row in packet_arrays(entry[key]))
    # Native: <=32 weighted operations/full14912-column cell (FP64 cube,
    # cast, multiply/reduce; FP32 gathers, young tails, finite/concat/scatter)
    # plus <=64 per join/core cell. Diagnostic: two large FP64 reconstructed
    # banks/transports have <=32 operations/cell INCLUDING dtype-rate2;
    # two reader squares/weights + gathers are included again conservatively.
    # The remaining matrix-shaped pointwise terms, scalar reductions, norm
    # envelopes, guards and selector copies fit128*n²; no GEMM hidden here.
    pointwise = dict(native_full_bank=32*n*14912,native_join_core=64*(a*b+r*R+r*r),
        diagnostic_large_arrays=64*(a*R+n*R+s*R+v*R),
        diagnostic_small_arrays=128*n*n,entry_copy_finite_export=8*entry_elements,
        weight_queries_copy_finite=8*n*n,control_copy_finite_export=8*(n*n+n))
    # The complete parent cold startup is1,944,970,298F. Adding a second
    # charge for all292,225,024 arena elements and1MF scalar bookkeeping is
    # <2.24BF.3BF bounds this fresh subset even without startup credit.
    setup,qr = 3_000_000_000,8*a*r*r
    upper = sum(x['upper_flops'] for x in products+diagnostics)+sum(pointwise.values())+setup+qr
    require(upper <= FLOP_CAP, 'Explicit complete source price established before numerical import')
    return dict(schema='current_comp_join_trace_cost_v1',a=a,rank=r,old_before=old,
        joined_columns=R,joining_width=b,response_rows=s,live_rows=v,
        native_products=products,FP64_diagnostic_products=diagnostics,pointwise=pointwise,
        native_QR_upper_flops=qr,initialization_upper_flops=setup,
        complete_upper_flops=upper,hard_cap_flops=FLOP_CAP,actual_bill_still_required=True,
        no_full_model_cost_or_speed_claim=True,explicit_array_peak_upper_bytes=3_500_000_000,
        actual_entire_tree_RSS_limit_bytes=RSS)


def packet_arrays(value):
    if isinstance(value,dict) and 'array' in value:
        yield '',value['array']
    elif isinstance(value,dict):
        for item in value.values(): yield from packet_arrays(item)
    elif isinstance(value,list):
        for item in value: yield from packet_arrays(item)


def admission(args,tick=None):
    """Stdlib source/JSON/byte admission only; no NPZ/NPY numerical decoding."""
    sources = {str(HERE/name):getattr(args,key.replace('-','_')+'_sha256')
        for name,key in zip(('probe.py','watch_probe.py','PROTOCOL.md','WATCH_SPEC.md'),KEYS)}
    merge(sources,PINS); bind(sources,tick)
    c,a,h,backend,legacy = helpers()
    inputs,records = {},{}
    for folder,expected in c.CAPTURE_CHAINS.items():
        paths = [JCELL/folder/name for name in ('result.json','completion.json','finalization.json')]
        actual = tuple(sha(p,tick) for p in paths)
        require(actual==expected, 'Actual positive capture six-seal boundary')
        record,end,final = map(read,paths)
        require(all(x.get('passed') is True and x.get('completed') is True for x in (record,end,final))
            and end['result_sha256']==final['result_sha256']==actual[0]
            and final['completion_sha256']==actual[1], 'All linked positive seals, not partial eligibility')
        merge(inputs,dict(zip(paths,actual))); records[folder]=record
    captured,watch = records['capture_v1'],records['capture_watch_v1']
    require(captured['schema']=='current_comp_joint_cell_capture_v1'
        and watch['schema']=='current_comp_joint_cell_capture_watch_v1'
        and captured['candidate_sha256']==watch['candidate_sha256']==CANDIDATE_SHA
        and captured['full_model_calls']==captured['retired_call_count']==1
        and captured['truth_accessed'] is False and all(captured.get(k) is True for k in
            ('baseline_output_byte_exact','all16_domains_qualified','all47_selectors_qualified',
             'all_original_operations_exact','own_complete_bills','all_capture_arrays_released',
             'all_parameter_and_estimator_owners_retired','hashes_unchanged')),
        'Actual unchanged target-free c2 capture, all16/47 and complete original operation identity')
    require([watch['child_'+k] for k in ('result_sha256','completion_sha256','finalization_sha256')]
        ==list(c.CAPTURE_CHAINS['capture_v1']) and watch['whole_owned_tree_retired']
        and watch['cleanup_passed'] and watch['maximum_combined_rss_bytes']<RSS
        and watch['maximum_interval_s']<=1 and watch['maximum_swap_growth_bytes']==0
        and watch['elapsed_s']<180, 'Original actual watch joins/resources/retirement')
    for record in (captured,watch):
        for group in ('source_bindings','input_bindings','artifacts'):
            merge(sources if group=='source_bindings' else inputs,record[group])
    merge(sources,c.PINS); merge(sources,backend._BP_PINS); merge(sources,legacy.PINS)
    merge(sources,{h.LEGACY:h.LEGACY_SHA})
    manifest_path = JCELL/'capture_v1/restart_packet.json'
    manifest = read(manifest_path); packet_proof = c.validate_packet(manifest)
    require(inputs[str(manifest_path)]==sha(manifest_path,tick)
        and manifest['packet_file']==str(JCELL/'capture_v1/restart_packet.bin')
        and inputs[manifest['packet_file']]==c.PACKET_SHA
        and captured['packet_bytes']==manifest['bytes'], 'Exact sealed packet, no surrogate checkpoint')
    fixture = [p for p in inputs if p.endswith('/0035/weights.npy')]
    require(len(fixture)==1, 'Exactly one physical rank35 canonical weights source')
    require(not os.environ.get('FLOPSCOPE_DISABLE_WEIGHTS')
        and not os.environ.get('FLOPSCOPE_WEIGHTS_FILE'), 'Original complete default tariffs')
    tariff = NEW/'openblas_runtime_env_v1/lib/python3.11/site-packages/flopscope/data/default_weights.json'
    require(sources[str(tariff)]=='f729bc47f07c080b8c7b9d7b045e5663f0af313669b4c2a962b5a9044a94cb7a'
        and all(read(tariff)['dtype_rates'][k]==v for k,v in
            dict(bool=1,float32=1,float64=2,int32=1,int64=2).items()), 'Exact FP64 rate2 price basis')
    source,proof = derived_sources(a,tick)
    certificate = cost_certificate(manifest)
    bind(sources,tick); bind(inputs,tick)
    require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules), 'No admission numerics')
    return dict(source_bindings=sources,input_bindings=inputs,capture_seals=c.CAPTURE_CHAINS,
        manifest=manifest,packet_proof=packet_proof,fixture=fixture[0],source=source,
        source_proof=proof,cost_certificate=certificate,truth_accessed=False)


def remember(refs,value,seen=None):
    """References only; no numerical copy/scan and no borrowed array return."""
    if seen is None: seen=set()
    if id(value) in seen: return
    seen.add(id(value))
    if hasattr(value,'shape') and hasattr(value,'dtype') and hasattr(value,'tobytes'):
        refs.append(weakref.ref(value))
    elif isinstance(value,dict):
        for item in value.values(): remember(refs,item,seen)
    elif isinstance(value,(list,tuple)):
        for item in value: remember(refs,item,seen)


def fixed_query(row,raw,np,fnp,refs):
    data = raw[row['offset']:row['offset']+row['bytes']]
    require(row['dtype']=='int64' and digest(data)==row['sha256'], 'Fixed archived selector bytes')
    view = np.frombuffer(data,dtype='int64').reshape(row['shape'])
    answer = fnp.array(view,dtype=fnp.int64,copy=True)
    require(bool(fnp.all(fnp.isfinite(answer))) and bool(fnp.all((answer>=0)&(answer<1024)))
        and answer.tobytes()==data, 'Paid exact fixed query copy/range/finite')
    refs.append(weakref.ref(answer)); return answer


def projection_diagnostics(fnp,X0,Q,F,lam,W,response,live,refs):
    """All arithmetic (including copies, finite/norm/error/floor) is metered FNP."""
    a,r,R,n = 896,384,8128,1024
    X0,Q,F,lam,W = (x.astype(fnp.float64) for x in (X0,Q,F,lam,W))
    require(X0.shape==(a,R) and Q.shape==(a,r) and F.shape==(r,R)
        and lam.shape==(R,) and W.shape==(a,n), 'Entire selected old prefix and exact active support')
    for value in (X0,Q,F,lam,W):
        require(bool(fnp.all(fnp.isfinite(value))), 'Finite actual FP64 diagnostic inputs')
    one = fnp.asarray((1.,),dtype=fnp.float64)
    two = fnp.asarray((2.,),dtype=fnp.float64)
    three = fnp.asarray((3.,),dtype=fnp.float64)
    gamma = fnp.asarray((64*(2**-52)*(2*a+2*r+4*R+64),),dtype=fnp.float64)
    def total(x): return fnp.sum(x,keepdims=True).reshape((1,))
    def norm2(x): return total(x*x)
    def finite(x): require(bool(fnp.all(fnp.isfinite(x))), 'Finite metered diagnostic result'); return x
    def scalar(x):
        require(tuple(x.shape)==(1,), 'Typed one-element scalar, never a backend RemoteScalar')
        return float(x[0])
    X1 = finite(Q@F)  # Norms of this actual dense FP64 product define exact trace.
    norms0 = fnp.sum(X0*X0,axis=0)
    norms1 = fnp.sum(X1*X1,axis=0)
    normF = fnp.sum(F*F,axis=0)
    t0,t1 = X0@(lam*norms0),X1@(lam*norms1)
    delta = finite(t0-t1)
    shortcut_delta = finite(t0-Q@(F@(lam*normF)))
    gram_error = Q.T@Q-fnp.eye(r,dtype=fnp.float64)
    epsilon_Q = fnp.sqrt(norm2(gram_error))
    Qnorm_bound = fnp.sqrt(one+epsilon_Q)
    # Cauchy envelopes include both represented tensors and signed absolute
    # weights. The floor is a fixed conservative arithmetic scale, not a CI.
    Fcube = normF*fnp.sqrt(normF)
    S = total(fnp.abs(lam)*(norms0*fnp.sqrt(norms0)+Qnorm_bound**3*Fcube))
    trace_floor = (gamma*S)**2
    delta_norm2 = norm2(delta)
    shortcut_norm2 = norm2(shortcut_delta-delta)
    safe_trace = fnp.where(delta_norm2>trace_floor,delta_norm2,one)
    shortcut_relative = fnp.sqrt(shortcut_norm2/safe_trace)
    shortcut_bound = Qnorm_bound*epsilon_Q*total(fnp.abs(lam)*Fcube)
    d = delta/fnp.asarray((a+2.,),dtype=fnp.float64)
    u = finite(W.T@d)
    g = finite(fnp.sum(W*W,axis=0))
    D = finite(W[:,response].T@W[:,live])
    hdiag = three*u*g
    hplane = g[response,None]*u[None,live]+two*D*u[response,None]
    equal = response[:,None]==live[None,:]
    # Repeated-index entries own the same cubic diagonal convention.
    hplane = fnp.where(equal,hdiag[response,None],hplane)
    def query(X):
        Z = finite(W.T@X)
        diag = finite(fnp.sum((Z*Z)*Z*lam[None,:],axis=1))
        left,right = Z[response],Z[live]
        plane = finite((left*left)@(right*lam[None,:]).T)
        plane = fnp.where(equal,diag[response,None],plane)
        remember(refs,locals())
        return diag,plane
    k0,R0 = query(X0)
    k1,R1 = query(X1)
    del X0,X1,F,Q,gram_error
    diag_error,plane_error = k0-k1,R0-R1
    diag_after,plane_after = k0-(k1+hdiag),R0-(R1+hplane)
    diagonal_scale = g*fnp.sqrt(g)*S
    plane_scale = g[response,None]*fnp.sqrt(g[None,live])*S
    def report(de,da,pe,pa,ds,ps):
        require(de.size>0 and pe.size>0, 'Nonempty fixed sketch-unused queries')
        dn,pn = norm2(de),norm2(pe)
        da2,pa2 = norm2(da),norm2(pa)
        df,pf = norm2(gamma*ds),norm2(gamma*ps)
        dvalid,pvalid = bool(dn>df),bool(pn>pf)
        dr = da2/fnp.where(dn>df,dn,one)
        pr = pa2/fnp.where(pn>pf,pn,one)
        for value in (dn,pn,da2,pa2,df,pf,dr,pr): finite(value)
        answer = dict(diagonal_queries=int(de.size),plane_entries=int(pe.size),
            diagonal_error_MSE=scalar(dn/fnp.asarray((de.size,),dtype=fnp.float64)),
            corrected_diagonal_error_MSE=scalar(da2/fnp.asarray((de.size,),dtype=fnp.float64)),
            response_plane_error_Frobenius_squared=scalar(pn),
            corrected_response_plane_error_Frobenius_squared=scalar(pa2),
            diagonal_denominator_squared=scalar(dn),plane_denominator_squared=scalar(pn),
            diagonal_roundoff_floor_squared=scalar(df),plane_roundoff_floor_squared=scalar(pf),
            denominators_resolved=dvalid and pvalid,
            diagonal_error_ratio=scalar(dr) if dvalid else None,
            plane_error_ratio=scalar(pr) if pvalid else None)
        remember(refs,locals()); return answer
    # These columns did not construct the join sketch. Existing selectors
    # depend on all columns: no independence/Gaussian-sampling inference.
    rp,lp = response>=384,live>=384
    primary = report(diag_error[384:],diag_after[384:],plane_error[rp][:,lp],
        plane_after[rp][:,lp],diagonal_scale[384:],plane_scale[rp][:,lp])
    full = report(diag_error,diag_after,plane_error,plane_after,diagonal_scale,plane_scale)
    trace_resolved = bool(delta_norm2>trace_floor)
    for value in (delta_norm2,trace_floor,shortcut_norm2,shortcut_relative,shortcut_bound,epsilon_Q,S): finite(value)
    shortcut_ok = trace_resolved and scalar(shortcut_relative)<=.05
    scientific = bool(primary['denominators_resolved'] and trace_resolved and shortcut_ok
        and primary['diagonal_error_ratio']<=.75 and primary['plane_error_ratio']<=.75)
    primary.update(scientific_passed=scientific,
        outcome='projection_proxy_pass' if scientific else 'projection_proxy_negative'
            if primary['denominators_resolved'] and trace_resolved else 'roundoff_unresolved',
        sketch_unused_query_columns=[384,1023],selected_response_rows=int(fnp.sum(rp)),
        selected_live_columns=int(fnp.sum(lp)),not_independent_statistical_queries=True,
        shortcut_direction_qualified=shortcut_ok)
    answer = dict(primary=primary,full_support=full,trace=dict(
        exact_dense_stored_QF=True,orthogonal_shortcut_is_not_exact=True,
        exact_direction_squared_norm=scalar(delta_norm2),roundoff_floor_squared=scalar(trace_floor),
        shortcut_discrepancy_squared_norm=scalar(shortcut_norm2),
        shortcut_relative_discrepancy=scalar(shortcut_relative) if trace_resolved else None,
        shortcut_absolute_norm_bound=scalar(shortcut_bound),Q_orthogonality_Frobenius=scalar(epsilon_Q),
        signed_tensor_absolute_scale=scalar(S),fixed_roundoff_gamma=scalar(gamma),
        active_support=a,all_old_columns=R,identity_star_denominator=a+2),
        scientific_passed=scientific,no_raw_or_J_measurement=True,no_nonlinear_intervention=True)
    remember(refs,locals())
    return answer


def evaluate(context,c,np,flops,fnp,refs):
    """Returns only detached scalar metadata; every allocation is in caller ctx."""
    raw = Path(context['manifest']['packet_file']).read_bytes()
    require(digest(raw)==context['manifest']['packet_sha256'], 'Unchanged sealed packet bytes')
    weights = np.load(context['fixture'],mmap_mode='r',allow_pickle=False)
    require(weights.shape==(16,1024,1024) and weights.dtype==np.float32, 'Physical weights representation')
    module = types.ModuleType('join_trace_private_c2')
    refs.append(weakref.ref(module))
    module.__file__='<unchanged-c2-with-zero-F-join-observer>'
    observed = {}
    def capture(Q,F):
        require(not observed, 'Exactly one layer8 join observation')
        observed.update(Q=Q,F=F)  # Borrowed references only, no numerical work.
    module._jtrace_capture = capture
    try:
        for key in ('estimator','setup','reader'):
            exec(compile(context['source'][key],module.__file__+':'+key,'exec'),vars(module))
        require(module._SCRATCH is None, 'Fresh owned numerical module/setup')
        module._jtrace_setup()
        packets = context['manifest']['packets']; entry = packets['entry8']; pre = packets['pre8']
        restored = {key:c.restore(entry[key],raw,np,fnp,refs,'entry8.'+key) for key in ENTRY_KEYS}
        response,live = [fixed_query(pre[key]['array'],raw,np,fnp,refs) for key in QUERY_KEYS]
        w32 = fnp.array(weights[8],dtype=fnp.float32,copy=True)
        require(bool(fnp.all(fnp.isfinite(w32))) and w32.tobytes()==weights[8].tobytes(),
                'Paid exact finite physical W8 input')
        pp,cp,diagonal = module._jtrace_reader(8,1024,w32,restored['p'],restored['previous'],
            restored['newborn'],restored['lam'],restored['shared_old'],response,live)
        require(set(observed)=={'Q','F'} and observed['F'].shape==(384,8128)
            and restored['shared_old']['columns']==8128, 'Actual original join selected every old prefix column')
        control = {}
        for name,value in (('cp_pre',cp),('diagonal',diagonal)):
            row = pre[name]['array']; copy = fnp.copy(value)
            require(bool(fnp.all(fnp.isfinite(copy))), 'Paid finite current CP control')
            data = copy.tobytes(); lo = row['offset']
            require(list(copy.shape)==row['shape'] and str(copy.dtype)==row['dtype']
                and digest(data)==row['sha256'] and data==raw[lo:lo+row['bytes']],
                'BYTE-EXACT archived current pre8 '+name+' BEFORE trace diagnostics')
            control[name]=dict(sha256=digest(data),bytes=len(data),byte_exact=True)
            refs.append(weakref.ref(copy)); del copy,data
        remember(refs,(pp,cp,diagonal))
        del pp,cp,diagonal,value,weights
        # Checkpoints above are comparison-only; no captured plane becomes a
        # tensor operand. Original full p defines the pre-join source tensor.
        answer = projection_diagnostics(fnp,
            restored['p'][restored['previous'],:8128],observed['Q'],observed['F'],
            restored['lam'][:8128],w32[restored['previous'],:],response,live,refs)
        answer.update(current_CP_reader_byte_exact=True,current_CP_control=control,
            source_proof=context['source_proof'],cost_certificate=context['cost_certificate'],
            captured_selectors_are_fixed_queries_only=True,truth_accessed=False,score_measured=False)
        return answer
    finally:
        remember(refs,vars(module)); remember(refs,locals())
        module.__dict__.clear(); observed.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    for key in KEYS: parser.add_argument('--'+key+'-sha256')
    args = parser.parse_args()
    if not args.run: parser.print_help(); return 0
    require(not OUTPUT.exists() and not HERE.is_symlink(), 'Exclusive never-resumed source-selected stage')
    began = time.monotonic(); OUTPUT.mkdir()
    result = dict(schema=SCHEMA,passed=False,completed=False,candidate_sha256=CANDIDATE_SHA,
        rank=35,layer=8,roster=['trace'],retired_call_count=0,full_model_calls=0,
        partial_model_calls=0,parameter_updates=0,source_bindings={},input_bindings={},artifacts={},
        truth_accessed=False,score_measured=False,automatic_next_stage=False,
        scientific_passed=False,no_deployable_candidate=True)
    def tick():
        require(time.monotonic()-began<CAP and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<RSS,
                'Whole worker120s/8GB, external watch additionally owns all descendants/swap')
    try:
        context = admission(args,tick)
        for key in ('source_bindings','input_bindings','capture_seals','packet_proof','source_proof','cost_certificate'):
            result[key] = context[key]
        c,a,h,backend,legacy = helpers()
        env = dict(time=time,hashlib=hashlib,json=json,require=require)
        exec(compile(context['source']['ledger'],'<original-closed-ledger-metrics>','exec'),env)
        metrics = env['_v2_ledger_metrics'](legacy.metrics)
        backend.backend_select(); h.environment()
        write(OUTPUT/'started.json',result)
        import numpy as np
        import flopscope as flops
        import flopscope.numpy as fnp
        from threadpoolctl import threadpool_limits,threadpool_info
        flops.configure(symmetry_warnings=False)
        refs = []
        with threadpool_limits(limits=1,user_api='blas'):
            result['threadpools'] = backend.backend_threadpool_info(threadpool_info,result,tick)()
            started = time.monotonic()
            with flops.BudgetContext(flop_budget=FLOP_CAP,wall_time_limit_s=120,quiet=True) as ctx:
                diagnostic = evaluate(context,c,np,flops,fnp,refs)
            bill = dict(call=0,rank=35,arm='trace',**metrics(ctx,started),
                complete_research_flops=int(ctx.flops_used),includes_initialization=True,
                includes_all_input_diagnostic_copy_finite_work=True,not_a_complete_estimator_score=True)
            require(0<bill['flops']==bill['complete_research_flops']
                <=context['cost_certificate']['complete_upper_flops']<=FLOP_CAP
                and bill['wall_s']<120 and bill['context_wall_s']<120
                and bill['peak_rss_bytes']<RSS and all(math.isfinite(bill[k]) and bill[k]>=0
                    for k in legacy.TIMINGS), 'Actual whole owned bill/resources, no component-only discount')
            bill_path = OUTPUT/'call_00_bill.json'; write(bill_path,bill)
            ctx_ref = weakref.ref(ctx)
            life = legacy.reset_saved_history(flops,ctx,bill_path)
            del ctx; gc.collect()
            require(ctx_ref() is None and all(ref() is None for ref in refs)
                and life['closed_context_bill_unchanged_by_reset'], 'Every owned numerical object/context retired')
            life.update(call=0,rank=35,arm='trace',context_released=True,
                all_numerical_array_owners_retired=True,all_parameter_and_estimator_owners_retired=True)
        diagnostic.update(call=0,rank=35,arm='trace')
        write(OUTPUT/'call_00_lifecycle.json',life)
        write(OUTPUT/'call_00_diagnostics.json',diagnostic)
        complete = dict(call=0,rank=35,arm='trace',passed=True,completed=True,context_retired=True,
            all_parameter_and_estimator_owners_retired=True,
            bill_sha256=sha(bill_path,tick),lifecycle_sha256=sha(OUTPUT/'call_00_lifecycle.json',tick),
            diagnostic_sha256=sha(OUTPUT/'call_00_diagnostics.json',tick))
        write(OUTPUT/'call_00_complete.json',complete)
        result.update(retired_call_count=1,research_flops=bill['complete_research_flops'],
            primary=diagnostic['primary'],scientific_passed=diagnostic['scientific_passed'],
            current_CP_reader_byte_exact=True,own_complete_bills=True,
            all_numerical_array_owners_retired=True,all_parameter_and_estimator_owners_retired=True)
        print(json.dumps(dict(event='retired',call=0,rank=35,arm='trace',
            complete_sha256=sha(OUTPUT/'call_00_complete.json',tick))),flush=True)
        for name in ('started.json','call_00_bill.json','call_00_lifecycle.json',
                     'call_00_diagnostics.json','call_00_complete.json'):
            result['artifacts'][str(OUTPUT/name)] = sha(OUTPUT/name,tick)
        backend.backend_origins(result,tick)
        bind(result['source_bindings'],tick); bind(result['input_bindings'],tick)
        result.update(hashes_unchanged=True,sources_unchanged=True,passed=True,completed=True)
        tick()
    except BaseException:
        result.update(passed=False,completed=False,error=traceback.format_exc())
    result.update(elapsed_s=time.monotonic()-began,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if result['passed']: tick()
    write(OUTPUT/'result.json',result)
    end = dict(passed=result['passed'],completed=result['completed'],result_sha256=sha(OUTPUT/'result.json'),
        elapsed_s=time.monotonic()-began,finished_utc=datetime.now(timezone.utc).isoformat())
    if result['passed']: tick()
    write(OUTPUT/'completion.json',end)
    if result['passed']: tick()
    write(OUTPUT/'finalization.json',dict(passed=result['passed'],completed=result['completed'],
        result_sha256=end['result_sha256'],completion_sha256=sha(OUTPUT/'completion.json'),
        elapsed_s=time.monotonic()-began,finished_utc=datetime.now(timezone.utc).isoformat()))
    if result['passed']: tick()
    print(json.dumps(dict(passed=result['passed'],scientific_passed=result['scientific_passed'],
        retired=result['retired_call_count'],error=result.get('error'))),flush=True)
    return 0 if result['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
