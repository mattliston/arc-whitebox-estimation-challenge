"""DRAFT: fixed sealed current-c2 L8/L9 information analysis; inert without --run.

No estimator, fit, new sample, strength choice, or subsequent-stage authority.
All numerical array work is charged in one native FNP context. Byte/header
inspection outside it is representation only, after six retired consumers.
"""
import argparse
import ast
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import struct
import sys
import time
import traceback
import weakref
import zipfile

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
SHARED = NEW/'current_comp_shared_old384_v1'
OUTPUT = HERE/'analysis_v1'
SCHEMA = 'current_comp_joint_cell_analysis_v1'
ROSTER = ('tiny','N','T','Q','TQ','ALL-local')
CAP, RSS = 60., 1_000_000_000
ANALYSIS_FLOP_CAP = 1_000_000_000
N, WIDTH = 8_000_000, 1024
CANDIDATE_SHA = 'c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea'
DESIGN_SHA = '360add35b70970cba3af6181e7e786ceb28187c89a07813cba8a4143823a6f31'
CONSUMER_V1_SHA = 'cb9649482c362a9b13a0ebe3e57cf536ac5657cd970defc18ab09f659967e2bb'
KEYS = ('source','protocol','watcher','watch-spec','consumer-source','consumer-protocol',
        'consumer-watch-source','consumer-watch-spec','consumer-result','consumer-watch')
PINS = {
    HERE/'DESIGN.md': DESIGN_SHA,
    HERE/'consumer.py': CONSUMER_V1_SHA,
    HERE/'consumer_v2.py': 'c57d149127a6e23ba5403fc2c260f765327b63e2a36bb84ff06915e65b512a82',
    HERE/'watch_consumer_v2.py': 'adc96cd270d4a6edc38c02047af66d33110839eff7ba96c629572aa98c0b2015',
    HERE/'CONSUMER_PROTOCOL_V2.md': 'b07144d1e4560ba62e8455ecaff297e40e52bc0b47ae30f84d5eb23696feae53',
    HERE/'CONSUMER_WATCH_SPEC_V2.md': '4157f3aa16b25cef8fbd05ca00202a6a29f6cc27b8051739a360d5ffcce23ccc',
    SHARED/'probe.py': '6193396f83150e7e95f3c16ad5095eeb8d4c62da17d69d5bd7dcb08fad9a743b',
    SHARED/'backend_probe.py': '36d9068c8dead7f32b53da9c5c5382f9004a8a45c0c47c29d7e1f4b2435fdcc8',
    SHARED/'check4_v2.py': '8d09d9b1f443e229b1641d277312b52c204392648b25445ffab63b82dba16058',
    NEW/'birth_direction_fit_v1/native_v2.py': '8a1b888be28a7ba6d7f8e020bb9b1f2367786e189d952364710b34517ca2b35b',
    NEW/'covariance_replica_truth_v1/rank_0035_8M_v1.json':
        '133e3720488415b2a86790466fa4f8475352b6a2de7526830c596fbe794b303a',
    P2/'public_g19_attribution_v1/mc_truth_mps_indep.py':
        '5b887ec3ef484f1e715273a375dd4358fa57165f9a955b9061c6c5a228087cb5',
}
A = P2/'public_g19_attribution_v1/mps_cov2_rank0035.npz'
B = NEW/'covariance_replica_truth_v1/rank_0035_8M_v1.npz'
MOMENTS = {A:'7e6b97166885de0cdd870a835ae536830fb6da41ca345d8a66e2eb8d0d6380e6',
           B:'e6a7787d974c6b56f39a4444bd54c793ff3efa6e62acb55054774ddb2a2e3358'}
FIXED_CHAINS = {
    'capture_v1': ('7867fb6b645d042d88fb359d7f831708d2e7feb37e26d845309805aec4796804',
        '0bdb594c411d07ffe9c1fd5c21f394c51d8d558d147aa18752e4067b75021c54',
        '5e5589cbb3fa91d8ef8c3d6dae6587e68c4f174283c0e0e3ec049a44f15cbb78'),
    'capture_watch_v1': ('790b0d8ad7e88df9aa258d16de35ec6f5638c3f202f3a0cae51f46812abb9a34',
        '4fd68816a97ae3fb3ee905923fc55eb2fb258eaa531e6359ae927c893551f04e',
        'aa5e395b811694fdbbedee8141cf62238aef6e017fb1c077b4a4cb12d6c703ed'),
    'consumer_v1': ('6b7e1332fc4d3603ffee80db7d610209c334b6cdad9fc6cdebc60af333f74476',
        '9f6e8e0c79d6ecd8e47e843768d3b5078618617fabe7f8adade6dee90f892aee',
        '292a252779923fbe32fcb461b56e6238c6b69187ab864e25ebecb7df8bb812b0'),
    'consumer_watch_v1': ('be3bf99d91ba57c7af4d1ac46abf4249cee971a9dea1ed821de4a783195494cc',
        'e4683a00a67f706b644ccd2cea6405a8c5bb29724898f0ebc8db35d3a57d1200',
        '2494420ab4b4c094fca20ac2cdb9f7bf46fa8a7ea090bcc12d45fa7a1d5479b4'),
}
BASE_NAMES = frozenset(('pair8.m','pair8.C','postP8.m','postP8.C','postP8.host','postP8.dm',
    'end8.C','pair9.m','postP9.m','postP9.host','postP9.dm'))


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path, tick=None, *, moments=False):
    path = Path(path)
    require(not path.is_symlink(), 'No symlink owner')
    path = path.resolve()
    require(path.is_relative_to(P2) and path.name != 'key.txt', 'Local noncredential owner')
    require(path != B or moments, 'B cannot be opened before the pre-evaluation seal')
    if path.suffix in ('.npy','.npz') and ('truth' in str(path).lower() or 'truthpost' in path.parts):
        require(moments and path == B, 'Only the single fixed B archive is admitted')
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1 << 20),b''):
            h.update(block)
            if tick is not None: tick()
    return h.hexdigest()


def read(path):
    path = Path(path)
    require(not path.is_symlink() and path.stat().st_size < 16*1024**2, 'Bounded metadata only')
    return json.loads(path.read_text())


def write(path, value):
    def safe(x):
        if isinstance(x,float) and not math.isfinite(x): return {'nonfinite_float':repr(x)}
        if isinstance(x,dict): return {str(k):safe(v) for k,v in x.items()}
        if isinstance(x,(tuple,list)): return [safe(v) for v in x]
        return x
    with Path(path).open('x') as f:
        json.dump(safe(value),f,sort_keys=True,indent=2,allow_nan=False); f.write('\n')


def merge(target, values):
    require(isinstance(values,dict), 'Flat direct binding map')
    for p,d in values.items():
        require(re.fullmatch('[0-9a-f]{64}',d or '') and (p not in target or target[p]==d),
                'Consistent actual immutable binding')
        target[p]=d


def bind(values, tick=None, *, moments=False):
    for p,d in values.items(): require(sha(p,tick,moments=moments)==d,'Fresh file: '+p)


def chain(folder, expected, inputs, tick, positive=True):
    paths=[HERE/folder/n for n in ('result.json','completion.json','finalization.json')]
    actual=tuple(sha(p,tick) for p in paths)
    require(actual[0]==expected if isinstance(expected,str) else actual==tuple(expected),
            'Explicit actual three-seal boundary: '+folder)
    r,e,f=map(read,paths)
    require(all(x.get('passed') is positive and x.get('completed') is positive for x in (r,e,f))
        and e['result_sha256']==f['result_sha256']==actual[0]
        and f['completion_sha256']==actual[1], 'Actual linked seals, not a result-only PASS')
    merge(inputs,{str(p):d for p,d in zip(paths,actual)})
    return r,e,f,actual


def ledger(bill):
    x=bill['operation_ledger']
    require(x['schema']=='closed_operation_ledger_v1' and x['after_context_exit']
        and x['unchanged_closed_context_bill'] and x['no_array_access_or_candidate_mutation']
        and x['total_flops']==bill['flops'] and x['operation_count']==bill['operation_count']
        and sum(v['flops'] for v in x['by_operation'])==bill['flops']
        and sum(v['count'] for v in x['by_operation'])==bill['operation_count'],
        'Actual complete closed context ledger, never a borrowed estimator bill')


def admission(args, tick=None):
    """Stdlib only. No A member or B archive is opened by this admission."""
    require(not any(k.split('.')[0] in ('numpy','flopscope','torch','threadpoolctl')
        for k in sys.modules), 'No numerical import before retired-evidence admission')
    require(not os.environ.get('FLOPSCOPE_WEIGHTS_FILE') and not os.environ.get('FLOPSCOPE_DISABLE_WEIGHTS'),
            'Original complete dtype/operator tariffs, no billing override')
    required={k:getattr(args,k.replace('-','_')+'_sha256',None) for k in KEYS}
    require(all(re.fullmatch('[0-9a-f]{64}',v or '') for v in required.values()), 'Ten reviewed/actual SHA arguments')
    source={str(p):d for p,d in PINS.items()}
    for filename,key in (('analysis.py','source'),('ANALYSIS_PROTOCOL.md','protocol'),
        ('watch_analysis.py','watcher'),('ANALYSIS_WATCH_SPEC.md','watch-spec'),
        ('consumer_v2.py','consumer-source'),('CONSUMER_PROTOCOL_V2.md','consumer-protocol'),
        ('watch_consumer_v2.py','consumer-watch-source'),('CONSUMER_WATCH_SPEC_V2.md','consumer-watch-spec')):
        merge(source,{str(HERE/filename):required[key]})
    bind(source,tick); inputs={}
    provenance=read(NEW/'covariance_replica_truth_v1/rank_0035_8M_v1.json')
    require(provenance['rank']==35 and provenance['samples']==N and provenance['batch']==131072
        and provenance['seed_xor']=='0x4153545241434f56'
        and provenance['rng']=='PCG64DXSM CPU float32 standard_normal'
        and provenance['source_sha256']==PINS[P2/'public_g19_attribution_v1/mc_truth_mps_indep.py']
        and provenance['output_sha256']==MOMENTS[B], 'Existing B source/noise frame, no invented precision authority')
    r,e,f,seals=chain('consumer_v2',required['consumer-result'],inputs,tick)
    w,we,wf,wseals=chain('consumer_watch_v2',required['consumer-watch'],inputs,tick)
    require(r['schema']=='current_comp_joint_cell_consumer_v2'
        and w['schema']=='current_comp_joint_cell_consumer_watch_v2'
        and r['candidate_sha256']==w['candidate_sha256']==CANDIDATE_SHA
        and r['roster']==w['roster']==list(ROSTER) and r['ranks']==w['ranks']==[35]*6
        and r['retired_call_count']==w['retired_calls']==6
        and [w['child_'+k] for k in ('result_sha256','completion_sha256','finalization_sha256')]==list(seals),
        'Actual fixed positive six-arm worker and independent watch')
    require(all(r[k] is True for k in ('tiny_qualified','null_byte_exact','own_complete_bills',
        'all_parameter_and_estimator_owners_retired','supply_A_decoded_after_null_retirement',
        'all_outputs_sealed_before_evaluation_B','hashes_unchanged','sources_unchanged'))
        and r['evaluation_B_accessed'] is False and r['score_measured'] is False
        and r['parameter_updates']==r['full_model_calls']==0 and r['partial_model_calls']==5
        and r['tiny_contexts']==1 and r['current_cell_centering_error_bound_available'] is False,
        'Fresh tiny/null/source/domain stage; no B leakage or existing scientific result')
    require(all(w[k] is True for k in ('whole_owned_tree_retired','cleanup_passed',
        'observational_groups_retired','actual_completed_samples','hashes_unchanged','sources_unchanged'))
        and 0<w['maximum_combined_rss_bytes']<8_000_000_000 and 0<w['maximum_interval_s']<=1
        and w['maximum_swap_growth_bytes']==0 and 0<wf['elapsed_s']<600
        and w['child_execution_completed'] and w['post_observation_memory_sample_claimed'] is False
        and 0<=wf['elapsed_s']-w['last_completed_observation_elapsed_s']<=1,
        'Actual completed full-tree resource qualification and retirement')
    for record in (r,w):
        merge(source,record['source_bindings'])
        merge(inputs,record['input_bindings']); merge(inputs,record['artifacts'])
    # Explicit source arguments must be the actual sources in BOTH positive records.
    for filename,key in (('consumer_v2.py','consumer-source'),('watch_consumer_v2.py','consumer-watch-source'),
        ('CONSUMER_PROTOCOL_V2.md','consumer-protocol'),('CONSUMER_WATCH_SPEC_V2.md','consumer-watch-spec')):
        require(all(x['source_bindings'].get(str(HERE/filename))==required[key] for x in (r,w)),
                'No unrelated positive record/source join')
    fixed={}
    for folder,digests in FIXED_CHAINS.items():
        fixed[folder]=chain(folder,digests,inputs,tick,positive=not folder.startswith('consumer'))
    old,oldw=fixed['consumer_v1'][0],fixed['consumer_watch_v1'][0]
    require(old['retired_call_count']==oldw['retired_calls']==2 and old['tiny_qualified'] and old['null_byte_exact']
        and old['evaluation_B_accessed'] is False and 'Fixed original rank/P/layer frame' in old['error'],
        'The actual negative pre-intervention attempt remains negative')
    prior_flops=sum(int(x['complete_research_flops']) for x in old['trials'])
    require(prior_flops==61_060_590_585 and int(old['trials'][0]['flops'])==713_596,
            'Failed tiny/null work remains in the unchanged aggregate')
    merge(inputs,old['artifacts'])
    bills=[]; diagnostics=[]; readouts=[]
    require(len(r['trials'])==len(r['diagnostic_records'])==6, 'All fixed arm metadata')
    for i,arm in enumerate(ROSTER):
        paths={k:HERE/'consumer_v2'/f'call_{i:02d}_{k}.json'
               for k in ('bill','lifecycle','diagnostics','complete')}
        require(all(r['artifacts'].get(str(p))==sha(p,tick) for p in paths.values()), 'Each ordinal fully sealed')
        bill,life,diag,done=[read(paths[k]) for k in ('bill','lifecycle','diagnostics','complete')]
        require(all(x['call']==i and x['rank']==35 and x['arm']==arm for x in (bill,life,diag,done))
            and done['passed'] and done['completed'] and done['context_retired']
            and done['all_parameter_and_estimator_owners_retired']
            and done['bill_sha256']==sha(paths['bill']) and done['lifecycle_sha256']==sha(paths['lifecycle'])
            and done['diagnostic_sha256']==sha(paths['diagnostics'])
            and life['context_released'] and life['closed_context_bill_unchanged_by_reset']
            and life['all_parameter_and_estimator_owners_retired'] and r['trials'][i]==bill
            and r['diagnostic_records'][i]==dict(path=str(paths['diagnostics']),sha256=sha(paths['diagnostics'])),
            'Actual ordered bills, input ownership and complete releases')
        ledger(bill)
        require(int(bill['flops'])+int(bill['input_flops'])==int(bill['complete_research_flops'])
            and 0<bill['wall_s']<120 and 0<bill['context_wall_s']<120,
            'Measured physical and input costs remain separate and complete')
        bills.append(bill); diagnostics.append(diag)
        if i==0:
            require(diag['tiny']['passed'] and 0<bill['flops']<=10_000_000, 'Fresh complete tiny qualification')
            continue
        local=arm=='ALL-local'; domain=diag['domains']; checkpoint=diag['checkpoints']
        require(domain['variance_count']==(1 if local else 2) and domain['selector_count']==(0 if local else 5)
            and all(v['finite'] and v['minimum']>1e-10 and v['below_floor_count']==0
                    for v in domain['variance_domain'].values())
            and checkpoint['all_explicit_arrays_released'] and checkpoint['head7_pending_never_loaded']
            and diag['evaluation_B_accessed'] is False, 'Actual fixed unfloored local consumers')
        if arm=='N': require(checkpoint['null_byte_exact'],'The new N, not old partial eligibility')
        else:
            require(diag['supply']['strength']==1. and diag['supply']['ordered_Q31_not_symmetrized']
                and diag['supply']['independent_marginal_diagonals_installed']
                and diag['supply']['centering_error_bound_available'] is False,
                'Unchanged physical natural-strength supply and honest precision limits')
        rr=diag['readouts']; path=Path(rr['path'])
        require(path==HERE/'consumer_v2'/f'call_{i:02d}_readouts.bin'
            and r['artifacts'].get(str(path))==rr['sha256']==sha(path,tick)
            and path.stat().st_size==rr['bytes']<40*1024**2, 'Actual detached readout bytes')
        expected=BASE_NAMES if not local else frozenset(k for k in BASE_NAMES if k.startswith(('pair8.','postP8.')))
        require(set(rr['arrays'])==expected, 'Exact fixed output names; no invented ALL carried covariance')
        cursor=0
        for name,row in sorted(rr['arrays'].items(),key=lambda item:item[1]['offset']):
            shape=[1024,1024] if name.endswith('.C') else [1024]
            require(row['shape']==shape and row['dtype'] in ('float32','float64')
                and row['file_storage_order']=='C' and row['byteorder']==sys.byteorder
                and row['offset']==cursor and row['bytes']==math.prod(shape)*(4 if row['dtype']=='float32' else 8),
                'Dense typed output representation, no unexplained gaps')
            cursor+=row['bytes']
        require(cursor==rr['bytes'] and bill['output']==str(HERE/'consumer_v2'/f'call_{i:02d}_served.npy')
            and r['artifacts'].get(bill['output'])==bill['output_sha256'],
                'Every readout byte and served output was already sealed')
        readouts.append(dict(call=i,arm=arm,readouts=rr,served=bill['output'],served_sha256=bill['output_sha256']))
    total=sum(int(x['complete_research_flops']) for x in bills)
    require(total==r['complete_consumer_flops'] and 0<total and
        all(0<x['complete_research_flops']<=250_000_000_000 for x in bills[1:5])
        and bills[5]['complete_research_flops']<=5_000_000_000, 'Original per-arm upper bills')
    capture=fixed['capture_v1'][0]
    require(capture['candidate_sha256']==CANDIDATE_SHA and capture['truth_accessed'] is False
        and all(capture[k] is True for k in ('all16_domains_qualified','all47_selectors_qualified',
            'all_original_operations_exact','baseline_output_byte_exact','all_capture_arrays_released',
            'all_parameter_and_estimator_owners_retired')), 'Original all16/all47 complete exact-current capture')
    capture_flops=int(capture['trials'][0]['flops'])
    require(capture_flops==403_695_296_665, 'Actual complete metered capture included')
    tiny_flops=int(bills[0]['complete_research_flops'])+int(old['trials'][0]['complete_research_flops'])
    prior_total=capture_flops+prior_flops+total
    require(prior_total==r['capture_plus_all_consumer_flops']==710_007_601_555
        and r['prior_failed_consumer_flops']==prior_flops
        and r['source_proof']['derived_sha256']=='dcb9a89d163bbe552269771e000642b904550de24f5827eaf84c0b5f143e9cbd'
        and r['source_proof']['actual_predict_binary_identity_preserved']
        and r['source_proof']['full_P_NODE_BD']
        and r['local_source_proof']['derived_sha256']=='d4088c0d329e1029817ad4bf6bf60683a89fdd9c91321b57d8cef7251aad80ab'
        and r['local_source_proof']['ALL_local_pair_and_P_only'], 'Actual frozen numerical-source and cost lineage')
    require(prior_total+ANALYSIS_FLOP_CAP<=1_425_000_000_000
        and tiny_flops+ANALYSIS_FLOP_CAP<=10_000_000_000, 'All attempted work and remaining tiny-plus-analysis allowance')
    # Reserve the original separate metadata allowance once; nested worker
    # times are not double-counted with their supervisors' whole-stage times.
    prior_elapsed=60.+sum(float(fixed[k][2]['elapsed_s']) for k in ('capture_watch_v1','consumer_watch_v1'))+float(wf['elapsed_s'])
    require(prior_elapsed+CAP<900., 'Original total serial envelope, including failed attempt and metadata reserve')
    new_arrays={p:Path(p).stat().st_size for group in (capture['artifacts'],old['artifacts'],r['artifacts'])
                for p in group if Path(p).suffix in ('.bin','.npy','.npz')}
    require(sum(new_arrays.values())==r['aggregate_saved_numerical_bytes']==494_273_728
        and sum(new_arrays.values())+16*1024**2<1024**3,
        'All attempted saved arrays and new vectors below1GiB')
    # At most27 mean reports (one n-by-n matvec each),42 matrix-error
    # reports,9 cached baseline matrix errors,22 copied n-by-n inputs,
    # both unbiased covariance scalings, full finite scans and all exports.
    # 640*n^2 includes a factor2 allowance for every FP64 operation;
    # 8192*n+1MF covers vectors/scalars/metadata-sized FNP constants.
    source_cost_bound=640*WIDTH**2+8192*WIDTH+1_000_000
    require(source_cost_bound<=ANALYSIS_FLOP_CAP,'Complete analysis constructor/read/reduction source bound')
    require(inputs.get(str(A))==MOMENTS[A], 'Analysis A covariance is exactly the previously supplied archive')
    bind(source,tick); bind(inputs,tick)
    return dict(source_bindings=source,input_bindings=inputs,consumer_seals=list(seals),
        consumer_watch_seals=list(wseals),capture_and_negative_seals=FIXED_CHAINS,
        readouts=readouts,prior_total_flops=prior_total,prior_tiny_flops=tiny_flops,
        prior_elapsed_s_including_metadata_reserve=prior_elapsed,prior_saved_array_bytes=sum(new_arrays.values()),
        source_cost_upper_flops=source_cost_bound,
        negative_v1_preserved=True,all_six_retired_before_evaluation=True,
        current_cell_centering_error_bound_available=False,authenticated_joint_noise_replica_available=False,
        candidate_sha256=CANDIDATE_SHA)


def npy_header(stream):
    require(stream.read(6)==b'\x93NUMPY','Actual plain NPY representation')
    version=tuple(stream.read(2)); require(version in ((1,0),(2,0),(3,0)),'NPY format')
    length=struct.unpack('<H' if version==(1,0) else '<I',stream.read(2 if version==(1,0) else 4))[0]
    require(0<length<=65536,'Bounded typed literal header')
    h=ast.literal_eval(stream.read(length).decode('utf8' if version==(3,0) else 'latin1'))
    require(set(h)=={'descr','fortran_order','shape'} and h['descr'] in ('<f4','<f8','<i4','<i8')
        and h['fortran_order'] is False and isinstance(h['shape'],tuple), 'Plain fixed C-order dtype')
    return h


def archive_member(path, member, layers, tick):
    """Decode bytes only for explicit member/layers, not the surrounding moments."""
    allowed={A:{'mc_post_cov':(8,)},B:{'mc_post':(8,9),'mc_post_cov':(8,9)}}
    require(path in allowed and allowed[path].get(member)==tuple(layers),'Only declared A/B post8/9 members')
    with zipfile.ZipFile(path) as z:
        require(len(z.namelist())==len(set(z.namelist())),'Unique NPZ names')
        for key,value in (('P',N),('rank',35)):
            with z.open(key+'.npy') as stream:
                h=npy_header(stream)
                require(h['shape']==() and h['descr'] in ('<i4','<i8'),'Actual scalar frame metadata')
                raw=stream.read(); require(len(raw)==int(h['descr'][-1]),'Scalar metadata length')
                require(struct.unpack('<i' if len(raw)==4 else '<q',raw)[0]==value,'Actual archived P/rank')
        with z.open(member+'.npy') as stream:
            h=npy_header(stream)
            covariance=member=='mc_post_cov'; shape=(16,WIDTH,WIDTH) if covariance else (16,WIDTH)
            require(h['shape']==shape and h['descr']==('<f4' if covariance else '<f8'),
                    'Original stored precision and fixed network frame')
            size=math.prod(shape[1:])*int(h['descr'][-1]); saved={}
            # Reading/skipping compression bytes is representation, not a
            # numerical reduction. Only two (or one) declared slices survive.
            for layer in range(max(layers)+1):
                parts=[]; remain=size
                while remain:
                    block=stream.read(min(remain,1<<20)); require(bool(block),'Complete selected NPY slab')
                    if layer in layers: parts.append(block)
                    remain-=len(block); tick()
                if layer in layers:
                    saved[layer]=dict(data=b''.join(parts),dtype=h['descr'],shape=list(shape[1:]))
            return saved


def load_readout_bytes(row, tick):
    path=Path(row['readouts']['path']); raw=path.read_bytes(); tick()
    require(hashlib.sha256(raw).hexdigest()==row['readouts']['sha256'],'Already sealed readout file')
    out={}
    for name,meta in row['readouts']['arrays'].items():
        block=raw[meta['offset']:meta['offset']+meta['bytes']]
        require(hashlib.sha256(block).hexdigest()==meta['sha256'],'Exact selected readout bytes')
        out[name]=dict(data=block,dtype='<f4' if meta['dtype']=='float32' else '<f8',shape=meta['shape'])
    require(sha(row['served'],tick)==row['served_sha256'],'Exact already sealed served file')
    with Path(row['served']).open('rb') as stream:
        h=npy_header(stream); data=stream.read()
    shape=(1 if row['arm']=='ALL-local' else 2,WIDTH)
    require(h['descr']=='<f4' and h['shape']==shape and len(data)==math.prod(shape)*4,
            'Actual stored FP32 served output, never an alternate re-projection')
    out['served']=dict(data=data,dtype='<f4',shape=list(shape))
    return out


def evaluate(fnp,np,ctx,context,raw_targets,tick):
    """All scalar/vector/matrix statistics, copies and finite scans are paid."""
    refs=[]; copies=finite=exports=0; blocks=[]; vector_manifest={}; vector_offset=0
    def array(raw):
        nonlocal copies,finite
        before=int(ctx.flops_used)
        view=np.frombuffer(raw['data'],dtype=raw['dtype']).reshape(raw['shape'])
        x=fnp.array(view,dtype=fnp.float64,order='C',copy=True); copies+=int(ctx.flops_used)-before
        refs.append(weakref.ref(x)); before=int(ctx.flops_used)
        require(bool(fnp.all(fnp.isfinite(x))),'Paid complete input finite scan')
        finite+=int(ctx.flops_used)-before
        return x
    def raw_scalar(x):
        require(x.size==1,'Exactly one native scalar statistic')
        return float(x.reshape((1,))[0])
    def scalar(x):
        require(bool(fnp.all(fnp.isfinite(x))),'Finite paid statistic, not repaired')
        return raw_scalar(x)
    def vector(name,x):
        nonlocal exports,vector_offset
        before=int(ctx.flops_used); copied=fnp.array(x,dtype=fnp.float64,order='C',copy=True)
        require(bool(fnp.all(fnp.isfinite(copied))),'Paid detached reported-vector copy/finite')
        exports+=int(ctx.flops_used)-before
        block=copied.tobytes(); refs.append(weakref.ref(copied)); del copied
        vector_manifest[name]=dict(offset=vector_offset,bytes=len(block),dtype='<f8',shape=[WIDTH],
            sha256=hashlib.sha256(block).hexdigest()); vector_offset+=len(block); blocks.append(block)
    # Every numerical constant is a typed one-element array; b/r or sample
    # factors are never accidentally converted into a remote scalar policy.
    n=fnp.array([WIDTH],dtype=fnp.float64); samples=fnp.array([N],dtype=fnp.float64)
    one=fnp.array([1.],dtype=fnp.float64); two=fnp.array([2.],dtype=fnp.float64)
    four=fnp.array([4.],dtype=fnp.float64); three=fnp.array([3.],dtype=fnp.float64)
    factor=samples/(samples-one)
    means={l:array(raw_targets['Bmean'][l]) for l in (8,9)}
    covB={l:array(raw_targets['Bcov'][l]) for l in (8,9)}
    S={l:covB[l]*factor for l in (8,9)}
    Acov=array(raw_targets['Acov'][8])
    noise={l:fnp.sum(fnp.diag(S[l]))/(samples*n) for l in (8,9)}
    require(all(scalar(noise[l])>=0 for l in (8,9)), 'Nonnegative marginal sampling variance, no clipping')
    raw_baseline=load_readout_bytes(context['readouts'][0],tick)
    baseline={k:array(v) for k,v in raw_baseline.items()}; del raw_baseline
    require(bool(fnp.all(baseline['served']>=0)), 'Paid nonnegative stored served baseline')
    covariance_targets={'A_stored_denominator_N':Acov,'B_stored_denominator_N':covB[8],
        'B_unbiased_denominator_N_minus_1':S[8]}
    stages=('pair8.C','postP8.C','end8.C'); base_cov={}
    for stage in stages:
        base_cov[stage]={}
        for name,target in covariance_targets.items():
            e=baseline[stage]-target; base_cov[stage][name]=fnp.sum(e*e)
    def mean_record(arm,kind,layer,b,q):
        e=means[layer]-b; delta=q-b; residual=means[layer]-q
        sum_b=fnp.sum(e*e); sum_q=fnp.sum(residual*residual)
        mse_b=sum_b/n; mse_q=sum_q/n
        signed=fnp.sum(delta*e); delta2=fnp.sum(delta*delta)
        D=(two*signed-delta2)/n
        quadratic=fnp.sum(delta*(S[layer]@delta))
        variance=four*quadratic/(samples*n*n)
        variance_finite=bool(fnp.all(fnp.isfinite(variance)))
        quadratic_finite=bool(fnp.all(fnp.isfinite(quadratic)))
        vb=raw_scalar(variance); mb=scalar(mse_b); nq=scalar(mse_q-noise[layer]); nb=scalar(mse_b-noise[layer])
        delta_mean=fnp.sum(delta)/n; error_mean=fnp.sum(e)/n
        cov=fnp.sum((delta-delta_mean)*(e-error_mean))/n
        name=f'{arm}.{kind}.L{layer}'
        for suffix,x in (('baseline',b),('arm',q),('delta',delta),('signed_error_B',e)):
            vector(name+'.'+suffix,x)
        se=fnp.sqrt(variance) if variance_finite and quadratic_finite and vb>=0 else None
        return dict(layer=layer,kind=kind,arm=arm,baseline_mse=mb,arm_mse=scalar(mse_q),
            observed_ratio=scalar(mse_q/mse_b) if mb>0 else None,
            signed_improvement_D=scalar(D),mse_difference_roundoff=scalar((mse_b-mse_q)-D),
            baseline_error_signed_mean=scalar(error_mean),delta_signed_mean=scalar(delta_mean),
            baseline_error_rms=scalar(fnp.sqrt(mse_b)),delta_rms=scalar(fnp.sqrt(delta2/n)),
            delta_error_covariance=scalar(cov),delta_dot_error=scalar(signed),delta_squared_norm=scalar(delta2),
            signed_full_covariance_quadratic=raw_scalar(quadratic),signed_variance_D=vb,
            variance_finite=variance_finite,quadratic_finite=quadratic_finite,
            variance_qualification=('nonfinite_inconclusive' if not (variance_finite and quadratic_finite)
                else 'positive' if vb>0 else 'zero_inconclusive' if vb==0 else 'negative_inconclusive'),
            standard_error=scalar(se) if se is not None else None,
            three_standard_errors=scalar(three*se) if se is not None else None,
            noise_mse=scalar(noise[layer]),unclipped_noise_corrected_baseline_mse=nb,
            unclipped_noise_corrected_arm_mse=nq,
            unclipped_noise_corrected_ratio=scalar((mse_q-noise[layer])/(mse_b-noise[layer])) if nb>0 else None,
            vector_prefix=name,sampling_yardstick_not_postselection_confidence=True)
    reports={}; primary=None
    for row in context['readouts']:
        arm=row['arm']; local=arm=='ALL-local'
        raw=load_readout_bytes(row,tick); values={k:array(v) for k,v in raw.items()}; del raw
        require(bool(fnp.all(values['served']>=0)), 'Paid finite/nonnegative actual served arm')
        means_report=[]; cov_report={}
        for layer in ((8,) if local else (8,9)):
            for kind,b,q in (('served_FP32',baseline['served'][layer-8],values['served'][layer-8]),
                ('host_unprojected',baseline[f'postP{layer}.host'],values[f'postP{layer}.host']),
                ('postP_unprojected',baseline[f'postP{layer}.m'],values[f'postP{layer}.m'])):
                means_report.append(mean_record(arm,kind,layer,b,q))
        for stage in (stages[:2] if local else stages):
            cov_report[stage]={}
            for name,target in covariance_targets.items():
                e=values[stage]-target; numerator=fnp.sum(e*e); denominator=base_cov[stage][name]
                d=scalar(denominator)
                cov_report[stage][name]=dict(squared_frobenius_error=scalar(numerator),
                    baseline_squared_frobenius_error=d,matrix_mse=scalar(numerator/(n*n)),
                    baseline_matrix_mse=scalar(denominator/(n*n)),
                    error_ratio=scalar(numerator/denominator) if d>0 else None,
                    signed_entry_mean_error=scalar(fnp.sum(e)/(n*n)),
                    covariance_error_noise_correction_available=False,standard_error_available=False)
        reports[arm]=dict(mean=means_report,covariance=cov_report,ALL_local_has_no_carried_state=local)
        if arm=='TQ':
            mean=next(x for x in means_report if x['layer']==9 and x['kind']=='served_FP32')
            cov=cov_report['end8.C']['B_unbiased_denominator_N_minus_1']
            valid=(mean['baseline_mse']>0 and cov['baseline_squared_frobenius_error']>0
                and mean['variance_finite'] and mean['quadratic_finite'] and mean['signed_variance_D']>0)
            witness=bool(valid and mean['observed_ratio']<=.75 and cov['error_ratio']<=.75 and
                mean['signed_improvement_D']>=mean['three_standard_errors']>0)
            primary=dict(arm='TQ',baseline='N',covariance_stage='end8.C',mean_stage='served_FP32_L9',
                covariance_error_ratio=cov['error_ratio'],served_mean_observed_ratio=mean['observed_ratio'],
                signed_improvement_D=mean['signed_improvement_D'],signed_variance_D=mean['signed_variance_D'],
                three_standard_errors=mean['three_standard_errors'],denominators_and_variance_qualified=valid,
                provisional_joint_supply_witness=witness,
                outcome='provisional_witness' if witness else 'negative' if valid else 'inconclusive')
        del values
        tick()
    require(primary is not None and len(blocks)*WIDTH*8==vector_offset<16*1024**2,'Fixed complete scalar/vector report')
    return (dict(reports=reports,primary=primary,sample_count=N,coordinate_count=WIDTH,
        covariance_targets=list(covariance_targets),unbiased_factor=scalar(factor),
        full_covariance_noise=True,covariance_joint_noise_standard_error_available=False,
        current_cell_centering_error_bound_available=False,original_A_execution_seal_available=False,
        B_independent_RNG_not_independent_precision=True,B_reused_development_data=True),b''.join(blocks),vector_manifest,
        dict(input_copy_flops=copies,input_finite_check_flops=finite,detached_vector_export_flops=exports),refs)


def source_helpers():
    """Only authenticated stdlib owners; no estimator module is imported."""
    def load(path,name):
        require(sha(path)==PINS[path], 'Pinned helper before import')
        spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module); return module
    backend=load(SHARED/'backend_probe.py','jcell_analysis_backend_owner')
    h=load(SHARED/'probe.py','jcell_analysis_environment_owner')
    legacy=load(NEW/'birth_direction_fit_v1/native_v2.py','jcell_analysis_lifecycle_owner')
    source=(SHARED/'check4_v2.py').read_text(); lines=source.splitlines(keepends=True)
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_v2_ledger_metrics')
    code=''.join(lines[node.lineno-1:node.end_lineno]); env=dict(time=time,hashlib=hashlib,json=json,require=require)
    exec(compile(code,'<exact-owned-closed-ledger>','exec'),env)
    return backend,h,legacy,env['_v2_ledger_metrics'](legacy.metrics),dict(
        ledger_owner_sha256=PINS[SHARED/'check4_v2.py'],ledger_function_sha256=hashlib.sha256(code.encode()).hexdigest(),
        environment_owner_sha256=PINS[SHARED/'probe.py'],
        original_closed_reset_owner_sha256=PINS[NEW/'birth_direction_fit_v1/native_v2.py'])


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--run',action='store_true')
    for key in KEYS: parser.add_argument('--'+key+'-sha256')
    args=parser.parse_args()
    if not args.run: parser.print_help(); return 0
    require(not OUTPUT.exists(),'Exclusive analysis output; no append/retry')
    began=time.monotonic(); OUTPUT.mkdir()
    result=dict(schema=SCHEMA,stage='analysis',passed=False,completed=False,candidate_sha256=CANDIDATE_SHA,
        rank=35,ranks=[35],layer=8,consumer_layer=9,roster=['analysis'],consumer_roster=list(ROSTER),
        source_bindings={},input_bindings={},artifacts={},
        evaluation_B_accessed=False,all_six_retired_before_evaluation=False,parameter_updates=0,
        full_model_calls=0,partial_model_calls=0,analysis_contexts=0,retired_call_count=0,
        automatic_next_stage=False,deployable_correction_demonstrated=False,raw_gain_measured=False,
        estimator_J_measured=False,score_measured=False,bank_or_submission_authorized=False)
    def tick():
        require(time.monotonic()-began<CAP and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<RSS,
                'Whole analysis60s/1GB; root supervisor owns whole tree/sample/swap limits')
    try:
        context=admission(args,tick)
        for key in ('source_bindings','input_bindings','consumer_seals','consumer_watch_seals',
                    'capture_and_negative_seals','prior_total_flops','prior_tiny_flops',
                    'prior_elapsed_s_including_metadata_reserve','prior_saved_array_bytes','source_cost_upper_flops',
                    'negative_v1_preserved','all_six_retired_before_evaluation'):
            result[key]=context[key]
        backend,h,legacy,metrics,proof=source_helpers(); result['numerical_owner_proof']=proof
        merge(result['source_bindings'],{str(p):d for p,d in backend._BP_PINS.items()})
        bind(result['source_bindings'],tick)
        # This exclusive seal precedes ANY evaluation member/header/B byte read.
        seal=OUTPUT/'pre_evaluation.json'
        write(seal,dict(passed=True,completed=True,consumer_seals=context['consumer_seals'],
            consumer_watch_seals=context['consumer_watch_seals'],source_bindings=result['source_bindings'],
            input_bindings=result['input_bindings'],all_six_retired_before_evaluation=True,
            evaluation_B_accessed=False,elapsed_s=time.monotonic()-began))
        result['artifacts'][str(seal)]=sha(seal,tick)
        backend.backend_select(); h.environment()
        require(not any(k.split('.')[0] in ('numpy','flopscope','torch','threadpoolctl') for k in sys.modules),
                'No pre-seal numerical imports')
        import numpy as np
        import flopscope as flops
        import flopscope.numpy as fnp
        from threadpoolctl import threadpool_limits,threadpool_info
        flops.configure(symmetry_warnings=False)
        with threadpool_limits(limits=1):
            result['threadpools']=backend.backend_threadpool_info(threadpool_info,result,tick)()
            bind({str(p):d for p,d in MOMENTS.items()},tick,moments=True)
            result['evaluation_B_accessed']=True
            raw_targets=dict(Acov=archive_member(A,'mc_post_cov',(8,),tick),
                Bmean=archive_member(B,'mc_post',(8,9),tick),Bcov=archive_member(B,'mc_post_cov',(8,9),tick))
            result['evaluation_bindings']={str(p):d for p,d in MOMENTS.items()}
            started=time.monotonic()
            with flops.BudgetContext(flop_budget=ANALYSIS_FLOP_CAP,quiet=True) as ctx:
                numerical,vectors,manifest,parts,refs=evaluate(fnp,np,ctx,context,raw_targets,tick)
            bill=metrics(ctx,started); ledger(bill)
            require(0<bill['flops']<=ANALYSIS_FLOP_CAP and all(math.isfinite(bill[k]) and bill[k]>=0 for k in legacy.TIMINGS)
                and bill['wall_s']<CAP and bill['context_wall_s']<CAP,'Actual complete analysis bill')
            bill.update(parts,call=0,rank=35,arm='analysis',complete_research_flops=int(bill['flops']),
                input_flops=0,all_numerical_work_in_single_owned_context=True,not_a_complete_estimator_score=True)
            bp=OUTPUT/'call_00_bill.json'; write(bp,bill)
            life=legacy.reset_saved_history(flops,ctx,bp); ref=weakref.ref(ctx)
            del ctx,raw_targets; gc.collect()
            require(ref() is None and all(r() is None for r in refs)
                and life['closed_context_bill_unchanged_by_reset'],'Every context/array owner actually retired')
            life.update(call=0,rank=35,arm='analysis',context_released=True,all_numerical_array_owners_retired=True,
                all_parameter_and_estimator_owners_retired=True)
            write(OUTPUT/'call_00_lifecycle.json',life)
            with (OUTPUT/'vectors.bin').open('xb') as stream: stream.write(vectors)
            vector_bytes=len(vectors); del vectors
            write(OUTPUT/'vectors.json',dict(path=str(OUTPUT/'vectors.bin'),bytes=vector_bytes,
                sha256=sha(OUTPUT/'vectors.bin'),arrays=manifest,stored_dtype='<f8',file_order='C'))
            write(OUTPUT/'statistics.json',numerical)
            write(OUTPUT/'call_00_diagnostics.json',dict(call=0,rank=35,arm='analysis',
                fixed_DESIGN_arithmetic=True,statistics_sha256=sha(OUTPUT/'statistics.json'),
                vector_manifest_sha256=sha(OUTPUT/'vectors.json'),numerical_owner_proof=proof,
                all_six_retired_before_evaluation=True,no_estimator_or_parameter_owners_created=True,
                all_numerical_array_owners_retired=True,primary=numerical['primary']))
            require(context['prior_total_flops']+bill['flops']<=1_425_000_000_000
                and context['prior_tiny_flops']+bill['flops']<=10_000_000_000
                and context['prior_saved_array_bytes']+vector_bytes<1024**3,
                'Actual entire research and saved-array ceilings, failed work included')
            result.update(analysis_contexts=1,retired_call_count=1,own_complete_bills=True,
                all_numerical_array_owners_retired=True,all_parameter_and_estimator_owners_retired=True,
                analysis_flops=bill['flops'],
                total_research_flops=context['prior_total_flops']+bill['flops'],
                tiny_plus_analysis_flops=context['prior_tiny_flops']+bill['flops'],
                provisional_joint_supply_witness=numerical['primary']['provisional_joint_supply_witness'],
                primary=numerical['primary'],current_cell_centering_error_bound_available=False,
                covariance_joint_noise_standard_error_available=False)
            for name in ('call_00_bill.json','call_00_lifecycle.json','call_00_diagnostics.json',
                         'vectors.bin','vectors.json','statistics.json'):
                path=OUTPUT/name; result['artifacts'][str(path)]=sha(path,tick)
            cp=OUTPUT/'call_00_complete.json'
            write(cp,dict(passed=True,completed=True,call=0,rank=35,arm='analysis',context_retired=True,
                all_parameter_and_estimator_owners_retired=True,bill_sha256=sha(bp),
                lifecycle_sha256=sha(OUTPUT/'call_00_lifecycle.json'),
                diagnostic_sha256=sha(OUTPUT/'call_00_diagnostics.json'),elapsed_s=time.monotonic()-began))
            result['artifacts'][str(cp)]=sha(cp)
            print(json.dumps(dict(event='retired',call=0,rank=35,arm='analysis',complete_sha256=sha(cp))),flush=True)
        for key in ('source_bindings','input_bindings','artifacts'): bind(result[key],tick)
        bind(result['evaluation_bindings'],tick,moments=True)
        require(context['prior_elapsed_s_including_metadata_reserve']+time.monotonic()-began<900.,
                'Original complete serial clock remains unchanged')
        result.update(passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True,
            scientific_information_measured=True,no_automatic_rescue_or_followup=True)
    except BaseException:
        result.update(passed=False,completed=False,error=traceback.format_exc())
    result.update(elapsed_s=time.monotonic()-began,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if result['elapsed_s']>=CAP or result['peak_rss_bytes']>=RSS: result.update(passed=False,completed=False)
    write(OUTPUT/'result.json',result); rh=sha(OUTPUT/'result.json')
    ok=bool(result['passed'] and time.monotonic()-began<CAP)
    write(OUTPUT/'completion.json',dict(passed=ok,completed=ok,result_sha256=rh))
    ch=sha(OUTPUT/'completion.json'); elapsed=time.monotonic()-began; ok=bool(ok and elapsed<CAP)
    write(OUTPUT/'finalization.json',dict(passed=ok,completed=ok,result_sha256=rh,completion_sha256=ch,elapsed_s=elapsed))
    return 0 if ok and time.monotonic()-began<CAP else 1


if __name__=='__main__':
    raise SystemExit(main())
