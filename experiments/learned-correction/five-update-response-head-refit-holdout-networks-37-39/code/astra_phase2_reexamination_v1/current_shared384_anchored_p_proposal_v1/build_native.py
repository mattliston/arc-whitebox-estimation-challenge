"""Sole-terminal P-only native source build; inert and NumPy-free at import.

Parameter NPZ bytes are decoded only after actual positive training six-seal
admission. No target, estimator, surrogate, anchor or fitting code is executed.
The emitter preserves every source byte outside changed existing P values.
"""
import argparse
import ast
import hashlib
import io
import json
import math
import os
from pathlib import Path
import resource
import signal
import struct
import sys
import time
import traceback
import zipfile

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
PARENT = NEW/'current_comp_shared_gain_slice_v1/build_v1/estimator.py'
PARENT_SHA = 'c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea'
OUTPUT = HERE/'native_build_v1'
SCHEMA = 'current_shared384_anchored_p_native_build_v1'
NAMES = ('result.json','completion.json','finalization.json')
SEAL_KEYS = ('result_sha256','completion_sha256','finalization_sha256')
KEYS = ('source','watch','protocol','watch_spec','training','training_watch','payload')
PKEYS = ('msg_w','msg_b','enc0_w','enc0_b','enc2_w','enc2_b',
         'enc4_w','enc4_b','head','head_b','gain')
TRAIN = (35,36,38,40,41,43,44,45,46,47,49,50,51,52,53,54,55,56,57,58)
TRAIN_OWNERS = {
    'run_training.py':'c43e972c2fddf9280fe2d83f0204ce4afeb75935e7f6ae47548cec689d01624e',
    'watch_training.py':'4cd6d9bdc8b49f7e9e63717c01393b6e58b62d0040c51c5da434c37a873f8819',
    'train_engine.py':'0aa7100f97a08b1dd0a081dc757869f4ae0b1949f65a580105cfd9103144920b',
    'proposal_admission.py':'b3fdff53879586f100da5ae44f183a6ef68b89e62d859d69240aa0aaf07f5328',
    'TRAIN_PROTOCOL.md':'2dc300b5db2cd8f0230225ef5134a78639fcb910a2f46d55f45d6443ba056ece',
}
PINS = {PARENT:PARENT_SHA,
    NEW/'current_d50_native_objective_p1_v1/build.py':'57f87081cdcc742266bc8487ef5ba11358bdf25e8341b9b1f4dd4458f06da54c',
    NEW/'current_bd_p_coadaptation_v1/build_terminal.py':'8b621d1713841c6271f048504205030ec7bc78df1842be4a3071e00ecf01d084',
    NEW/'current_native_trajectory_adjoint_v1/supervise.py':'648e5227795a8d2e5a91a1a421e0e54ffea1f76d6d6d65b28335178c8772025d',
    **{HERE/k:v for k,v in TRAIN_OWNERS.items()}}


def require(ok,message):
    if not ok: raise RuntimeError(message)


def no_numerics():
    require(not any(k.split('.')[0] in ('numpy','torch','scipy','flopscope',
        'flopscope_server','threadpoolctl') for k in sys.modules),'Stdlib-only source/representation process')


def sha(path,tick=None):
    path=Path(path)
    require(path.is_file() and 'truthpost' not in path.parts and not
        (path.suffix in ('.npz','.npy') and 'truth' in path.name.lower()),'No target file access')
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):
            h.update(block)
            if tick is not None: tick()
    return h.hexdigest()


def read(path):
    path=Path(path)
    require(path.suffix=='.json' and path.is_file() and not path.is_symlink()
        and path.stat().st_size<64*1024**2,'Bounded regular JSON metadata')
    return json.loads(path.read_text())


def merge(target,items):
    for path,digest in items.items():
        key=str(path)
        require(key not in target or target[key]==digest,'No conflicting source/input owner')
        target[key]=digest


def bind(items,tick=None):
    for path,digest in items.items():
        require(isinstance(digest,str) and len(digest)==64 and sha(path,tick)==digest,'Exact admitted bytes: '+str(path))


def chain(folder,expected,inputs,tick=None):
    paths=[folder/name for name in NAMES]
    ds=[sha(p,tick) for p in paths]; records=list(map(read,paths))
    r,e,f=records
    require(ds[0]==expected and all(x.get('passed') is True and x.get('completed') is True for x in records)
        and e['result_sha256']==f['result_sha256']==ds[0] and f['completion_sha256']==ds[1],
        'Actual positive linked three-seal prerequisite')
    merge(inputs,dict(zip(paths,ds)))
    return r,f,dict(zip(SEAL_KEYS,ds))


def admission(args,tick=None):
    """No NPZ member, output array or target decoding; usable by the build watch."""
    no_numerics()
    require(all(isinstance(getattr(args,k+'_sha256',None),str)
        and len(getattr(args,k+'_sha256'))==64
        and set(getattr(args,k+'_sha256'))<=set('0123456789abcdef') for k in KEYS),'Seven actual SHA authorities')
    sources={str(p):d for p,d in PINS.items()}; inputs={}
    merge(sources,{HERE/'build_native.py':args.source_sha256,HERE/'watch_build_native.py':args.watch_sha256,
        HERE/'BUILD_NATIVE_PROTOCOL.md':args.protocol_sha256,HERE/'BUILD_NATIVE_WATCH_SPEC.md':args.watch_spec_sha256})
    bind(sources,tick)
    r,rf,rs=chain(HERE/'training_v1',args.training_sha256,inputs,tick)
    w,wf,ws=chain(HERE/'training_watch_v1',args.training_watch_sha256,inputs,tick)
    require(r['training_dose_completed'] and r['all_owners_retired'] and r['fits']==1
        and r['full_model_calls']==60 and r['backwards']==20 and r['optimizer_steps']==5
        and r['truth_accessed'] is True and r['native_score_measured'] is False
        and r['native_gradient_certified'] is False and r['release_ready'] is False
        and r['automatic_next_stage'] is False and r['actual_flopscope_bill_claimed'] is False
        and r['qualification_scope']=='approximate_proposal_only_native_acceptance_pending'
        and r['complete_source_upper']==134_939_477_450_240
        and 0<r['elapsed_s']<=rf['elapsed_s']<1500
        and 0<r['maximum_self_rss_bytes']<8_000_000_000,'One completed fixed approximate proposal, not native authority')
    require(w['phase']=='training' and w['retired_call_count']==60 and w['whole_owned_tree_retired']
        and {k:w['child_'+k] for k in SEAL_KEYS}==rs
        and w['source_bindings']==TRAIN_OWNERS and 0<w['elapsed_s']<=wf['elapsed_s']<1500
        and 0<w['maximum_combined_rss_bytes']<8_000_000_000
        and 0<w['maximum_interval_s']<=1 and w['maximum_swap_growth_bytes']==0,
        'Actual independent1500s/8GB/no-swap/completed-progress training supervisor')
    require(all(r['source_bindings'].get(str(HERE/k))==v for k,v in TRAIN_OWNERS.items()),
        'The completed dose used exactly the reviewed five training sources')
    bind(r['source_bindings'],tick); merge(sources,r['source_bindings'])
    bind(w['inherited_source_bindings'],tick); merge(sources,w['inherited_source_bindings'])
    # TRAIN truths are deliberately never opened or rehashed here. Their final
    # hashes were owned by the actual successful training worker, not borrowed
    # as a new truth-read authority for the builder or native acquisition.
    bind(r['input_bindings'],tick); merge(inputs,r['input_bindings'])
    require(len(r['calls'])==60 and len(r['updates'])==5 and len(r['payloads'])==1,'Exact fixed call/update/payload roster')
    expected={}
    for i,item in enumerate(r['calls']):
        name=HERE/'training_v1'/f'call_{i:03d}.json'
        require(item['index']==i and item['rank']==TRAIN[i%20]
            and item['stage']==('baseline','training','terminal')[i//20]
            and item['call_graph_owners_retired'] and item['input_owners_retired']
            and item['path']==str(name) and r['artifacts'][str(name)]==item['sha256'],
            'All sixty actual ordered retired calls, no selected checkpoint')
        expected[str(name)]=item['sha256']
    for i,item in enumerate(r['updates']):
        path=HERE/'training_v1'/f'update_{i:02d}.json'
        require(item['index']==i and item['path']==str(path)
            and item['sha256']==r['artifacts'][str(path)],'Exact update receipt path')
        u=read(path)
        require(u['optimizer_step']==u['parameter_updates']==i+1
            and u['ranks']==list(TRAIN[4*i:4*i+4])
            and u['learning_rate']==5e-5+.5*(5e-4-5e-5)*(1+math.cos(math.pi*i/4))
            and math.isfinite(u['pre_clip_gradient_norm']) and u['pre_clip_gradient_norm']>=0
            and u['frozen_A_cstar_nonP_inactive_unchanged'] and u['inactive66_gradient_moments_zero']
            and u['checkpoint_saved'] is False and set(u['adam_states'])==set(PKEYS)
            and all(x['step']==i+1 for x in u['adam_states'].values()),'Actual fixed five Adam updates and immutable fields')
        expected[str(path)]=item['sha256']
    item=r['payloads'][0]; path=HERE/'training_v1/terminal_payload.npz'
    meta_path=HERE/'training_v1/terminal_payload.json'; diagnostic_path=HERE/'training_v1/diagnostics.json'
    require(item==dict(path=str(path),sha256=args.payload_sha256,record=str(meta_path))
        and path.is_file() and not path.is_symlink() and path.stat().st_size<4*1024**2
        and r['artifacts'][str(path)]==args.payload_sha256,'Only the sole terminal artifact, never a substitute or interpolant')
    p=read(meta_path); d=read(diagnostic_path)
    require(p['sha256']==args.payload_sha256 and p['nbytes']==path.stat().st_size
        and p['terminal_epoch']==1 and p['parameter_updates']==5 and p['training_graphs']==20
        and p['trainable_keys']==list(PKEYS) and p['dtype']=='float32'
        and p['original_candidate_sha256']==PARENT_SHA and p['checkpoint_selection'] is False
        and p['validation_targets_accessed'] is False and p['native_score_measured'] is False
        and p['native_gradient_certified'] is False and p['parameter_hashes']==u['parameter_hashes'],
        'Sole terminal sealed after exactly five updates, not selected by diagnostics')
    require(d['schema']=='current_shared384_anchored_p_train_engine_v1'
        and d['passed'] and d['completed'] and d['roster']==list(TRAIN)
        and d['full_model_calls']==60 and d['baseline_forwards']==d['training_forwards']==d['terminal_forwards']==20
        and d['backwards']==20 and d['parameter_updates']==5 and d['epochs']==1
        and d['terminal_payload']==p and d['all_engine_numerical_owners_retired']
        and d['baseline20_before_truth'] and d['anchors_used_only_at_loss'] and d['teacher_states_used'] is False
        and d['checkpoint_selection'] is False and d['validation_targets_accessed'] is False
        and d['native_score_measured'] is False and d['native_gradient_certified'] is False
        and d['old_failed_gates_preserved'] and d['automatic_next_stage'] is False,
        'Actual complete autonomous proposal, original negative scopes unchanged')
    require(all(p in r['artifacts'] for p in (str(meta_path),str(diagnostic_path))), 'Sealed terminal metadata and complete engine diagnostic')
    bind(r['artifacts'],tick); merge(inputs,r['artifacts'])
    require(all(inputs[p]==h for p,h in expected.items()),'All terminal/dose records joined to actual artifact inventory')
    no_numerics()
    return dict(source_bindings=sources,input_bindings=inputs,training_seals=rs,training_watch_seals=ws,
        payload=str(path),payload_sha256=args.payload_sha256,payload_record=p,
        authority={k:getattr(args,k+'_sha256') for k in KEYS},truth_files_opened=False,
        terminal_training_prerequisite_qualified=True)


def pack_literal(value):
    """Exact FP32 representation conversion only; no model/numerical library."""
    if isinstance(value,list):
        require(value,'No empty deployed literal dimension')
        parts=[pack_literal(v) for v in value]
        require(all(s==parts[0][0] for s,_ in parts),'Rectangular deployed literal')
        return (len(value),)+parts[0][0],b''.join(raw for _,raw in parts)
    require(type(value) in (int,float) and math.isfinite(value),'Finite numeric literal')
    raw=struct.pack('<f',value)
    require(math.isfinite(struct.unpack('<f',raw)[0]),'Finite representable FP32 literal')
    return (),raw


def surface(raw):
    """One UTF8 line-offset index, never repeated giant-line source rescans."""
    tree=ast.parse(raw); lines=raw.splitlines(keepends=True); offsets=[0]
    for line in lines: offsets.append(offsets[-1]+len(line))
    def span(node): return (offsets[node.lineno-1]+node.col_offset,offsets[node.end_lineno-1]+node.end_col_offset)
    names=('_GNN_LITERAL','_BDNH_A_LITERAL','_BD_COEFFICIENT_LITERAL'); nodes={}
    for node in tree.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            if name in names:
                require(name not in nodes,'Unique deployed literal assignment'); nodes[name]=node
    require(set(nodes)==set(names) and isinstance(nodes['_GNN_LITERAL'].value,ast.Dict),'Exact current P/A/cstar literal surface')
    literal=nodes['_GNN_LITERAL'].value; values={}; spans={}
    for key,value in zip(literal.keys,literal.values,strict=True):
        name=ast.literal_eval(key)
        require(isinstance(name,str) and name not in values,'Unique literal field')
        values[name]=pack_literal(ast.literal_eval(value)); spans[name]=span(value)
    fixed={k:raw[slice(*span(nodes[k]))] for k in names[1:]}
    fixed_values={k:pack_literal(ast.literal_eval(nodes[k].value))[1] for k in names[1:]}
    return values,spans,fixed,fixed_values


def decode_payload(path,metadata):
    """Representation-only NPY payload decode, callable only after admission."""
    values={}
    with zipfile.ZipFile(path) as archive:
        entries=archive.infolist()
        require(len(entries)==len(metadata['shapes']) and {x.filename for x in entries}
            =={k+'.npy' for k in metadata['shapes']} and sum(x.file_size for x in entries)<4*1024**2,
            'Exact bounded NPZ member roster')
        for item in entries:
            require(not item.flag_bits&1 and item.compress_type==zipfile.ZIP_STORED,'Original unencrypted np.savez encoding')
            raw=archive.read(item); require(raw[:8]==b'\x93NUMPY\x01\x00','Original NPY1 typed member')
            size=struct.unpack('<H',raw[8:10])[0]
            header=ast.literal_eval(raw[10:10+size].decode('latin1').strip()); key=item.filename[:-4]
            shape=tuple(metadata['shapes'][key]); data=raw[10+size:]
            require(header==dict(descr='<f4',fortran_order=False,shape=shape)
                and len(data)==4*math.prod(shape) and hashlib.sha256(data).hexdigest()==metadata['parameter_hashes'][key]
                and all(math.isfinite(x) for x, in struct.iter_unpack('<f',data)),'Exact finite same-layout terminal FP32 bytes')
            values[key]=(shape,data)
    return values


def nested(shape,raw):
    values=iter(x for x, in struct.iter_unpack('<f',raw))
    def build(dim): return next(values) if not dim else [build(dim[1:]) for _ in range(dim[0])]
    value=build(shape); require(next(values,None) is None,'Exact complete literal consumption'); return value


def emit(parent,terminal):
    old,spans,fixed,fixed_values=surface(parent)
    require(set(old)==set(terminal) and set(PKEYS)<=set(old),'Full same-field payload, never a partial head')
    require(sum(math.prod(old[k][0]) for k in PKEYS)==45200,'Exactly existing allocated P parameters')
    edits=[]
    for key,(shape,raw) in terminal.items():
        require(shape==old[key][0] and len(raw)==len(old[key][1]),'Original field layout')
        require(key in PKEYS or raw==old[key][1],'Frozen non-P/normalizer/input bytes')
        if key in ('head','head_b','gain'):
            count=4*math.prod(shape[1:]); require(raw[:count]==old[key][1][:count],'Frozen inactive layer0 bytes')
        if raw!=old[key][1]:
            start,stop=spans[key]; edits.append((start,stop,repr(nested(shape,raw)).encode()))
    candidate=parent
    for start,stop,after in sorted(edits,reverse=True): candidate=candidate[:start]+after+candidate[stop:]
    actual,new_spans,new_fixed,new_fixed_values=surface(candidate)
    require(actual==terminal and fixed==new_fixed and fixed_values==new_fixed_values,'Exact native literal roundtrip and A/cstar source bytes')
    inverse=candidate
    for key,(start,stop) in sorted(new_spans.items(),key=lambda item:item[1][0],reverse=True):
        if key in PKEYS:
            a,b=spans[key]; inverse=inverse[:start]+parent[a:b]+inverse[stop:]
    require(inverse==parent,'Whole current-c2 source inverse, every non-P source byte retained')
    return candidate,dict(parent_sha256=PARENT_SHA,candidate_sha256=hashlib.sha256(candidate).hexdigest(),
        whole_source_inverse_verified=True,only_existing_P_literal_values_changed=True,
        modified_keys=[k for k in old if actual[k]!=old[k]],allocated_parameters=45200,active_parameters=45134,
        inactive66_byte_exact=True,non_P_byte_exact=True,A_source_byte_exact=True,cstar_source_byte_exact=True,
        A_raw_sha256=hashlib.sha256(fixed_values['_BDNH_A_LITERAL']).hexdigest(),
        cstar_raw_sha256=hashlib.sha256(fixed_values['_BD_COEFFICIENT_LITERAL']).hexdigest(),
        parameter_shapes={k:list(v[0]) for k,v in actual.items()},
        parameter_hashes={k:hashlib.sha256(v[1]).hexdigest() for k,v in actual.items()},
        anchors_deployed=False,surrogate_deployed=False,estimator_arithmetic_changed=False)


def verify_build(build_sha,candidate_sha,builder_sha,protocol_sha,tick=None):
    """Saved-source proof replay; no NPZ member or output/truth array decode."""
    no_numerics(); inputs={}
    r,f,seals=chain(OUTPUT,build_sha,inputs,tick)
    require(r['schema']==SCHEMA and r['candidate_sha256']==candidate_sha
        and r['authority']['source']==builder_sha and r['authority']['protocol']==protocol_sha
        and r['source_build_qualified'] and r['all_owners_retired'] and r['payload_decoded']
        and r['truth_accessed'] is False and r['estimator_executed'] is False
        and r['numerical_imports'] is False and r['parameter_updates']==0
        and 0<r['elapsed_s']<=f['elapsed_s']<60 and 0<r['peak_rss_bytes']<2_000_000_000,
        'Actual completed bounded sole source build')
    args=argparse.Namespace(run=True,**{k+'_sha256':v for k,v in r['authority'].items()})
    require(set(r['authority'])==set(KEYS),'Exactly original actual builder authorities')
    q=admission(args,tick)
    require(r['source_bindings']==q['source_bindings'] and r['input_bindings']==q['input_bindings']
        and r['training_seals']==q['training_seals'] and r['training_watch_seals']==q['training_watch_seals']
        and r['payload_sha256']==q['payload_sha256'],'Identical completed sole terminal admission')
    parent=PARENT.read_bytes(); path=OUTPUT/'estimator.py'
    require(r['candidate']==str(path) and not path.is_symlink() and sha(path,tick)==candidate_sha,'Actual emitted source')
    candidate=path.read_bytes(); terminal=surface(candidate)[0]; replay,proof=emit(parent,terminal)
    null,_=emit(parent,surface(parent)[0]); proof['null_source_byte_exact']=null==parent
    require(replay==candidate and null==parent and proof==r['proof']
        and proof['parameter_hashes']==q['payload_record']['parameter_hashes']
        and proof['parameter_shapes']==q['payload_record']['shapes']
        and proof['A_raw_sha256']==q['payload_record']['A_sha256']
        and proof['cstar_raw_sha256']==q['payload_record']['cstar_sha256'],'Complete source/typed sealed terminal proof')
    expected={str(path):candidate_sha,str(OUTPUT/'null_estimator.py'):PARENT_SHA,
        str(OUTPUT/'proof.json'):sha(OUTPUT/'proof.json',tick)}
    require(r['artifacts']==expected and read(OUTPUT/'proof.json')==proof,'Exact three source-build artifacts')
    bind(expected,tick); merge(inputs,q['input_bindings']);merge(inputs,expected)
    return dict(build=r,candidate=str(path),candidate_sha256=candidate_sha,proof=proof,
        source_bindings=q['source_bindings'],input_bindings=inputs,build_seals=seals,
        training_seals=q['training_seals'],training_watch_seals=q['training_watch_seals'],payload_sha256=q['payload_sha256'])


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true')
    for key in KEYS: parser.add_argument('--'+key.replace('_','-')+'-sha256')
    args=parser.parse_args()
    if not args.run: parser.print_help();return 0
    no_numerics(); require(sys.flags.optimize==0 and os.getpriority(os.PRIO_PROCESS,0)>=10,'Assertions and nice10')
    began=time.monotonic()
    def tick(): require(time.monotonic()-began<60 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2_000_000_000,'Whole60s/self2GB source build')
    def alarm(*unused): raise TimeoutError('Whole source build60s')
    signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,60)
    require(not OUTPUT.exists() and not HERE.is_symlink(),'Exclusive source build; no overwrite or resume');OUTPUT.mkdir()
    r=dict(schema=SCHEMA,passed=False,completed=False,estimator_executed=False,truth_accessed=False,
        numerical_imports=False,payload_decoded=False,parameter_updates=0,source_build_qualified=False,
        all_owners_retired=False,source_bindings={},input_bindings={},artifacts={},automatic_next_stage=False)
    try:
        q=admission(args,tick);r.update({k:q[k] for k in ('source_bindings','input_bindings','training_seals',
            'training_watch_seals','payload_sha256','authority')})
        parent=PARENT.read_bytes();require(hashlib.sha256(parent).hexdigest()==PARENT_SHA,'Exact current c2 before decoding')
        terminal=decode_payload(q['payload'],q['payload_record']);r['payload_decoded']=True
        candidate,proof=emit(parent,terminal);null,_=emit(parent,surface(parent)[0])
        proof['null_source_byte_exact']=null==parent
        require(null==parent and proof['parameter_shapes']==q['payload_record']['shapes']
            and proof['parameter_hashes']==q['payload_record']['parameter_hashes']
            and proof['A_raw_sha256']==q['payload_record']['A_sha256']
            and proof['cstar_raw_sha256']==q['payload_record']['cstar_sha256'],'Exact sole terminal/P0/frozen-field representation')
        for name,raw in (('estimator.py',candidate),('null_estimator.py',null)):
            with (OUTPUT/name).open('xb') as stream:stream.write(raw)
            r['artifacts'][str(OUTPUT/name)]=sha(OUTPUT/name,tick)
        write(OUTPUT/'proof.json',proof);r['artifacts'][str(OUTPUT/'proof.json')]=sha(OUTPUT/'proof.json',tick)
        del terminal
        bind(r['source_bindings'],tick);bind(r['input_bindings'],tick);bind(r['artifacts'],tick);no_numerics();tick()
        r.update(passed=True,completed=True,source_build_qualified=True,all_owners_retired=True,
            candidate=str(OUTPUT/'estimator.py'),candidate_sha256=proof['candidate_sha256'],parent_sha256=PARENT_SHA,proof=proof)
    except BaseException: r['error']=traceback.format_exc()
    r.update(elapsed_s=time.monotonic()-began,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if r['elapsed_s']>=60 or r['peak_rss_bytes']>=2_000_000_000:r.update(passed=False,completed=False)
    write(OUTPUT/'result.json',r);rh=sha(OUTPUT/'result.json')
    write(OUTPUT/'completion.json',dict(passed=r['passed'],completed=r['completed'],result_sha256=rh))
    write(OUTPUT/'finalization.json',dict(passed=r['passed'],completed=r['completed'],result_sha256=rh,
        completion_sha256=sha(OUTPUT/'completion.json'),elapsed_s=time.monotonic()-began))
    tick();signal.setitimer(signal.ITIMER_REAL,0)
    print(json.dumps({k:r.get(k) for k in ('passed','completed','candidate_sha256','elapsed_s','peak_rss_bytes','error')}),flush=True)
    return 0 if r['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
