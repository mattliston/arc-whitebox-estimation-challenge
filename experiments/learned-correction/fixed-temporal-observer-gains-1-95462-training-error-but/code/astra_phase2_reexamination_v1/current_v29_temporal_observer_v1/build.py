"""DRAFT: sole actual-terminal fused build; inert without separately watched --run.

No NumPy, Torch, target, estimator or parameter archive is read on import.
The explicit build stage alone may decode the admitted terminal FP32 archives.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import struct
import sys
import time
import traceback
import types
import zipfile

HERE = Path(__file__).resolve().parent
SCHEMA = 'current_v29_temporal_observer_build_v1'
OUT = HERE/'build_v1'
TRAIN_PINS = {
    HERE/'train.py':'f0649f03e47741f7f1bdc13e38c7c0d68192f3e0b40751205cd62483dca03020',
    HERE/'watch_train.py':'b54d26fa23d3b936f923046ef402b3546347c4e29696f13450e7584173fcbc28',
    HERE/'TRAIN_PROTOCOL.md':'6512b7aa4e6b19755488000fa01c62e23c396ebd80920838135a77b9eafbec60',
    HERE/'TRAIN_WATCH_SPEC.md':'ab0bec144e441740b1f8f60a4ff4ba4ab304ed5ca885b0320fa815f23237b7d3',
}
SOURCE_KEYS = {
    'source':'train.py','watcher':'watch_train.py','protocol':'TRAIN_PROTOCOL.md','watch-spec':'TRAIN_WATCH_SPEC.md',
    'model':'projected_model.py','neural-source':'neural_qualification_v2.py',
    'neural-watch-source':'watch_neural_qualification_v2.py','neural-protocol':'NEURAL_QUALIFICATION_V2_PROTOCOL.md',
    'neural-watch-spec':'NEURAL_QUALIFICATION_V2_WATCH_SPEC.md','capture-worker':'projection_capture_worker.py',
    'capture-watch-source':'projection_capture_watch.py','capture-protocol':'PROJECTION_CAPTURE_PROTOCOL.md',
    'capture-watch-spec':'PROJECTION_CAPTURE_WATCH_SPEC.md',
}
RESULT_KEYS = {'neural':'neural','neural-watch':'neural_watch','check-capture':'check',
    'check-capture-watch':'check_watch','capture':'capture','capture-watch':'capture_watch',
    'projection-qualification':'qualification','projection-qualification-watch':'qualification_watch'}
FLAGS = ('source','protocol','fused','watcher','watch-spec','training-result','training-watch')
_TRAIN = None


def require(ok,message):
    if not ok:
        raise RuntimeError(message)


def load(path,name,digest):
    require(not path.is_symlink() and hashlib.sha256(path.read_bytes()).hexdigest()==digest,'Exact inert source '+str(path))
    spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module); return module


def training_owner():
    global _TRAIN
    if _TRAIN is None:
        _TRAIN=load(HERE/'train.py','v29_build_training_metadata',TRAIN_PINS[HERE/'train.py'])
    return _TRAIN


def training_arguments(record):
    values={key.replace('-','_')+'_sha256':record['source_bindings'][str(HERE/name)] for key,name in SOURCE_KEYS.items()}
    values.update({key.replace('-','_')+'_sha256':record['prerequisite_seals'][role]['result_sha256']
        for key,role in RESULT_KEYS.items()})
    values['joint_inventory_sha256']=record['joint_inventory_sha256']
    return types.SimpleNamespace(**values)


def chain(folder,digest,inputs,limit,tick=None):
    t=training_owner(); paths=[folder/name for name in ('result.json','completion.json','finalization.json')]
    hashes=[t.sha(p,tick) for p in paths]
    require(t.valid(digest) and hashes[0]==digest,'Explicit actual prerequisite result')
    value,end,final=map(t.read,paths)
    require(all(v.get('passed') is True and v.get('completed') is True for v in (value,end,final))
        and value['hashes_unchanged'] and value['sources_unchanged']
        and end['result_sha256']==final['result_sha256']==hashes[0]
        and final['completion_sha256']==hashes[1]
        and 0<=value['elapsed_s']<=final['elapsed_s']<limit,'Actual complete three-seal authority')
    for p,h in zip(paths,hashes):
        t.add(inputs,p,h,tick)
    return value,dict(zip(('result_sha256','completion_sha256','finalization_sha256'),hashes))


def own(args):
    t=training_owner()
    require(all(t.valid(getattr(args,key.replace('-','_')+'_sha256',None)) for key in FLAGS),'Every actual reviewed owner/receipt hash')
    return {HERE/'build.py':args.source_sha256,HERE/'BUILD_PROTOCOL.md':args.protocol_sha256,
        HERE/'fused.py':args.fused_sha256,HERE/'watch_build.py':args.watcher_sha256,
        HERE/'BUILD_WATCH_SPEC.md':args.watch_spec_sha256,**TRAIN_PINS}


def admit(args,sources,inputs,tick=None):
    t=training_owner()
    for path,digest in own(args).items():
        t.add(sources,path,digest,tick)
    result,seals=chain(HERE/'training_v1',args.training_result_sha256,inputs,600.,tick)
    watched,wseals=chain(HERE/'training_watch_v1',args.training_watch_sha256,inputs,600.,tick)
    for path,digest in TRAIN_PINS.items():
        require(result['source_bindings'].get(str(path))==watched['source_bindings'].get(str(path))==digest,
                'The single immutable training implementation, not a different fit')
    ta=training_arguments(result); admitted=t.admit(ta,sources,inputs,{},tick)
    t.verify_result(result,ta,admitted,tick)
    require(watched['schema']=='current_v29_temporal_observer_training_watch_v1' and watched['stage']=='train'
        and watched['retired_calls']==watched['completed_graph_count']==624
        and all(watched.get(k) is True for k in ('child_execution_completed','whole_owned_tree_retired','cleanup_passed',
            'observational_groups_retired','actual_completed_samples','foreign_numerical_workers_checked',
            'proposal_execution_complete','TRAIN_target_bytes_hashed','no_postpublication_observations'))
        and [watched['child_'+key] for key in seals]==list(seals.values())
        and watched['whole_cap_s']==600 and 0<watched['maximum_combined_rss_bytes']<12_000_000_000
        and 0<watched['maximum_interval_s']<=1 and watched['maximum_swap_growth_bytes']==0
        and watched['observation_samples']==len(watched['completed_sample_intervals_s'])
        and all(math.isfinite(v) and 0<v<=1 for v in watched['completed_sample_intervals_s'])
        and watched['prerequisite_seals']==result['prerequisite_seals']
        and watched['model_source_proof']==result['model_source_proof']
        and watched['target_bindings']==result['target_bindings']
        and watched['TRAIN_target_values_decoded'] is False and watched['validation_targets_accessed'] is False
        and watched['publication_tail_elapsed_before_result_s']<1,
        'Actual independent fixed624/150 resource, truth-boundary and retirement authority')
    require(result['TRAIN_screen_passed'] is True and result['TRAIN_screen']['ratio']<=.90,
            'The fixed TRAIN-only usefulness screen passed; no alternative payload or continuation')
    for value in (result,watched):
        t.inventory(value,sources,inputs,tick)
    # All target_bindings deliberately remain sealed provenance, never file inputs.
    require(not any('truthpost' in Path(p).parts for p in {*sources,*inputs}),'Builder has no target-file authority')
    require(watched['model_call_roster_sha256']==result['artifacts'][str(HERE/'training_v1/model_call_roster.json')]
        and len(watched['graph_markers'])==624,'Externally observed complete original roster')
    for i,(packet,marker) in enumerate(zip(result['graph_records'],watched['graph_markers'])):
        p=HERE/'training_v1'/f'graph_{i:04d}_complete.json'
        require(marker['call']==i and marker['path']==str(p) and marker['sha256']==result['artifacts'][str(p)]
            and marker['rank']==packet['rank'] and marker['kind']==packet['kind'],'Every actual externally observed completed graph')
    fused=load(HERE/'fused.py','v29_build_fused_source',args.fused_sha256)
    template,proof=fused.derive_template()
    for path,digest in proof['source_bindings'].items():
        t.add(sources,path,digest,tick)
    for name in ('terminal_payload','normalizers'):
        item=result[name]; path=HERE/'training_v1'/('terminal_payload.npz' if name=='terminal_payload' else 'normalizers.npz')
        require(item['path']==str(path) and item['sha256']==result['artifacts'][str(path)],'Only actual sealed terminal archives')
    require(not {'numpy','torch','flopscope','threadpoolctl'}&set(sys.modules),'Eligibility and template remain stdlib-only')
    return dict(training=result,training_seals=seals,training_watch_seals=wseals,training_args=ta,
        fused=fused,template=template,template_proof=proof)


def decode_npz(path,metadata,tick=None):
    """Explicit build only: strict bounded NPY decoding, no numerical library."""
    t=training_owner(); require(t.sha(path,tick)==metadata['sha256'],'Actual admitted archive bytes')
    require(Path(path).stat().st_size<1_000_000,'Small fixed literal archive')
    arrays={}
    with zipfile.ZipFile(path) as archive:
        require(len(archive.infolist())==len(metadata['arrays']) and set(archive.namelist())=={k+'.npy' for k in metadata['arrays']},
                'Exactly prescribed arrays, no archive duplicates or extra payload')
        for key,meta in metadata['arrays'].items():
            info=archive.getinfo(key+'.npy')
            require(info.file_size<1_000_000 and info.compress_type==zipfile.ZIP_STORED and not info.flag_bits&1,'Original uncompressed unencrypted NPZ member')
            raw=archive.read(info); require(raw[:6]==b'\x93NUMPY' and raw[6:8] in (b'\x01\x00',b'\x02\x00'),'Original NPY format')
            length_size=2 if raw[6]==1 else 4
            length=int.from_bytes(raw[8:8+length_size],'little'); start=8+length_size
            require(length<65536 and start+length<=len(raw),'Bounded exact NPY header')
            header=ast.literal_eval(raw[start:start+length].decode('latin1')); payload=raw[start+length:]
            shape=tuple(meta['shape']); count=math.prod(shape)
            require(header==dict(descr='<f4',fortran_order=False,shape=shape) and meta['dtype']=='float32'
                and len(payload)==count*4 and hashlib.sha256(payload).hexdigest()==meta['raw_sha256'],
                'Exact original FP32 shape, order and payload SHA')
            values=struct.unpack('<'+str(count)+'f',payload)
            require(all(math.isfinite(v) for v in values),'Finite terminal FP32 values')
            cursor=0
            def tree(dims):
                nonlocal cursor
                if not dims:
                    value=values[cursor]; cursor+=1; return value
                return [tree(dims[1:]) for _ in range(dims[0])]
            arrays[key]=tree(shape)
            require(cursor==count,'All literal elements consumed exactly')
            if tick is not None:
                tick()
    return arrays


def literal_payload(admitted,tick=None):
    result=admitted['training']; fused=admitted['fused']
    learned=decode_npz(result['terminal_payload']['path'],result['terminal_payload'],tick)
    norms=decode_npz(result['normalizers']['path'],result['normalizers'],tick)
    require(set(learned)==set(fused.MODEL_KEYS) and set(norms)=={'centers','scales','r'},'Sole full terminal and fixed normalizer payloads')
    values=dict(learned,**norms); values['b']=[values['b']]; values['r']=[values['r']]
    values={key:values[key] for key in fused.KEYS}; metadata=fused.literal_metadata(values)
    for key,item in metadata.items():
        prior=(result['terminal_payload'] if key in learned else result['normalizers'])['arrays'][key]
        require(item['raw_sha256']==prior['raw_sha256'] and item['dtype']==prior['dtype']
            and item['shape']==([1] if key in ('b','r') else prior['shape']), 'Exact payload bytes; only explicit scalar deployment shape')
    return values,metadata


def verify_build(result_sha,candidate_sha,source_sha,protocol_sha,tick=None):
    """No estimator import or truth reads; decode only the small sealed literals."""
    t=training_owner(); inputs={}
    built,seals=chain(OUT,result_sha,inputs,120.,tick)
    args=types.SimpleNamespace(**built['arguments'])
    require(args.source_sha256==source_sha and args.protocol_sha256==protocol_sha
        and built['schema']==SCHEMA and built['candidate_sha256']==candidate_sha,'Explicit exact build/source/protocol/candidate')
    sources={}; admitted=admit(args,sources,inputs,tick); literal,metadata=literal_payload(admitted,tick)
    source,proof=admitted['fused'].compose(literal); candidate=OUT/'estimator.py'
    require(candidate.read_text()==source and t.sha(candidate,tick)==candidate_sha
        and built['candidate']==str(candidate) and built['proof']==proof and built['deployed_literal']==metadata
        and built['training_seals']==admitted['training_seals'] and built['training_watch_seals']==admitted['training_watch_seals']
        and built['estimator_executed'] is False and built['truth_accessed'] is False and built['parameter_updates']==0,
        'Exact sole-payload candidate with whole physical/arithmetic/literal inverse')
    for path,digest in built['artifacts'].items():
        t.add(inputs,path,digest,tick)
    require(built['source_bindings']==sources,'Exact completed source inventory')
    return dict(build=built,build_seals=seals,candidate=candidate,proof=proof,source_bindings=sources,input_bindings=inputs)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--run',action='store_true')
    for key in FLAGS:
        parser.add_argument('--'+key+'-sha256')
    args=parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    began=time.monotonic(); t=training_owner()
    require(sys.dont_write_bytecode and sys.flags.optimize==0 and os.getpriority(os.PRIO_PROCESS,0)>=10
        and all(os.environ.get(k)=='1' for k in t.BLAS),'Actual nice10/sixBLAS1/assertions/bytecodeoff')
    require(not OUT.exists() and not HERE.is_symlink(),'Exclusive future build stage'); OUT.mkdir()
    record=dict(schema=SCHEMA,passed=False,completed=False,source_bindings={},input_bindings={},artifacts={},
        arguments={k:v for k,v in vars(args).items() if k!='run'},estimator_executed=False,truth_accessed=False,
        numerical_imports=False,parameter_updates=0,submission_authorized=False)
    def guard():
        require(time.monotonic()-began<120 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2_000_000_000,'Build whole120s/self2GB')
    try:
        admitted=admit(args,record['source_bindings'],record['input_bindings'],guard)
        literal,metadata=literal_payload(admitted,guard); source,proof=admitted['fused'].compose(literal)
        candidate=OUT/'estimator.py'
        with candidate.open('x') as stream:
            stream.write(source)
        record['artifacts'][str(candidate)]=t.sha(candidate,guard)
        t.write(OUT/'proof.json',proof); record['artifacts'][str(OUT/'proof.json')]=t.sha(OUT/'proof.json',guard)
        record.update(candidate=str(candidate),candidate_sha256=record['artifacts'][str(candidate)],proof=proof,deployed_literal=metadata,
            training_seals=admitted['training_seals'],training_watch_seals=admitted['training_watch_seals'],
            terminal_payload=admitted['training']['terminal_payload'],normalizers=admitted['training']['normalizers'],
            TRAIN_screen=admitted['training']['TRAIN_screen'])
        for group in ('source_bindings','input_bindings','artifacts'):
            for path,digest in record[group].items():
                require(t.sha(path,guard)==digest,'Final exact build source/input/artifact bytes')
        require(not {'numpy','torch','flopscope','threadpoolctl'}&set(sys.modules),'No numerical imports or estimator execution')
        record.update(passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True)
        guard()
    except BaseException:
        record.update(passed=False,completed=False,error=traceback.format_exc())
    record.update(elapsed_s=time.monotonic()-began,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    if record['elapsed_s']>=120 or record['peak_rss_bytes']>=2_000_000_000:
        record.update(passed=False,completed=False)
    t.write(OUT/'result.json',record); rh=t.sha(OUT/'result.json'); ok=bool(record['passed'] and time.monotonic()-began<120)
    t.write(OUT/'completion.json',dict(passed=ok,completed=ok,result_sha256=rh)); ch=t.sha(OUT/'completion.json')
    elapsed=time.monotonic()-began; ok=bool(ok and elapsed<120)
    t.write(OUT/'finalization.json',dict(passed=ok,completed=ok,result_sha256=rh,completion_sha256=ch,elapsed_s=elapsed))
    ok=bool(ok and time.monotonic()-began<120)
    print(json.dumps(dict(passed=ok,completed=ok,error=record.get('error'))),flush=True)
    return 0 if ok else 1


if __name__=='__main__':
    raise SystemExit(main())
