"""Source-only fixed pair256 build, after actual tiny and current37 admission.

No import-time stage, numerical import, estimator execution, targets or fit.
The separate build watcher owns the60s/2GB full process-tree scope.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
P2 = HERE.parent.parent
GAIN = HERE.parent/'current_comp_shared_gain_slice_v1'
OUTPUT = HERE/'build37_v1'
SCHEMA = 'current_comp_shared_pair256_build37_v1'
PARENT_SHA = 'c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea'
CANDIDATE_SHA = 'd7e9cba805c286bc7e1f81d36d05e4bc8cfb9b992324e61037d3aa6bee5cf6ba'
NAMES = ('result.json','completion.json','finalization.json')
PINS = {
    HERE/'compose.py':'1d73f6b9c951917a5af0bbc57ec2cd020e01cff471cddbfd37720f73995b1b3e',
    HERE/'metric_fnp.py':'52731186b0ee2ecd932fb9b3a7a44bd23a305dd17d3afbf58f8949019df5c44e',
    HERE/'DESIGN.md':'5290ec27913961c438768730320f1641bf0d7fcd530b70fbce4e5dc6d30a6653',
    HERE/'tiny_worker.py':'c6afc327c9acb6a9cd6ba3ed66ef293f8a05d9a22e9031b97ffe9cec49d42e1d',
    HERE/'tiny_client.py':'a8876ee4dffab18e5c468c90a91ef7c96289264ecf5014d95b2feea7cdf6ca10',
    HERE/'watch_tiny.py':'736304302b274697b19e031c645203f41bd25d0a6ee041e20b8ce9f82126f950',
    HERE/'TINY_PROTOCOL.md':'9989c0e869cb10614043f49855229247e788cb9c79caef9f31ed24c8f783c1dd',
    HERE/'TINY_WATCH_SPEC.md':'50752c1a67057eb78f75c314fb52b5161a532192b825499e1915e486c77f0c97',
    HERE/'TINY_SOURCE_PREPARATION.md':'202f4618b5e0f31d2943694e036b0e552d8d7fa90a6160f6dbf68723fa034b7a',
    GAIN/'build_v1/estimator.py':PARENT_SHA,
    GAIN/'native40_v2.py':'93012679ba77f392353a7311b25c3e770451e8a88ef7a6067505832b969629a6',
    GAIN/'watch_native40_v2.py':'a854b402e8112e848c3157c0c9fef59739302ac4309bdee5596585d93dd4bb97',
    GAIN/'NATIVE40_V2_PROTOCOL.md':'ef94ffd39c40b0be0ae8144c6e3e741f79999500c0071ed26f95b2ebf9ea8906',
    GAIN/'NATIVE40_V2_WATCH_SPEC.md':'29e9b3840fdf7920379a70ae712d02f0ceb527e7c2ca771bfb9772abf0bfaaf3',
}
CHAINS = {
    HERE/'tiny_qualification_v1/local':(
        'f61d26c1d610dd5acbb1c8caa86758d8a8980b2f332c9f566d4ccd20f580bc26',
        '6d1456e34c1d6023b905443f08ae9529afa2c6e7b6f4c06f71da2d433fcd81b7',
        'da263c6206d153c8412ca67426b90b31b1cdd5d0d0d52142a3008856d2afccb5'),
    HERE/'tiny_qualification_v1/remote':(
        '8c77f7a56cd4fffe27d1af628d91a43792183a8631543030e5c16bbefa63d361',
        '1f382d1df543d721b6b832d41a2608c6c8785daba097927eb1c79ac507b148c2',
        'aef0cda799bf0e58d4dc2ed0803453d5d5ba5201f9f161558046eb8d6ef9e03e'),
    HERE/'tiny_qualification_watch_v1':(
        'fdbd5578a042894d8e1d421d55c4db336521165b42411d7f37af93c7fb85c036',
        '11e60b92566f10c220cc58ed9be80567779dbe8e88fe640c4a5bb8c560f41121',
        'd3434d893a457e21af00dbe394ab1b7e94545389fafff284f65538181fed5146'),
    GAIN/'native40_v2':(
        '95d80ab15860bd51865c828a48d09f195642fa6ac6d8f86f9c5e99b4f864d83e',
        'a4896323273ce08d9e8c18a8ee9b2e0a8ada6439bbfbb5d334c530f276eb513c',
        'a113ac65d0360f4021ccc89d10f9bc0165e0be8086fe95afb77ff23e44e56319'),
    GAIN/'native40_watch_v2':(
        '1bc1135f275609a4ea64bd7414de2cae9881f9396af3b561f6af6e1fe2bc7bfb',
        'ff445d8e23b4b010c03b4137685294ad0bd85b4484f9a2862096c610b229490c',
        'be9293ac03e21ac637bb1dc04af8fb4646639d5cd58c39688e027add935d9541'),
}


def require(ok,message):
    if not ok:
        raise RuntimeError(message)


def no_numerics():
    require(not any(n.split('.')[0] in ('numpy','scipy','torch','flopscope',
        'flopscope_server','threadpoolctl') for n in sys.modules),'Source-only process')


def sha(path,tick=None):
    path=Path(path)
    require(path.is_file() and not path.is_symlink() and path.is_relative_to(P2)
        and 'truthpost' not in path.parts and path.name != 'key.txt', 'Regular in-scope non-target binding')
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):
            h.update(block)
            if tick: tick()
    return h.hexdigest()


def read(path):
    path=Path(path)
    require(path.suffix=='.json' and not path.is_symlink() and path.stat().st_size<64*1024**2,
        'Bounded actual JSON only, no array decoding')
    return json.loads(path.read_text())


def merge(target,items):
    for path,digest in items.items():
        key=str(path)
        require(key not in target or target[key]==digest,'Consistent exact binding')
        target[key]=digest


def bind(items,tick=None):
    for path,digest in items.items():
        require(isinstance(digest,str) and len(digest)==64 and sha(path,tick)==digest,
            'Current source/input bytes: '+str(path))


def load(path,digest,name):
    require(sha(path)==digest,'Exact inert owner')
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    no_numerics()
    return module


def chain(folder,inputs,tick=None):
    paths=[folder/name for name in NAMES]
    ds=tuple(sha(path,tick) for path in paths)
    require(ds==CHAINS[folder],'Exact existing positive three seals')
    r,e,f=map(read,paths)
    require(all(x.get('passed') is True and x.get('completed') is True for x in (r,e,f))
        and e['result_sha256']==f['result_sha256']==ds[0] and f['completion_sha256']==ds[1],
        'Actual linked positive evidence')
    merge(inputs,dict(zip(paths,ds)))
    return r,f


def watched(record,final,limit,rss):
    require(all(record.get(k) is True for k in
        ('whole_owned_tree_retired','cleanup_passed','observational_groups_retired'))
        and 0<=record['elapsed_s']<=final['elapsed_s']<limit
        and 0<record['maximum_combined_rss_bytes']<rss
        and 0<=record['maximum_interval_s']<=1 and record['maximum_swap_growth_bytes']==0,
        'Actual bounded and fully retired prerequisite watch')


def prerequisites(tick=None):
    sources,inputs,records,finals={},{},{},{}
    bind(PINS,tick);merge(sources,PINS)
    for folder in CHAINS:
        records[folder],finals[folder]=chain(folder,inputs,tick)
    local=records[HERE/'tiny_qualification_v1/local']
    remote=records[HERE/'tiny_qualification_v1/remote']
    tw=records[HERE/'tiny_qualification_watch_v1']
    watched(tw,finals[HERE/'tiny_qualification_watch_v1'],300,8_000_000_000)
    require(tw['schema']=='current_comp_shared_pair256_tiny_watch_v1'
        and tw['qualification_passed'] and tw['retired_contexts']==4 and tw['model_calls']==0
        and tw['truth_accessed'] is False and tw['full_estimator_fidelity_established'] is False,
        'Actual tiny-only numerical qualification')
    tiny=load(HERE/'tiny_worker.py',PINS[HERE/'tiny_worker.py'],'pair37_exact_tiny_verifier')
    require(tiny.compare(local,remote,tick)==tw['comparison'],'Fresh independent tiny reference/full-bill replay')
    for record in (local,remote,tw):
        merge(sources,record['source_bindings'])
        merge(inputs,record.get('input_bindings',{}));merge(inputs,record['artifacts'])
    n,w=records[GAIN/'native40_v2'],records[GAIN/'native40_watch_v2']
    watched(w,finals[GAIN/'native40_watch_v2'],3000,8_000_000_000)
    require(n['schema']=='current_comp_shared_gain_slice_native40_v2'
        and w['schema']=='current_comp_shared_gain_slice_native40_watch_v2'
        and tuple(w['child_'+name+'_sha256'] for name in ('result','completion','finalization'))
            ==CHAINS[GAIN/'native40_v2'] and w['child_execution_completed']
        and n['candidate_sha256']==w['candidate_sha256']==PARENT_SHA
        and n['full_model_calls']==n['retired_call_count']==w['retired_calls']==41
        and n['initialization_flops']==1944970298
        and all(n.get(k) is True for k in ('all16_domains_qualified','all47_selectors_qualified',
            'repeat_byte_exact','own_complete_bills','sources_unchanged','hashes_unchanged',
            'same_production_backend_qualified','backend_actual_origins_qualified'))
        and n['truth_accessed'] is False and n['score_measured'] is False,
        'Direct current-c2 acquisition boundary, not ancestral or new-candidate fidelity')
    for record in (n,w): merge(sources,record['source_bindings'])
    selected={path:digest for path,digest in n['artifacts'].items()
        if Path(path).parent==GAIN/'native40_v2' and Path(path).name.startswith(('call_00_','call_40_'))}
    require(len(selected)==14,'Exactly two current37 artifact sets')
    merge(inputs,selected)
    for key in ('row_37','weights_37'):
        item=n['fixture_bindings'][key]
        merge(inputs,{item['path']:item['sha256']})
    controls=[]
    for index,expected in ((0,406399683439),(40,404454713141)):
        trial=n['trials'][index];diag=n['diagnostic_records'][index]
        folder=GAIN/'native40_v2';marker=read(folder/f'call_{index:02d}_complete.json')
        require(trial['call']==index and trial['rank']==37 and trial['flops']==expected
            and trial['includes_initialization'] is (index==0)
            and trial['multiplier']==max(.1,expected/2**41)
            and trial['output_sha256']==selected[trial['output']]
            and marker['passed'] and marker['completed'] and marker['context_retired']
            and marker['output_sha256']==trial['output_sha256']
            and marker['diagnostic_sha256']==diag['sha256']
            and marker['bill_sha256']==selected[str(folder/f'call_{index:02d}_bill.json')],
            'Actual sealed cold/repeat37 served output and owned complete cost')
        controls.append(dict(call=index,rank=37,flops=expected,output=trial['output'],
            output_sha256=trial['output_sha256'],diagnostics=diag))
    require(controls[0]['output_sha256']==controls[1]['output_sha256'],
        'Own current-c2 repeat37 payload equality')
    bind(sources,tick);bind(inputs,tick);no_numerics()
    proof=dict(tiny_qualification_replayed=True,current_c2_native37_boundary_qualified=True,
        current_candidate_native_fidelity_qualified=False,current_candidate_raw_value_measured=False,
        controls=controls,prerequisite_seals={str(folder):dict(zip(
            ('result_sha256','completion_sha256','finalization_sha256'),digests)) for folder,digests in CHAINS.items()})
    return sources,inputs,proof


def derive_source(tick=None):
    composer=load(HERE/'compose.py',PINS[HERE/'compose.py'],'pair37_frozen_composer')
    text,proof=composer.derive_source(tick)
    require(proof['full_source_inverse_verified'] and proof['exact_contextual_edits']==8
        and proof['composed_source_sha256']==CANDIDATE_SHA and not proof['numerical_estimator_executed']
        and not proof['candidate_written'] and proof['original_learned_literals_and_heads_byte_exact'],
        'Exactly the preflighted unfit pair256 source')
    return text,dict(proof,candidate_sha256=CANDIDATE_SHA)


def write(path,value):
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    with Path(path).open('xb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()


def verify_build(build_sha,candidate_sha,builder_sha,protocol_sha,tick=None):
    own={HERE/'build37.py':builder_sha,HERE/'BUILD37_PROTOCOL.md':protocol_sha}
    bind(own,tick)
    paths=[OUTPUT/name for name in NAMES];ds=[sha(p,tick) for p in paths]
    r,e,f=map(read,paths)
    require(ds[0]==build_sha and candidate_sha==CANDIDATE_SHA
        and all(x.get('passed') is True and x.get('completed') is True for x in (r,e,f))
        and e['result_sha256']==f['result_sha256']==ds[0] and f['completion_sha256']==ds[1]
        and r['schema']==SCHEMA and r['candidate_sha256']==candidate_sha
        and r['estimator_executed'] is False and r['truth_accessed'] is False and r['parameter_updates']==0
        and 0<=r['elapsed_s']<=f['elapsed_s']<60 and 0<r['peak_rss_bytes']<2_000_000_000,
        'Actual source build three seals, no model/fit authority')
    text,proof=derive_source(tick)
    sources,inputs,prerequisite=prerequisites(tick)
    merge(sources,own)
    require(r['proof']==proof and r['prerequisite_proof']==prerequisite
        and r['source_bindings']==sources and r['input_bindings']==inputs
        and Path(r['candidate'])==OUTPUT/'estimator.py'
        and (OUTPUT/'estimator.py').read_bytes()==text.encode(), 'Exact candidate and prerequisites replay')
    bind(r['artifacts'],tick)
    require(r['artifacts']=={str(OUTPUT/'estimator.py'):candidate_sha,
        str(OUTPUT/'proof.json'):sha(OUTPUT/'proof.json',tick)} and read(OUTPUT/'proof.json')==proof,
        'Exactly candidate and proof artifacts')
    return dict(build=r,candidate=str(OUTPUT/'estimator.py'),candidate_sha256=candidate_sha,
        proof=proof,source_bindings=sources,input_bindings={**inputs,**r['artifacts'],
            **dict(zip(map(str,paths),ds))})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--source-sha256');parser.add_argument('--protocol-sha256')
    args=parser.parse_args()
    if not args.run:
        parser.print_help();return 0
    began=time.monotonic()
    def tick():
        require(time.monotonic()-began<60 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2_000_000_000,
            'Source-only60s/self2GB including publication')
    def alarm(*unused): raise TimeoutError('Original whole source60s deadline')
    signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,60)
    no_numerics()
    require(not OUTPUT.exists() and not HERE.is_symlink(),'Exclusive never-resumed source build')
    own={HERE/'build37.py':args.source_sha256,HERE/'BUILD37_PROTOCOL.md':args.protocol_sha256}
    bind(own,tick)
    sources,inputs,prerequisite=prerequisites(tick);merge(sources,own)
    text,proof=derive_source(tick);tick()
    OUTPUT.mkdir(exist_ok=False)
    r=dict(schema=SCHEMA,passed=False,completed=False,estimator_executed=False,truth_accessed=False,
        parameter_updates=0,source_bindings=sources,input_bindings=inputs,artifacts={},
        prerequisite_proof=prerequisite,automatic_next_stage=False,uploaded=False)
    try:
        with (OUTPUT/'estimator.py').open('xb') as stream:
            stream.write(text.encode());stream.flush();os.fsync(stream.fileno())
        r.update(candidate=str(OUTPUT/'estimator.py'),candidate_sha256=CANDIDATE_SHA,proof=proof)
        r['artifacts'][str(OUTPUT/'estimator.py')]=sha(OUTPUT/'estimator.py',tick)
        r['artifacts'][str(OUTPUT/'proof.json')]=write(OUTPUT/'proof.json',proof)
        for group in ('source_bindings','input_bindings','artifacts'): bind(r[group],tick)
        no_numerics();tick()
        r.update(passed=True,completed=True,sources_unchanged=True,hashes_unchanged=True)
    except BaseException as error:
        r.update(passed=False,completed=False,error_type=type(error).__name__,error=str(error)[:1000])
    r.update(elapsed_s=time.monotonic()-began,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if not r['passed']: signal.setitimer(signal.ITIMER_REAL,2)
    if r['passed']: tick()
    rh=write(OUTPUT/'result.json',r)
    eh=write(OUTPUT/'completion.json',dict(passed=r['passed'],completed=r['completed'],result_sha256=rh))
    if r['passed']: tick()
    write(OUTPUT/'finalization.json',dict(passed=r['passed'],completed=r['completed'],result_sha256=rh,
        completion_sha256=eh,elapsed_s=time.monotonic()-began))
    if r['passed']: tick()
    signal.setitimer(signal.ITIMER_REAL,0)
    print(json.dumps(dict(passed=r['passed'],result_sha256=rh,candidate_sha256=r.get('candidate_sha256'))),flush=True)
    return 0 if r['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
