"""Retired source admission and five fully metered fixed native calls."""
import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
NEW,P2=HERE.parent,HERE.parent.parent
COMP=NEW/'current_p1_node_composition_v1'
RANKS=(37,39,42,48)
OWNERS={
    COMP/'native.py':'fafc28732701aacfa70ada15d30ed083af76e91747bc01dc840ec8ba14d23d01',
    NEW/'current_d50_response10_candidate_v1/native.py':'cccd8b50d18e6d093a48a3e3675aa20cbe2ef31cb45c2f167f153ddc842fe09c',
    NEW/'birth_direction_fit_v1/native_v2.py':'8a1b888be28a7ba6d7f8e020bb9b1f2367786e189d952364710b34517ca2b35b',
    NEW/'current_d50_terminal_linear_v2/native.py':'744f8c46bfdbe0198eb138c1fcacc5a0b90ca6db6eb08ad7d928847c34021abe',
    NEW/'current_d50_terminal_linear_v2/common.py':'52467763b2131ad325869ea8e1fffb214ca2a89ab7bd24b3150a3350f5ac3f2a',
    COMP/'native_terminal_v2/result.json':'a9071e5afa6dbf35e6a6871d6997a7dbc8a1105d2bf3e97bd172c7a1b0e0180f',
    COMP/'native_terminal_v2/completion.json':'6c050c04fd565d840273f2a917f2a106182cfed1b69f59a09cd5392497706772',
}
def require(ok,message):
    if not ok:raise RuntimeError(message)
def sha(path):
    path=Path(path);require('truthpost' not in str(path),'No target-file access')
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def extract(path,name):
    source=Path(path).read_text();tree=ast.parse(source)
    found=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name]
    require(len(found)==1,'Unique source owner '+name)
    n=found[0];return ''.join(source.splitlines(keepends=True)[n.lineno-1:n.end_lineno])
def splice(source,changes):
    result=source
    for a,b in changes:require(result.count(a)==1,'Unique source span');result=result.replace(a,b)
    inverse=result
    for a,b in reversed(changes):require(inverse.count(b)==1,'Unique inverse span');inverse=inverse.replace(b,a)
    require(inverse==source,'Exact source inverse');return result
def prepare(n):
    original=extract(COMP/'native.py','Observer')
    observer=splice(original,[
        ('        self.begin()\n','        self.begin()\n'
         '        self.interested_code_ids = frozenset(id(x) for x in self.codes.values()) | frozenset(\n'
         "            id(getattr(self.small,k)) for k in ('predict_code','gaussian_code','select_code','gnn_code'))\n"
         "        require(len(self.interested_code_ids) == 8, 'Exactly eight original observed code identities')\n"),
        ('        self.small(frame,event,answer)\n',
         "        if event not in ('call','return') or id(frame.f_code) not in self.interested_code_ids:\n"
         '            return\n        self.small(frame,event,answer)\n')])
    env=dict(vars(n),FIXED={k:RANKS for k in ('p_only','joint')})
    exec(compile(observer,str(__file__)+':exact_filtered_observer','exec'),env)
    exec(compile(extract(COMP/'native.py','finish_native'),str(__file__)+':original_finish','exec'),env)
    paid,original_sha=n.derived_main()
    changed=splice(paid,[
        ('                del mlp,served,materialized,values\n                gc.collect()\n',
         '                del mlp,served,materialized,values\n                gc.collect()\n'
         "                print(json.dumps(dict(event='retired',call=call)),flush=True)\n")])
    proof=dict(observer_original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        observer_filtered_sha256=hashlib.sha256(observer.encode()).hexdigest(),observer_inverse_verified=True,
        original_paid_main_sha256=original_sha,prior_paid_sha256=hashlib.sha256(paid.encode()).hexdigest(),
        paid_with_retirement_marker_sha256=hashlib.sha256(changed.encode()).hexdigest(),paid_loop_inverse_verified=True,
        changed_paid_statements='one post-release stdout marker only',new_candidate_arithmetic=False)
    return env['Observer'],env['finish_native'],changed,proof

def arguments():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',action='store_true');ap.add_argument('--admit',action='store_true')
    ap.add_argument('--arm',choices=('p_only','joint'))
    for k in ('source','model','protocol','watcher','builder','build-protocol','build','candidate','context'):ap.add_argument('--'+k+'-sha256')
    a=ap.parse_args()
    if not a.run and not a.admit:ap.print_help();return None
    require(a.arm in ('p_only','joint'),'Fixed arm')
    return a

def admit(a):
    began=time.monotonic()
    def guard(*_):require(time.monotonic()-began<60 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2_000_000_000,'60s/2GB source-only admission')
    signal.signal(signal.SIGALRM,guard);signal.setitimer(signal.ITIMER_REAL,.5,.5)
    folder=HERE/f'native_{a.arm}_admission_v1';folder.mkdir(exist_ok=False)
    result=dict(passed=False,completed=False,arm=a.arm,admission_pid=os.getpid(),numerical_imports=False,truth_accessed=False)
    try:
        require(os.getpriority(os.PRIO_PROCESS,0)>=10 and sys.flags.optimize==0,'Nice10,checks enabled')
        owners={**OWNERS,Path(__file__).resolve():a.source_sha256,HERE/'NATIVE_PROTOCOL.md':a.protocol_sha256,
            HERE/'watch_native.py':a.watcher_sha256,HERE/'build.py':a.builder_sha256,HERE/'BUILD_PROTOCOL.md':a.build_protocol_sha256}
        require(all(sha(p)==h for p,h in owners.items()),'Reviewed native/build source owners')
        builder=module(HERE/'build.py','joint_native_source_builder')
        context=builder.verify_build(a.arm,a.build_sha256,a.candidate_sha256,a.builder_sha256,a.build_protocol_sha256)
        n=module(COMP/'native.py','joint_native_source_owner');_,_,_,proof=prepare(n)
        sources={str(p.resolve()):h for p,h in owners.items()};sources.update(context['source_bindings'])
        inputs=dict(context['input_bindings'])
        baseline=read(COMP/'native_terminal_v2/result.json');end=read(COMP/'native_terminal_v2/completion.json')
        require(baseline['passed'] and baseline['completed'] and baseline['own_complete_bills'] and baseline['all16_domains_qualified']
            and baseline['all47_selectors_qualified'] and end['passed'] and end['result_sha256']==OWNERS[COMP/'native_terminal_v2/result.json'], 'Exact current COMP comparator')
        refs={}
        for trial,prepared in zip(baseline['trials'],baseline['input_preparation']):
            if trial['rank'] in RANKS and trial['rank'] not in refs:
                require(trial['rank']==prepared['rank'] and trial['call']==prepared['call'],'Input/prediction ownership')
                require(sha(trial['output'])==trial['output_sha256']==baseline['artifacts'][trial['output']],'Frozen baseline output bytes')
                refs[trial['rank']]=dict(trial,weight_file_sha256=prepared['weight_file_sha256'])
                inputs[trial['output']]=trial['output_sha256']
        require(set(refs)==set(RANKS),'All fixed four baseline rows')
        candidate=HERE/f'build_{a.arm}_v1/estimator.py';require(sha(candidate)==a.candidate_sha256,'Exact terminal candidate')
        sources[str(candidate)]=a.candidate_sha256
        document=dict(arm=a.arm,ranks=list(RANKS),candidate=str(candidate),candidate_sha256=a.candidate_sha256,
            build_sha256=a.build_sha256,source_bindings=sources,input_bindings=inputs,reference_by_rank=refs,
            proof=proof,admission_pid=os.getpid(),source_admission_complete=True,truth_accessed=False,numerical_imports=False)
        require(len(json.dumps(document).encode())<4_000_000,'Bounded metadata-only admission context')
        require(not any(k in sys.modules for k in ('numpy','torch','flopscope','threadpoolctl')),'Admission remained stdlib-only')
        require(all(sha(p)==h for p,h in {**sources,**inputs}.items()),'All admitted bytes unchanged');guard()
        write(folder/'context.json',document)
        result.update(passed=True,completed=True,context_sha256=sha(folder/'context.json'),sources_unchanged=True)
    except BaseException:result.update(error=traceback.format_exc())
    result['elapsed_s']=time.monotonic()-began
    result['peak_rss_bytes']=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if result['elapsed_s']>=60 or result['peak_rss_bytes']>=2_000_000_000:result.update(passed=False,completed=False)
    write(folder/'result.json',result)
    write(folder/'completion.json',dict(passed=result['passed'],completed=result['completed'],result_sha256=sha(folder/'result.json')))
    elapsed=time.monotonic()-began;ok=bool(result['passed'] and elapsed<60)
    write(folder/'finalization.json',dict(passed=ok,completed=ok,result_sha256=sha(folder/'result.json'),
        completion_sha256=sha(folder/'completion.json'),elapsed_s=elapsed))
    signal.setitimer(signal.ITIMER_REAL,0);print(json.dumps(result,indent=2));return 0 if ok else 1

def run(a):
    began=time.monotonic();folder=HERE/f'native_{a.arm}_v1'
    ap=HERE/f'native_{a.arm}_admission_v1';require(sha(ap/'context.json')==a.context_sha256,'Reviewed retired admission context')
    context=read(ap/'context.json');ar,ae,af=(read(ap/name) for name in ('result.json','completion.json','finalization.json'))
    require(ar['passed'] and ar['completed'] and ae['passed'] and ae['result_sha256']==sha(ap/'result.json')
        and ar['elapsed_s']<60 and ar['peak_rss_bytes']<2_000_000_000 and af['passed'] and af['completed']
        and af['elapsed_s']<60 and af['result_sha256']==sha(ap/'result.json')
        and af['completion_sha256']==sha(ap/'completion.json')
        and ar['context_sha256']==a.context_sha256 and context['arm']==a.arm and context['source_admission_complete'],'Complete context seals')
    try:os.kill(context['admission_pid'],0)
    except ProcessLookupError:pass
    else:raise RuntimeError('Full-admission process must be gone before numeric worker')
    require(context['source_bindings'][str(Path(__file__).resolve())]==a.source_sha256
        and context['source_bindings'][str(HERE/'model.py')]==a.model_sha256
        and context['source_bindings'][str(HERE/'NATIVE_PROTOCOL.md')]==a.protocol_sha256
        and context['source_bindings'][str(HERE/'watch_native.py')]==a.watcher_sha256,'Exact reviewed numerical/supervisor owners')
    require(all(sha(p)==h for p,h in {**context['source_bindings'],**context['input_bindings']}.items()),'Unchanged admitted source/input bytes')
    require(not any(k in sys.modules for k in ('numpy','torch','flopscope','threadpoolctl')),'All admission before numerical imports')
    versions={k:importlib.metadata.version(k) for k in ('numpy','flopscope','whestbench','threadpoolctl')}
    require(versions==dict(numpy='2.4.6',flopscope='0.12.0',whestbench='0.16.0',threadpoolctl='3.6.0'),'Official native versions')
    n=module(COMP/'native.py','joint_native_actual_owner');n.load_diagnostics()
    observer,finish,paid,proof=prepare(n);require(proof==context['proof'],'Identical admitted inverse source')
    old=n.legacy();candidate=Path(context['candidate']);require(not folder.exists(),'Exclusive complete native roster')
    pins={**context['source_bindings'],**context['input_bindings'],str(ap/'context.json'):a.context_sha256,
        str(ap/'result.json'):sha(ap/'result.json'),str(ap/'completion.json'):sha(ap/'completion.json'),
        str(ap/'finalization.json'):sha(ap/'finalization.json')}
    def after(np,observed,trial,result,out):n.after_bill(np,observed,trial,result,out,{int(k):v for k,v in context['reference_by_rank'].items()})
    def done(result):
        finish(result);result.update(full_model_calls=5,retired_call_count=5)
    header=dict(schema='current_comp_joint_p_node_native_v1',stage='native',role=a.arm,arm=a.arm,completed=False,
        sources_unchanged=False,build_sha256=context['build_sha256'],versions=versions,truth_accessed=False,no_fit=True,score_measured=False,
        source_inverse_verified=True,paid_arithmetic_and_lifecycle_unchanged=True,own_complete_bills=False,
        full_admission_in_retired_process=True,admission_context_sha256=a.context_sha256,proof=proof,
        response_floor=.01,active_floor=.01,birth_floor=.01)
    namespace=dict(vars(old),__file__=str(Path(__file__).resolve()),PARENT=NEW,CANDIDATE=candidate,NATIVE_PINS=pins,
        FIXED_RANKS=RANKS,Observer=observer,observed_predict=n.observed_predict,after_bill=after,finish_native=done,NATIVE_HEADER=header)
    os.chdir(NEW);saved=sys.argv
    try:
        sys.argv=[str(__file__),'--source',str(candidate),'--split','validation','--out',str(folder)]
        exec(compile(paid,str(__file__)+':unchanged_paid_native','exec'),namespace);code=namespace['main']()
    finally:sys.argv=saved
    record=read(folder/'result.json');completion=folder/'completion.json';elapsed=time.monotonic()-began
    passed=bool(code==0 and record['passed'] and record['completed'] and elapsed<600)
    write(folder/'finalization.json',dict(passed=passed,completed=passed,result_sha256=sha(folder/'result.json'),
        completion_sha256=sha(completion),elapsed_s=elapsed))
    return 0 if passed else 1
if __name__=='__main__':
    a=arguments()
    if a is not None:raise SystemExit(admit(a) if a.admit else run(a))
