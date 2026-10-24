"""Fixed independent rangefinder VJP/checkpoint qualification, no fit or score."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback
import weakref

HERE=Path(__file__).resolve().parent
NAMES={'rangefinder_vjp.py','watch_rangefinder_vjp.py','differentiable_rangefinder.py',
       'RANGEFINDER_VJP_PROTOCOL.md','rangefinder_vjp_inputs.json','rangefinder_vjp_unit.py'}


def require(ok,message):
    if not ok:raise RuntimeError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')


def load(path,name,digest):
    require(sha(path)==digest,'Reviewed inert source before import')
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def sealed(folder,pins):
    names=('result.json','completion.json','finalization.json')
    paths=[folder/k for k in names]
    require(all(str(p) in pins and sha(p)==pins[str(p)] for p in paths),'Exact prerequisite pins')
    r,c,f=map(read,paths)
    require(all(v['passed'] is True and v['completed'] is True for v in (r,c,f)) and
            c['result_sha256']==f['result_sha256']==pins[str(paths[0])] and
            f['completion_sha256']==pins[str(paths[1])],'Positive prerequisite three seals')
    return r


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pins',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();began=time.monotonic()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Whole120s')))
    signal.alarm(120)
    output=Path(args.output).resolve()
    require(output==HERE/'rangefinder_vjp_v1' and not output.exists(),'One fresh fixed VJP qualification')
    output.mkdir()
    result=dict(passed=False,completed=False,full_model_calls=0,parameter_updates=0,
                truth_accessed=False,score_measured=False,native_gradient_certified=False)
    def tick():
        require(time.monotonic()-began<120 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<8_000_000_000,
                'Whole120s/self8GB; external supervisor owns tree/swap/observations')
    try:
        pins=read(args.pins)
        require(set(pins)==NAMES and all(sha(HERE/k)==v for k,v in pins.items()),'Exact six source owners')
        inputs=read(HERE/'rangefinder_vjp_inputs.json')
        require(all(Path(k).resolve().is_relative_to(HERE) and sha(k)==v for k,v in inputs.items()),
                'Fixed local positive seals/bundle/source inputs only')
        nr=sealed(HERE/'entry8_native_v2',inputs)
        tr=sealed(HERE/'entry8_torch_v2',inputs)
        wr=sealed(HERE/'entry8_watch_v2',inputs)
        require(nr['all_owners_retired'] and nr['own_complete_bill'] and
                tr['all_owners_retired'] and tr['rangefinder_dominant_support'] and
                wr['whole_owned_tree_retired'] and wr['maximum_swap_growth_bytes']==0 and
                wr['rangefinder_dominant_support'] and len(wr['children'])==2,
                'Actual serial native/Torch attribution and retirement witness')
        for child,phase in zip(wr['children'],('native','torch')):
            folder=HERE/('entry8_'+phase+'_v2')
            require(child['phase']==phase and child['group_retired'] and
                    all(child['seals'][k]==inputs[str(folder/k)] for k in
                        ('result.json','completion.json','finalization.json')),
                    'Independent watcher joins exact positive child files')
        require(tr['native_seals']==wr['children'][0]['seals'],'Actual interprocess native-seal join')
        for name in ('bundle.bin','bundle.json'):
            path=HERE/'entry8_native_v2'/name
            require(nr['artifacts'][str(path)]==inputs[str(path)],'Actual published detached native artifact')
        prior=sealed(HERE/'rangefinder_replay_v1',inputs)
        prior_watch=sealed(HERE/'rangefinder_replay_watch_v1',inputs)
        require(prior['calls']==2 and prior['Y_byte_exact'] and prior['all_owners_retired']
                and prior['inputs_unchanged'] and prior_watch['whole_owned_tree_retired']
                and prior_watch['maximum_swap_growth_bytes']==0 and
                all(prior_watch['child_'+k+'_sha256']==inputs[str(HERE/'rangefinder_replay_v1'/(k+'.json'))]
                    for k in ('result','completion','finalization')),
                'Actual native-backend replay prerequisite joined to supervisor')
        unit=load(HERE/'rangefinder_vjp_unit.py','fixed_independent_dual_unit',pins['rangefinder_vjp_unit.py'])
        helper=load(HERE/'differentiable_rangefinder.py','reviewed_native_rangefinder',pins['differentiable_rangefinder.py'])
        context=helper.prepare(pins['differentiable_rangefinder.py'],tick)
        old=load(HERE/'entry8_diagnostic_v2.py','qualified_bundle_schema',
                 inputs[str(HERE/'entry8_diagnostic_v2.py')])
        bundle=dict(metadata=read(HERE/'entry8_native_v2/bundle.json'),
                    blob=(HERE/'entry8_native_v2/bundle.bin').read_bytes())
        old.validate_bundle(bundle)
        require(bundle['metadata']['source_proof']==nr['source_proof'] and
                bundle['metadata']['native_control']==nr['native_control'] and
                bundle['metadata']['candidate_sha256']==
                    context['native_context']['source_proof']['candidate_sha256'],
                'Exact current c2 native byte-control/source provenance')
        selected=helper.select_backend(context,tick)
        import numpy as np
        import torch
        from threadpoolctl import threadpool_info,threadpool_limits
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        refs=[]
        with threadpool_limits(limits=1):
            att=helper.attest(context,np,torch,threadpool_info,tick)
            workspace=helper.build_workspace(context,np,att)
            refs.append(weakref.ref(workspace));old.remember(refs,workspace.env)
            diagnostic=unit.evaluate(helper,context,bundle,np,torch,workspace,refs,old.remember,tick)
            old.remember(refs,workspace.env);workspace.close()
            del workspace,bundle
            gc.collect();require(all(r() is None for r in refs),'All derivative/checkpoint/input/workspace owners retired')
            final_att=helper.attest(context,np,torch,threadpool_info,tick)
            require(all(final_att['source_bindings'].get(k)==v for k,v in att['source_bindings'].items()),
                    'Previously imported backend owners unchanged')
        require(all(sha(k)==v for k,v in final_att['source_bindings'].items()) and
                all(sha(k)==v for k,v in inputs.items()) and all(sha(HERE/k)==v for k,v in pins.items()),
                'All source/input/backend identities unchanged')
        write(output/'diagnostics.json',diagnostic)
        result.update(passed=diagnostic['passed'],completed=True,all_owners_retired=True,
            unit_cases=diagnostic['unit_cases'],native_mm_calls=diagnostic['native_mm_calls'],
            rangefinder_real_algebra_VJP_qualified=diagnostic['passed'],
            complete_source_upper=diagnostic['complete_source_upper'],
            actual_deployment_bill_claimed=False,fit_authorized=False,
            source_proof=context['source_proof'],own_source_bindings=pins,
            input_bindings=inputs,backend_selection=selected,backend_attestation=final_att,
            diagnostic_sha256=sha(output/'diagnostics.json'))
        tick()
    except BaseException:
        result['error']=traceback.format_exc()
    result.update(elapsed_s=time.monotonic()-began,
                  maximum_self_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if result['elapsed_s']>=120 or result['maximum_self_rss_bytes']>=8_000_000_000:
        result.update(passed=False,completed=False,resource_error=True)
    write(output/'result.json',result);rh=sha(output/'result.json')
    write(output/'completion.json',dict(passed=result['passed'],completed=result['completed'],result_sha256=rh))
    write(output/'finalization.json',dict(passed=result['passed'],completed=result['completed'],
        result_sha256=rh,completion_sha256=sha(output/'completion.json'),elapsed_s=time.monotonic()-began))
    signal.alarm(0)
    print(json.dumps({k:result.get(k) for k in ('passed','completed','unit_cases','native_mm_calls','rangefinder_real_algebra_VJP_qualified','error')}),flush=True)
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
