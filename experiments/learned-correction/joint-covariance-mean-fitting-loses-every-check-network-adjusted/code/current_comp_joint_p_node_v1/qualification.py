"""Five target-free MPS lifecycle calls for the current-COMP joint proposal."""
import argparse
import ast
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback
import weakref

HERE=Path(__file__).resolve().parent
NEW=HERE.parent
RB=NEW/'current_comp_receiver_birth_v1'
CO=NEW/'current_bd_p_coadaptation_v1'
STORE=NEW.parent/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
OUT=HERE/'qualification_v1'
PINS={
    RB/'model.py':'af5ab0386bde2561543c631ace208f3649836d6581a04a69967705aec008c907',
    RB/'qualification_v3/result.json':'1213c0901ef6c0e9224034bbd099c7f727742c7e2336b9cda29adfbd4811496a',
    RB/'qualification_v3/completion.json':'bed253bd64803b7a625108caa40ec27a12f93de81a4d7963947af76503b92733',
    RB/'qualification_v3/finalization.json':'50462f5f3a90c10379348c3a0aa60c5556d335f19d7bd3ec26c8e64e061087ad',
    CO/'qualify.py':'1a3df2687ed3bef6ffcfc6f690f3cfe8665ad04aae4da94012c608d4871a7b20',
    STORE/'manifest.json':'9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50',
    STORE/'rows/observer_fit/0035/weights.npy':'14563b956b4e326aee081263077c0dd1dcb86037a0b7870dc58a5e33865f47f2',
    STORE/'rows/observer_fit/0100/weights.npy':'c4923171c8ddef28396aabd70335dad29141361efc13865ccb660a4a57421ee5',
}
def require(ok,message):
    if not ok:raise RuntimeError(message)
def sha(path):
    path=Path(path)
    require('truthpost' not in str(path),'No target-file access in qualification')
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',action='store_true')
    for key in ('source','model','protocol','watcher'):ap.add_argument('--'+key+'-sha256')
    args=ap.parse_args()
    if not args.run:ap.print_help();return 0
    began=time.monotonic()
    def expired(*_):raise TimeoutError('180s complete qualification')
    signal.signal(signal.SIGALRM,expired);signal.alarm(180)
    OUT.mkdir(exist_ok=False)
    result=dict(passed=False,completed=False,source_bindings={},input_bindings={},artifacts={},calls=[],
        full_model_calls=0,retired_call_count=0,parameter_updates=0,training_targets_accessed=False,
        validation_targets_accessed=False,native_gradient_qualified=False,native_score_measured=False,
        execution_integrity_passed=False,maximum_sampled_mps_driver_bytes=0)
    try:
        def pin(path,h):
            path=Path(path).resolve();require(sha(path)==h,'Source/input pin '+str(path))
            group='source_bindings' if path.suffix in ('.py','.md') else 'input_bindings'
            result[group][str(path)]=h
        for path,h in {**PINS,Path(__file__).resolve():args.source_sha256,
                HERE/'model.py':args.model_sha256,HERE/'QUALIFICATION_PROTOCOL.md':args.protocol_sha256,
                HERE/'watch_qualification.py':args.watcher_sha256}.items():pin(path,h)
        m=module(HERE/'model.py','joint_qualification_model')
        ref=module(RB/'model.py','joint_qualification_frozen_reference')
        for owner in (m,ref):
            for path,h in owner.authenticate().items():pin(path,h)
        qr,qc,qf=(read(RB/'qualification_v3'/name) for name in ('result.json','completion.json','finalization.json'))
        require(qr['passed'] and qr['completed'] and qc['passed'] and qf['passed']
            and qc['result_sha256']==PINS[RB/'qualification_v3/result.json']
            and qf['completion_sha256']==PINS[RB/'qualification_v3/completion.json'], 'Frozen reference lineage')
        require(sys.flags.optimize==0 and os.getpriority(os.PRIO_PROCESS,0)>=10
            and all(os.environ.get(k)=='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'))
            and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0'
            and os.environ.get('PYTORCH_MPS_LOW_WATERMARK_RATIO')=='0.1','Fixed nice10/BLAS/MPS environment')
        require(not any(k in sys.modules for k in ('numpy','torch','flopscope','threadpoolctl')),'Inert admission before numerics')
        import numpy as np
        import torch
        require(np.__version__=='1.26.4' and torch.__version__.split('+')[0]=='2.7.0'
            and torch.backends.mps.is_available() and 'threadpoolctl' not in sys.modules,'Qualified AP runtime')
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        recommended=int(torch.mps.recommended_max_memory());require(recommended>0,'Positive MPS capacity')
        fraction=min(1.,16_000_000_000/recommended);torch.mps.set_per_process_memory_fraction(fraction)
        result['mps_allocator']=dict(recommended_max_memory_bytes=recommended,configured_fraction=fraction)
        m.runtime();ref.runtime()
        funcs=[n for n in ast.parse((CO/'qualify.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='scalar']
        require(len(funcs)==1,'Original synthetic scalar source')
        scalar_env={'math':math};exec(compile(ast.Module(body=funcs,type_ignores=[]),str(CO/'qualify.py')+':scalar','exec'),scalar_env)
        scalar=scalar_env['scalar']
        def array(t):return t.detach().cpu().contiguous().numpy().copy()
        def rawsha(v):return hashlib.sha256(v.tobytes()).hexdigest()
        def save(name,v):
            path=OUT/(name+'.npy')
            with path.open('xb') as f:np.save(f,v,allow_pickle=False)
            h=sha(path);result['artifacts'][str(path)]=h
            return dict(path=str(path),sha256=h,raw_sha256=rawsha(v),shape=list(v.shape),dtype=str(v.dtype))
        def doc(name,v):
            path=OUT/(name+'.json');write(path,v);h=sha(path);result['artifacts'][str(path)]=h
            return dict(path=str(path),sha256=h)
        def guard():
            torch.mps.synchronize();driver=int(torch.mps.driver_allocated_memory())
            result['maximum_sampled_mps_driver_bytes']=max(result['maximum_sampled_mps_driver_bytes'],driver)
            require(driver<16_000_000_000 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<12_000_000_000
                and time.monotonic()-began<180,'180s/12GB host/16GB driver')
        def domain(card):
            d=card.diagnostics()
            require(set(d['variance_domain'])=={str(l) for l in range(16)} and
                set(d['selection_cells'])=={f'{l}:{s}' for l in range(16) for s in range(2 if l==15 else 3)},'All16/47 actual cells')
            require(all(x['finite'] and x['minimum']>=1e-10 and x['below_floor_count']==0
                for x in d['variance_domain'].values()),'No invalid variance or physical floor')
            if card._training:
                require(max(d['variance_visits'].values())>=2 and max(d['selection_visits'].values())>=2,'Actual checkpoint replay observed; unchanged early-stop semantics')
            return d
        def retire(index,refs):
            gc.collect();torch.mps.synchronize();torch.mps.empty_cache();gc.collect()
            require(all(r() is None for r in refs),'Card and output graph owners retired');guard()
            result['full_model_calls']+=1;result['retired_call_count']+=1
            print(json.dumps(dict(event='retired',call=index)),flush=True)
        cots={r:np.random.Generator(np.random.PCG64DXSM(9309098+r)).standard_normal((16,1024)).astype(np.float32) for r in (35,100)}
        result['cotangents']={str(r):save(f'cotangent_{r:04d}',v) for r,v in cots.items()}
        references={}
        for rank in (35,100):
            index=result['full_model_calls'];weights=np.load(STORE/f'rows/observer_fit/{rank:04d}/weights.npy',allow_pickle=False)
            P,A=ref.frozen('mps');theta=torch.zeros((64,64),dtype=torch.float32,device='mps')
            card=ref.card(rank,theta,P=P,A=A,device='mps',weights=weights,enabled=False)
            with torch.no_grad():raw,_=card.run(training=False)
            served=m._CO.served(raw);rv,sv=array(raw),array(served);d=domain(card)
            require(np.isfinite(rv).all() and np.isfinite(sv).all() and (sv>=0).all(),'Finite reference outputs')
            rec=dict(call=index,rank=rank,kind='reference_forward',raw=save(f'call_{index:02d}_raw',rv),
                served=save(f'call_{index:02d}_served',sv),diagnostics=doc(f'call_{index:02d}_diagnostics',d),passed=True,completed=True)
            references[rank]=(rv,sv,d);result['calls'].append(rec)
            weak=[weakref.ref(card),weakref.ref(raw)];del card,raw,served,P,A,theta,weights,rv,sv,d
            retire(index,weak)
        P,A=m.frozen('mps',trainable_P=True,trainable_A=True)
        initial={k:array(v) for k,v in P.items()};initialA=array(A)
        require(all(P[k].is_leaf and P[k].requires_grad for k in m.PKEYS) and A.is_leaf and A.requires_grad,'All12 live leaves')
        parameters=[P[k] for k in m.PKEYS]+[A]
        optimizer=torch.optim.Adam(parameters,lr=0.,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False)
        first_gradient=None
        for rank in (35,100,35):
            index=result['full_model_calls'];optimizer.zero_grad(set_to_none=True)
            weights=np.load(STORE/f'rows/observer_fit/{rank:04d}/weights.npy',allow_pickle=False)
            card=m.card(rank,P=P,A=A,device='mps',weights=weights)
            fixed_cstar=array(card.coefficients);fixed_weights_sha=rawsha(array(card.w))
            require(not card.coefficients.requires_grad and card.coefficients.grad is None,'Frozen cstar has no gradient')
            raw,_=card.run(training=True);served=m._CO.served(raw);rv,sv=array(raw),array(served)
            forward_d=card.diagnostics()
            require(rv.tobytes()==references[rank][0].tobytes() and sv.tobytes()==references[rank][1].tobytes(),'Joint null byte-exact full raw/served reference')
            loss=scalar(torch,m._CO,raw,cots[rank],card.dtype);require(bool(torch.isfinite(loss).detach().cpu()),'Finite synthetic scalar')
            loss.backward();d=domain(card)
            require(all(d[k]==forward_d[k]==references[rank][2][k] for k in ('variance_domain','selection_cells')),'Reference and every checkpoint replay agree')
            gradients={}
            for key,t in list(P.items())+[('NODE_A',A)]:
                if key in m.PKEYS or key=='NODE_A':
                    require(t.grad is not None and not t.grad.requires_grad and t.grad.grad_fn is None,'Detached owned full gradient')
                    g=array(t.grad);require(np.isfinite(g).all() and g.dtype==np.float32,'Finite FP32 gradient')
                    if key in m.INACTIVE_KEYS:require((g[0]==0).all(),'Inactive66 coordinate gradients zero')
                    gradients[key]=g
                else:require(not t.requires_grad and t.grad is None,'Fixed feature normalizers do not receive gradients')
            require(sum(x.size for x in gradients.values())==45456 and np.linalg.norm(gradients['NODE_A'])>0,'All joint coordinates and connected NODE A')
            require(all(array(v).tobytes()==initial[k].tobytes() for k,v in P.items()) and array(A).tobytes()==initialA.tobytes()
                and array(card.coefficients).tobytes()==fixed_cstar.tobytes() and rawsha(array(card.w))==fixed_weights_sha,'Parameters/cstar/weights unchanged through backward')
            rec=dict(call=index,rank=rank,kind='joint_backward',raw=save(f'call_{index:02d}_raw',rv),
                served=save(f'call_{index:02d}_served',sv),diagnostics=doc(f'call_{index:02d}_diagnostics',d),
                gradient={k:save(f'call_{index:02d}_gradient_{k}',v) for k,v in gradients.items()},
                synthetic_scalar=float(loss.detach().cpu()),current_COMP_byte_exact=True,gradient_coordinates=45456)
            if first_gradient is None:first_gradient={k:v.copy() for k,v in gradients.items()}
            elif rank==35:require(all(v.tobytes()==first_gradient[k].tobytes() for k,v in gradients.items()),'Exact repeated joint gradient bytes')
            optimizer.step()
            require(len(optimizer.state)==12 and all(torch.isfinite(v).all() for state in optimizer.state.values() for v in state.values()),'All12 finite persistent Adam states')
            require(all(array(v).tobytes()==initial[k].tobytes() for k,v in P.items()) and array(A).tobytes()==initialA.tobytes(),'Zero-LR Adam leaves all P/A bytes unchanged')
            optimizer.zero_grad(set_to_none=True);rec.update(passed=True,completed=True,zero_lr_Adam_P_A_byte_exact=True)
            result['calls'].append(rec);doc(f'call_{index:02d}_complete',rec)
            weak=[weakref.ref(card),weakref.ref(raw)];del card,raw,served,loss,weights,rv,sv,d,forward_d,gradients
            retire(index,weak)
        optimizer.state.clear();parameters.clear();del optimizer,P,A;gc.collect();guard()
        require(result['full_model_calls']==result['retired_call_count']==5,'Complete fixed5-call lifecycle')
        for group in ('source_bindings','input_bindings','artifacts'):
            require(all(sha(path)==h for path,h in result[group].items()),'Unchanged '+group)
        result.update(passed=True,completed=True,execution_integrity_passed=True,
            sources_unchanged=True,hashes_unchanged=True,all_graphs_retired=True,
            reference_forwards=2,joint_backwards=3,zero_lr_Adam_calls=3,repeat_gradient_byte_exact=True)
    except BaseException:result.update(passed=False,completed=False,error=traceback.format_exc())
    result['elapsed_s']=time.monotonic()-began
    write(OUT/'result.json',result);h=sha(OUT/'result.json')
    write(OUT/'completion.json',dict(passed=result['passed'],completed=result['completed'],result_sha256=h))
    end=sha(OUT/'completion.json');elapsed=time.monotonic()-began;ok=result['passed'] and elapsed<180
    write(OUT/'finalization.json',dict(passed=ok,completed=ok,result_sha256=h,completion_sha256=end,elapsed_s=elapsed))
    signal.alarm(0);return 0 if ok else 1
if __name__=='__main__':raise SystemExit(main())
