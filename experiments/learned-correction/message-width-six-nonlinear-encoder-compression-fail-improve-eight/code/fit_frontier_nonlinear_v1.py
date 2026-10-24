"""Fixed, target-free, per-layer nonlinear compression refinement.

Eight BLAS1 workers; train4/teacher-validation3; no truth. Each layer retains
the student's dimensions and state64. State, existing head-output and retained
message-source fidelity receive equal initial-error-normalized loss weights.
"""
import fit_frontier_v1 as f
import argparse
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import time
import numpy as np

STEPS = 3000
RATE = 1e-4
ANCHOR = 1e-4
WORKERS = 8
FIELDS = ('enc0_w','enc0_b','enc2_w','enc2_b','enc4_w','enc4_b')
LAYERS = None
INITIAL = None


def activation(z):
    s = 1/(1+np.exp(-np.clip(z,-60,60)))
    return z*s, s+z*s*(1-s)


def forward(data, p):
    h1,d1 = activation(data['x']@p['enc0_w'].T + data['extra_bias'] + p['enc0_b'])
    h2,d2 = activation(h1@p['enc2_w'].T + p['enc2_b'])
    return h2@p['enc4_w'].T+p['enc4_b'], (h1,d1,h2,d2)


def metrics(data, p):
    y,_ = forward(data,p);err=y-data['target'];head=(err@data['head'])*data['scale'];msg=err@data['msg_w'].T
    return dict(state_mse=float(np.mean(err.astype('f8')**2)), head_delta_mse=float(np.mean(head.astype('f8')**2)),
                message_mse=float(np.mean(msg.astype('f8')**2)))


def fit_layer(layer):
    started=time.monotonic();data=LAYERS[layer]['train'];val=LAYERS[layer]['valid']
    p={k:v.copy() for k,v in INITIAL.items()};p0={k:v.copy() for k,v in p.items()}
    m={k:np.zeros_like(v) for k,v in p.items()};v={k:np.zeros_like(x) for k,x in p.items()}
    before=metrics(data,p);valid_before=metrics(val,p)
    norm=np.asarray([max(before[k],1e-24) for k in ('state_mse','head_delta_mse','message_mse')],dtype='f8')
    anchor_scale={k:max(float(np.mean(x.astype('f8')**2)),1e-4) for k,x in p.items()}
    trace=[];N=len(data['x']);head=data['head'];mw=data['msg_w'];last_loss=None
    for step in range(1,STEPS+1):
        y,(h1,d1,h2,d2)=forward(data,p);err=y-data['target'];eh=err@head;em=err@mw.T
        loss=float((np.mean(err.astype('f8')**2)/norm[0]+np.mean((eh*data['scale']).astype('f8')**2)/norm[1]+np.mean(em.astype('f8')**2)/norm[2])/3)
        dy=(err*np.float32(2/(3*N*64*norm[0])) + ((eh*data['scale']**2)[:,None]*head[None,:])*np.float32(2/(3*N*norm[1])) + (em@mw)*np.float32(2/(3*N*mw.shape[0]*norm[2])))
        g={};g['enc4_w']=dy.T@h2;g['enc4_b']=dy.sum(axis=0)
        dz2=(dy@p['enc4_w'])*d2;g['enc2_w']=dz2.T@h1;g['enc2_b']=dz2.sum(axis=0)
        dz1=(dz2@p['enc2_w'])*d1;g['enc0_w']=dz1.T@data['x'];g['enc0_b']=dz1.sum(axis=0)
        for k in FIELDS:
            g[k]+=np.float32(2*ANCHOR/(p[k].size*anchor_scale[k]))*(p[k]-p0[k])
            m[k]=.9*m[k]+.1*g[k];v[k]=.999*v[k]+.001*g[k]*g[k]
            p[k]-=np.float32(RATE)*(m[k]/np.float32(1-.9**step))/(np.sqrt(v[k]/np.float32(1-.999**step))+np.float32(1e-8))
        if step==1 or step%250==0:
            assert np.isfinite(loss) and all(np.isfinite(x).all() for x in p.values())
            trace.append(dict(step=step,normalized_train_loss=loss))
        last_loss=loss
        if step>=500 and loss<=.10:
            break
    return dict(layer=layer,steps=step,wall_s=time.monotonic()-started,train_before=before,valid_before=valid_before,
                train_after=metrics(data,p),valid_after=metrics(val,p),trace=trace,
                final_normalized_train_loss=last_loss,parameters=f.encode(p))


def rss(pid):
    try:
        for line in Path('/proc/%d/status'%pid).read_text().splitlines():
            if line.startswith('VmRSS:'):return int(line.split()[1])*1024
    except FileNotFoundError:
        pass
    return 0


def main():
    global LAYERS,INITIAL
    ap=argparse.ArgumentParser()
    for key in ('captures','input-manifest','out'):
        ap.add_argument('--'+key,type=Path,required=True)
    a=ap.parse_args();assert f.sha(f.__file__)=='b9bc4ed11d70170db6d3909bd670e97616c38e8ab455bdc99ca80513c9faf363'
    a.out.mkdir(parents=True,exist_ok=False)
    f.jwrite(a.out/'PREREG_v1.json',dict(utc=f.utc(),source_sha256=f.sha(__file__),train=f.TRAIN,valid=f.VALID,
      source_candidate='enc64_msg8_teacher',steps=STEPS,rate=RATE,anchor=ANCHOR,workers=WORKERS,blas_threads=1,
      train_stop=dict(min_steps=500,normalized_loss_at_most=.10),parameter_fields=FIELDS,
      objective='Equal initial-train-error-normalized state64, existing head delta, retained message-source MSE',
      no_validation_selection=True,no_network_truth=True,reference_submission=337393,maximum_combined_fit_rss_bytes=7*1024**3))
    manifest=json.loads(a.input_manifest.read_text());spec=next(c for c in manifest['candidates'] if c['name']=='enc64_msg8_teacher')
    tail=Path(spec['tail']);assert f.sha(tail)==spec['tail_sha256'];lit,_=f.literals_and_functions(tail.read_text());Q=f.decode(lit['_GNN_LITERAL'])
    rows,paths,meta=f.collect(a.captures);P=f.decode(meta['gnn_literal'])
    keep1=np.asarray(spec['retained_first'],dtype=int);keep=[np.asarray(x,dtype=int) for x in spec['retained_by_power']]
    cols=np.concatenate([np.arange(30)]+[30+16*g+keep[g%3] for g in range(6)])
    LAYERS={l:{} for l in range(1,16)}
    for label,ranks in [('train',f.TRAIN),('valid',f.VALID)]:
        selected=[r for r in rows if r['rank'] in ranks];mat=f.matrices(selected,P)
        labels=np.concatenate([np.full(128,r['layer'],dtype=int) for r in selected]);sigma=np.concatenate([f.array(r,'sigma') for r in selected])
        for l in range(1,16):
            mask=labels==l;LAYERS[l][label]=dict(x=mat['H'][mask][:,cols].copy(),extra_bias=(mat['bias'][mask][:,keep1]-P['enc0_b'][keep1]).copy(),
              target=mat['state'][mask].copy(),head=Q['head'][l].copy(),msg_w=Q['msg_w'].copy(),
              scale=(sigma[mask]*P['rms'][l]*(1+.25*np.tanh(P['gain'][l]))).copy())
    INITIAL={k:Q[k].copy() for k in FIELDS}
    del rows,selected,mat,labels,sigma;gc.collect()
    try:
        import ctypes
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except Exception:
        pass
    pool=mp.get_context('fork').Pool(WORKERS);pids=[p.pid for p in pool._pool]
    f.jwrite(a.out/'RUNNING_v1.json',dict(utc=f.utc(),parent_pid=os.getpid(),worker_pids=pids,blas_threads=1,workers=WORKERS))
    print(json.dumps(dict(event='workers_started',parent_pid=os.getpid(),worker_pids=pids)),flush=True)
    pending={l:pool.apply_async(fit_layer,(l,)) for l in range(1,16)};results=[];peak=0
    try:
        while pending:
            total=rss(os.getpid())+sum(rss(pid) for pid in pids);peak=max(peak,total)
            assert total<=7*1024**3,('combined fitter RSS exceeded fixed7GiB ceiling',total)
            for l,job in list(pending.items()):
                if job.ready():
                    result=job.get();f.jwrite(a.out/('L%02d_RESULT_v1.json'%l),result);results.append(result);del pending[l]
                    print(json.dumps(dict(event='layer_done',layer=l,steps=result['steps'],wall_s=result['wall_s'],
                          train=result['train_after'],valid=result['valid_after'])),flush=True)
            if pending:time.sleep(1)
        pool.close();pool.join()
    except BaseException:
        pool.terminate();pool.join();raise
    f.jwrite(a.out/'FIT_COMPLETE_v1.json',dict(utc=f.utc(),status='NONLINEAR_TEACHER_FIT_NO_NETWORK_SCORE',source_spec=spec,
       source_manifest_sha256=f.sha(a.input_manifest),layers=sorted(results,key=lambda x:x['layer']),peak_combined_fit_rss_bytes=peak,
       capture_files=paths,reference_submission=337393,export_and_native_checks_pending=True))


if __name__=='__main__':main()
