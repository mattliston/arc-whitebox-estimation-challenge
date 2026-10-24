"""One fixed 150-update fit; sealed terminal predictions, then gated scoring."""
import json,math,resource,time
from common import HERE,TRAIN,CHECK,sha,read,write,seal_check
seal_check('fit_sources_sealed.json')
from model import np,torch,KEYS,TRUNK,HEAD,runtime,initialize,inputs,forward,export,retire

started=time.monotonic();runtime_info=runtime();out=HERE/'fit';out.mkdir(exist_ok=False)
source=read(HERE/'fit_inputs.json');rows={r['rank']:r for r in source['rows']}
assert list(rows)==TRAIN+CHECK
assert sha(source['parent_P']['path'])==source['parent_P']['sha256']
with np.load(source['parent_P']['path'],allow_pickle=False) as z:P0={k:z[k].copy() for k in z.files}

# Seal all permutations before opening even a TRAIN target archive.
rng=np.random.Generator(np.random.PCG64DXSM(2026092009))
schedule=[list(map(int,rng.permutation(TRAIN))) for _ in range(30)]
assert all(sorted(s)==sorted(TRAIN) for s in schedule)
write(out/'schedule.json',dict(seed=2026092009,bit_generator='PCG64DXSM',epochs=schedule,batch_size=4,updates=150))
write(out/'pretarget_seal.json',dict(source_seal_sha256=sha(HERE/'fit_sources_sealed.json'),schedule_sha256=sha(out/'schedule.json'),
    all24_features_sealed=True,targets_decoded=0,terminal_checkpoint_selection=False))
access=(out/'target_accesses.jsonl').open('x')
targets={}
def target(rank,allow_check=False):
    assert rank in TRAIN or (allow_check and read(out/'train_score.json')['passed'])
    if rank not in targets:
        row=rows[rank];assert sha(row['target_file'])==row['target_sha256']
        access.write(json.dumps(dict(rank=rank,role=row['role'],sha256=row['target_sha256'],elapsed_s=time.monotonic()-started))+'\n');access.flush()
        with np.load(row['target_file'],allow_pickle=False) as z:truth=z['mc_post'][-1].astype(np.float64)
        assert truth.shape==(1024,) and np.isfinite(truth).all()
        targets[rank]=truth
    return targets[rank]
def load_features(rank):
    rec=rows[rank]['capture'];assert sha(rec['path'])==rec['sha256']
    with np.load(rec['path'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    w=np.load(rec['weights'],allow_pickle=False)
    return a,w

# Hash each existing weight only once; no new dense files or powers are cached.
for rank in TRAIN+CHECK:
    rec=rows[rank]['capture'];assert sha(rec['weights'])==rec['weights_sha256']
p,fixed=initialize(P0,'mps',True)
initial=export(P0,p);assert all(initial[k].tobytes()==P0[k].tobytes() for k in P0);del initial
optimizer=torch.optim.AdamW([dict(params=[p[k] for k in TRUNK],lr=5e-5),dict(params=[p[k] for k in HEAD],lr=5e-4)],
    betas=(.9,.999),eps=1e-8,weight_decay=2e-5,foreach=False)
multipliers={r:max(.1,rows[r]['capture']['parent_flops']/2**41) for r in TRAIN}
mean_multiplier=sum(multipliers.values())/len(TRAIN)
step=0;backwards=0;progress=[];sampled_driver=[]
for epoch,order in enumerate(schedule):
    losses=[];norms=[]
    for start in range(0,20,4):
        c=(1+math.cos(math.pi*step/149))/2
        optimizer.param_groups[0]['lr']=5e-6+(5e-5-5e-6)*c
        optimizer.param_groups[1]['lr']=5e-5+(5e-4-5e-5)*c
        optimizer.zero_grad(set_to_none=True)
        for rank in order[start:start+4]:
            a,w=load_features(rank);data=inputs(a,w,'mps')
            truth=torch.tensor(target(rank).astype(np.float32),device='mps')
            served,delta,_=forward(data,p,fixed)
            loss=torch.mean((served-truth)**2)/2e-8*(multipliers[rank]/mean_multiplier)/4
            assert bool(torch.isfinite(loss));loss.backward();backwards+=1
            losses.append(float(loss.detach().cpu())*4)
            del served,delta,loss,truth,data,a,w
            torch.mps.synchronize()
        norm=torch.nn.utils.clip_grad_norm_([p[k] for k in KEYS],5.,error_if_nonfinite=True,foreach=False)
        norms.append(float(norm.detach().cpu()));optimizer.step();step+=1
        del norm
        torch.mps.synchronize();sampled_driver.append(int(torch.mps.driver_allocated_memory()));retire()
    record=dict(epoch=epoch+1,updates=step,backwards=backwards,online_normalized_loss=sum(losses)/len(losses),
                max_unclipped_gradient_norm=max(norms),elapsed_s=time.monotonic()-started)
    progress.append(record);print(json.dumps(record),flush=True)
    write(out/f'epoch_{epoch+1:02d}.json',record)
assert step==150 and backwards==600
terminal=export(P0,p);payload=out/'terminal_P.npz';np.savez(payload,**terminal)
write(out/'terminal_sealed.json',dict(path=str(payload),sha256=sha(payload),updates=step,backwards=backwards,
    source_seal_sha256=sha(HERE/'fit_sources_sealed.json'),schedule_sha256=sha(out/'schedule.json'),
    trainable_scalars=44210,frozen_fields_byte_exact=True,checkpoint_selection=False))
optimizer.zero_grad(set_to_none=True);del optimizer,p,fixed,terminal;retire()

# Fresh nontrainable endpoint. This loop opens no target and forward has no
# target argument. Native FP64 host reconstruction defines cached served means.
with np.load(payload,allow_pickle=False) as z:terminal={k:z[k].copy() for k in z.files}
p,fixed=initialize(terminal,'mps',False);assert all(not v.requires_grad for v in p.values())
prediction_dir=out/'predictions';prediction_dir.mkdir();predictions=[]
for rank in TRAIN+CHECK:
    a,w=load_features(rank);data=inputs(a,w,'mps')
    with torch.no_grad():served,delta,_=forward(data,p,fixed)
    dd=delta.cpu().numpy().copy();gpu_served=served.cpu().numpy().copy()
    answer=np.maximum(a['final_host']+dd.astype(np.float64),0).astype(np.float32)
    assert np.isfinite(answer).all() and np.isfinite(dd).all()
    diff=answer.astype(np.float64)-gpu_served.astype(np.float64)
    path=prediction_dir/f'rank{rank:04d}.npz';np.savez(path,prediction=answer,delta=dd,training_precision_served=gpu_served)
    predictions.append(dict(rank=rank,path=str(path),sha256=sha(path),capture_sha256=rows[rank]['capture']['sha256'],
        fp64_host_vs_training_precision_maxabs=float(np.max(np.abs(diff))),forward_targets_provided=False))
    del a,w,data,served,delta,dd,gpu_served,answer,diff;retire()
write(out/'predictions_sealed.json',dict(completed=True,records=predictions,terminal_sha256=sha(payload),
    source_seal_sha256=sha(HERE/'fit_sources_sealed.json'),all24_forward_targets_provided=False,
    native_port_still_required=True))
del p,fixed,terminal;retire()
prediction_map={r['rank']:r for r in predictions}
def score(roster,allow_check=False):
    records=[]
    for rank in roster:
        rr=prediction_map[rank];assert sha(rr['path'])==rr['sha256']
        with np.load(rr['path'],allow_pickle=False) as z:answer=z['prediction'].astype(np.float64)
        with np.load(rows[rank]['capture']['path'],allow_pickle=False) as z:parent=z['prediction'][-1].astype(np.float64)
        truth=target(rank,allow_check);raw=float(np.mean((answer-truth)**2));baseline=float(np.mean((parent-truth)**2))
        F=rows[rank]['capture']['parent_flops'];parent_adjusted=baseline*max(.1,F/2**41)
        adjusted=raw*max(.1,(F+8_000_000_000)/2**41)
        records.append(dict(rank=rank,raw=raw,parent_raw=baseline,parent_flops=F,provisional_added_flops=8_000_000_000,
            adjusted=adjusted,parent_adjusted=parent_adjusted,ratio=adjusted/parent_adjusted,raw_ratio=raw/baseline))
    adj=sum(r['adjusted'] for r in records)/len(records);par=sum(r['parent_adjusted'] for r in records)/len(records)
    return dict(records=records,adjusted=adj,parent_adjusted=par,ratio=adj/par,wins=sum(r['ratio']<1 for r in records),
                max_individual_ratio=max(r['ratio'] for r in records),provisional_added_flops=8_000_000_000,
                deployed_bill=False,predictions_seal_sha256=sha(out/'predictions_sealed.json'))
train_score=score(TRAIN);train_score['passed']=train_score['ratio']<=.95
write(out/'train_score.json',train_score);print(json.dumps(dict(stage='TRAIN',**{k:v for k,v in train_score.items() if k!='records'})),flush=True)
check_score=None
if train_score['passed']:
    check_score=score(CHECK,True);check_score['passed']=check_score['ratio']<=.97 and check_score['wins']>=3 and check_score['max_individual_ratio']<=1.02
    write(out/'check_score.json',check_score);print(json.dumps(dict(stage='CHECK',**{k:v for k,v in check_score.items() if k!='records'})),flush=True)
access.close()
assert set(targets)==set(TRAIN+(CHECK if train_score['passed'] else []))
write(out/'result.json',dict(completed=True,passed=bool(check_score and check_score['passed']),train_passed=train_score['passed'],
    check_targets_accessed=bool(check_score),runtime=runtime_info,updates=step,backwards=backwards,terminal_sha256=sha(payload),
    train_ratio=train_score['ratio'],check_ratio=check_score['ratio'] if check_score else None,
    source_seal_sha256=sha(HERE/'fit_sources_sealed.json'),elapsed_s=time.monotonic()-started,
    sampled_driver_bytes_max=max(sampled_driver),driver_samples_are_not_transient_peak=True,
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,new_native_calls=0,native_candidate_built=False))
assert sum(f.stat().st_size for f in HERE.rglob('*') if f.is_file())<220_000_000
