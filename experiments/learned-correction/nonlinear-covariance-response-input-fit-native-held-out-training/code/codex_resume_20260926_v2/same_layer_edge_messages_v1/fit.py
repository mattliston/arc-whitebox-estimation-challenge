"""Fixed twelve-update native-anchored pilot; only the terminal weights are exported."""
from pathlib import Path
import hashlib
import json
import math
import resource
import sys
import time
import model

HERE=Path(__file__).resolve().parent;H2=HERE.parent;P2=H2.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
mode=sys.argv[1];assert mode in ('message','control')
out=HERE/f'fit_{mode}';out.mkdir()
protocol=model.PROTOCOL
for name in ('qualification','capture','native_zero'):
    receipt=json.loads((HERE/f'{name}_guard.json').read_text())
    assert receipt['completed'] and receipt['whole_group_retired'] and receipt['max_swap_growth_bytes']==0
cal=json.loads((HERE/'calibration.json').read_text())
assert cal['protocol_sha256']==sha(HERE/'protocol.json')
assert cal['capture_sha256']==sha(HERE/'capture.py') and cal['rows_sha256']==sha(HERE/'capture_rows.jsonl')
api=model.initialize();t=api['torch'];np=api['np']
qualified=json.loads((HERE/'qualification.json').read_text())['binding']
assert api['binding']==qualified
stats=cal['statistics'][mode]
mean=t.tensor(stats['mean'],dtype=t.float32);scale=t.tensor(stats['scale'],dtype=t.float32)
reference=H2/'response_message_capture_train12_v1'
native={r:t.from_numpy(np.load(reference/f'pred_{r}.npy')[15].copy()) for r in protocol['fit_ranks']}
truth={r:t.from_numpy(np.load(P2/'fable_refit_v1/truth688'/f'truthpost_rank{r:04d}.npz')['mc_post'][15].astype(np.float64)) for r in protocol['fit_ranks']}
capture_rows={r['rank']:r for r in map(json.loads,(HERE/'capture_rows.jsonl').read_text().splitlines())}
for r in protocol['fit_ranks']:assert sha(reference/f'pred_{r}.npy')==capture_rows[r]['reference_prediction_sha256']
truth_hashes={str(r):sha(P2/'fable_refit_v1/truth688'/f'truthpost_rank{r:04d}.npz') for r in protocol['fit_ranks']}
started=time.monotonic();base_path=HERE/'torch_baselines.npz';base_meta=HERE/'torch_baselines.json'
if not base_path.exists():
    assert mode=='message' and not base_meta.exists()
    arrays={}
    for r in protocol['fit_ranks']:
        with t.no_grad():
            w=api['weights'](r);cd=api['card'](w);pred=api['run'](cd)
            arrays[str(r)]=pred.detach().numpy().copy()
        print(json.dumps(dict(event='torch_baseline',rank=r,elapsed_s=time.monotonic()-started)),flush=True)
        del cd,w,pred;model.collect()
    with base_path.open('xb') as fp:np.savez(fp,**arrays)
    with base_meta.open('x') as fp:json.dump(dict(binding=api['binding'],sha256=sha(base_path),ranks=protocol['fit_ranks']),fp,indent=2)
else:
    bm=json.loads(base_meta.read_text());assert bm['binding']==api['binding'] and bm['sha256']==sha(base_path) and bm['ranks']==protocol['fit_ranks']
with np.load(base_path) as arrays:baseline={r:t.from_numpy(arrays[str(r)][15].copy()) for r in protocol['fit_ranks']}
rng=np.random.Generator(np.random.PCG64(2026092702))
orders=[rng.permutation(protocol['fit_ranks']).tolist() for _ in range(4)]
lrs=[.005*step/3 if step<=3 else .0005+.0045*(1+math.cos(math.pi*(step-3)/9))/2 for step in range(1,13)]
binding=dict(mode=mode,protocol_sha256=sha(HERE/'protocol.json'),fit_sha256=sha(HERE/'fit.py'),
    calibration_sha256=sha(HERE/'calibration.json'),baseline_sha256=sha(base_path),model=api['binding'],
    orders=orders,learning_rates=lrs,truth_sha256=truth_hashes,targets_accessed=protocol['fit_ranks'],
    check_or_valid_targets_accessed=False)
with (out/'manifest.json').open('x') as fp:json.dump(binding,fp,indent=2)
theta=t.zeros((2,12,16),dtype=t.float32,requires_grad=True)
opt=t.optim.Adam([theta],lr=lrs[0],betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False)
step=0;allrows=[]
with (out/'networks.jsonl').open('x') as netlog,(out/'updates.jsonl').open('x') as uplog:
    for epoch,order in enumerate(orders,1):
        for start in range(0,12,4):
            opt.zero_grad(set_to_none=True);losses=[]
            for rank in order[start:start+4]:
                t0=time.monotonic();w=api['weights'](rank);cd=api['card'](w,theta,mode,mean,scale)
                pred=api['run'](cd,checkpointed=True)
                residual=pred[-1]-baseline[rank]+native[rank]-truth[rank]
                mse=residual.square().mean();loss=mse*1e8/4
                assert t.isfinite(loss)
                loss.backward()
                assert theta.grad is not None and t.isfinite(theta.grad).all()
                row=dict(epoch=epoch,update=step+1,rank=rank,anchored_mse=float(mse.detach()),
                    native_baseline_mse=float((native[rank]-truth[rank]).square().mean()),
                    wall_s=time.monotonic()-t0,elapsed_s=time.monotonic()-started,
                    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                netlog.write(json.dumps(row)+'\n');netlog.flush();losses.append(row['anchored_mse']);allrows.append(row)
                print(json.dumps(dict(event='fit_network',mode=mode,**row)),flush=True)
                del cd,w,pred,residual,mse,loss;model.collect()
            gradnorm=t.nn.utils.clip_grad_norm_([theta],1.)
            assert t.isfinite(gradnorm)
            opt.param_groups[0]['lr']=lrs[step];opt.step();step+=1
            assert t.isfinite(theta).all()
            row=dict(step=step,epoch=epoch,lr=lrs[step-1],mean_anchored_mse=float(np.mean(losses)),
                     gradient_norm_before_clip=float(gradnorm),theta_norm=float(theta.detach().norm()),
                     theta_max_abs=float(theta.detach().abs().max()),elapsed_s=time.monotonic()-started)
            uplog.write(json.dumps(row)+'\n');uplog.flush();print(json.dumps(dict(event='update',mode=mode,**row)),flush=True)
assert step==12
with (out/'terminal.npz').open('xb') as fp:np.savez(fp,theta=theta.detach().numpy(),mean=mean.numpy(),scale=scale.numpy())
result=dict(mode=mode,completed=True,updates=step,networks=48,terminal_sha256=sha(out/'terminal.npz'),
    elapsed_s=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    per_epoch_anchored_mse={str(e):float(np.mean([r['anchored_mse'] for r in allrows if r['epoch']==e])) for e in range(1,5)},
    final_theta_norm=float(theta.detach().norm()),baseline_mse=float(np.mean([float((native[r]-truth[r]).square().mean()) for r in protocol['fit_ranks']])),
    interpretation='Approximate Torch proposal training only; no native CHECK score is implied.')
with (out/'result.json').open('x') as fp:json.dump(result,fp,indent=2,allow_nan=False)
print(json.dumps(result),flush=True)
