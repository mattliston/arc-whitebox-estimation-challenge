"""One fixed60-parameter final-output fit; P and all validation labels frozen."""
import argparse
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if not args.run: parser.print_help(); return
    assert os.getpriority(os.PRIO_PROCESS,0)>=10
    assert 'threadpoolctl' not in sys.modules
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0')=='0'
    import numpy as np
    import torch
    import model
    from qualify_model import sha,write,resources
    torch.set_num_threads(3)
    assert torch.backends.mps.is_available()
    torch.manual_seed(9309081)
    ready=json.loads((HERE/'ready_v1.json').read_text())
    assert ready['passed'] and ready['model_sha256']==sha(HERE/'model.py')
    assert ready['protocol_sha256']==sha(HERE/'PROTOCOL.md')
    for path,digest in ready['source_bindings'].items(): assert sha(path)==digest,path
    out=HERE/'training_v1'
    assert not out.exists(), 'No overwrite or automatic resume'
    fixed_files=[HERE/'model.py',HERE/'PROTOCOL.md',Path(__file__),HERE/'ready_v1.json',
                 PARENT/'payload_response_rows_v1.npz']
    pins=dict(ready['source_bindings'])
    pins.update({str(p):sha(p) for p in fixed_files})
    target_manifest=PARENT/'joint_prefix_final_fit_v1/manifest.json'
    assert sha(target_manifest)=='d61757d28236e3d35d87415962d1a57d18ff38ae189651df52df5e3412b9a971'
    pins[str(target_manifest)]=sha(target_manifest)
    expected_targets={x['rank']:x['target_sha256'] for x in
                      json.loads(target_manifest.read_text())['target_sources']}
    assert set(expected_targets)==set(model.TRAIN)
    assert pins[str(PARENT/'payload_response_rows_v1.npz')]=='6b55c0d6071b2350e53e243bcd4137651e2c80a2cd55d04467bf194d0ea54a1c'
    out.mkdir()
    start=time.monotonic()
    result=dict(completed=False,seed=9309081,epochs=6,batch_nets=4,
                allocated_parameters=60,structurally_active_parameters=56,
                train_ranks=list(model.TRAIN),validation_ranks=list(model.VALID),
                sources=pins,validation_targets_accessed=False,
                started_utc=datetime.now(timezone.utc).isoformat())
    write(out/'started.json',result)
    def record(**item):
        value=dict(utc=datetime.now(timezone.utc).isoformat(),elapsed_s=time.monotonic()-start,**item)
        with (out/'progress.jsonl').open('a') as f: f.write(json.dumps(value,allow_nan=False)+'\n')
        print(json.dumps(value,allow_nan=False),flush=True)
    try:
        weight_pins={str(model.weight_path(rank)):sha(model.weight_path(rank))
                     for rank in tuple(model.TRAIN)+tuple(model.VALID)}
        write(out/'weight_bindings.json',weight_pins)
        targets={}
        target_pins={}
        for rank in model.TRAIN:
            path=PARENT.parent/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{rank:04d}.npz'
            target_pins[str(path)]=sha(path)
            assert target_pins[str(path)]==expected_targets[rank],rank
            with np.load(path) as src: values=src['mc_post'].copy()
            assert values.shape==(16,1024) and np.isfinite(values).all()
            targets[rank]=torch.tensor(values,device='mps',dtype=torch.float32)
        write(out/'training_targets.json',target_pins)
        theta=torch.nn.Parameter(torch.zeros((15,4),device='mps'))
        optimizer=torch.optim.Adam([theta],lr=.025,betas=(.9,.999),eps=1e-8,foreach=False)
        rng=np.random.Generator(np.random.PCG64DXSM(9309081))
        epochs,batch,updates=6,4,240
        step=0;history=[]
        record(stage='start',parameters=theta.numel(),training_nets=len(targets),validation_targets=0)
        for epoch in range(epochs):
            order=rng.permutation(model.TRAIN).tolist(); metrics=[]
            for offset in range(0,len(order),batch):
                ranks=order[offset:offset+batch]
                lr=.0025+.5*(.025-.0025)*(1+math.cos(math.pi*step/(updates-1)))
                optimizer.param_groups[0]['lr']=lr
                optimizer.zero_grad(set_to_none=True)
                batch_raw=[]
                for rank in ranks:
                    card=model.BirthDirectionCard(rank,theta=theta)
                    prediction,_=card.run(training=True)
                    prediction=torch.where(prediction<0,torch.zeros_like(prediction),prediction)
                    error=prediction-targets[rank]
                    final=error[-1].square().mean()
                    auxiliary=error[:-1].square().mean()
                    loss=(final+.05*auxiliary)/2e-8
                    raw=float(final.detach().cpu())
                    assert math.isfinite(raw) and bool(torch.isfinite(loss)),rank
                    (loss/len(ranks)).backward()
                    batch_raw.append(raw);metrics.append(raw)
                    resources(torch)
                    del card,prediction,error,final,auxiliary,loss
                    gc.collect()
                assert theta.grad is not None and bool(torch.isfinite(theta.grad).all())
                assert bool((theta.grad[0]==0).all()), 'Gaussian layer0 must be inactive'
                norm=float(torch.nn.utils.clip_grad_norm_([theta],5.,error_if_nonfinite=True,foreach=False).cpu())
                optimizer.step();step+=1
                assert bool(torch.isfinite(theta).all())
                resource_state=resources(torch)
                record(stage='batch',epoch=epoch+1,update=step,learning_rate=lr,ranks=ranks,
                       online_raw_mean=float(np.mean(batch_raw)),gradient_norm=norm,
                       maximum_absolute_coefficient=float(theta.detach().tanh().abs().max().cpu()),
                       **resource_state)
                torch.mps.empty_cache()
            values=theta.detach().cpu().numpy().copy()
            snapshot=out/f'epoch_{epoch+1:02d}_audit_only.json'
            write(snapshot,dict(theta=values.tolist(),epoch=epoch+1,checkpoint_selection=False))
            history.append(dict(epoch=epoch+1,online_raw_mean=float(np.mean(metrics))))
            record(stage='epoch_complete',**history[-1])
        assert step==updates
        values=theta.detach().cpu().numpy().copy()
        coefficients=model.deployment_coefficients(values)
        terminal=out/'terminal_theta.json'
        write(terminal,dict(theta=values.tolist(),coefficients=coefficients.tolist(),
            coefficient_rule='offline float32(math.tanh(float64(theta)))',terminal_epoch=6))
        terminal_sha=sha(terminal)
        # Terminal data fixed before any validation prediction or label.
        record(stage='terminal_sealed',theta_sha256=terminal_sha)
        del targets,optimizer
        gc.collect();torch.mps.empty_cache()
        coeff=torch.tensor(coefficients,device='mps')
        outputs={}
        with torch.no_grad():
            for rank in model.VALID:
                card=model.BirthDirectionCard(rank,coefficients=coeff)
                prediction,_=card.run()
                served=torch.where(prediction<0,torch.zeros_like(prediction),prediction).cpu().float().numpy().copy()
                assert np.isfinite(served).all()
                path=out/f'validation_{rank:04d}.npy'
                with path.open('xb') as f: np.save(f,served)
                outputs[str(rank)]=dict(path=str(path),sha256=sha(path))
                del card,prediction,served
                gc.collect();torch.mps.empty_cache()
                record(stage='validation_prediction_only',rank=rank,**resources(torch))
        assert sha(terminal)==terminal_sha
        assert all(sha(p)==v for p,v in pins.items())
        assert all(sha(p)==v for p,v in weight_pins.items())
        assert all(sha(p)==v for p,v in target_pins.items())
        result.update(completed=True,history=history,updates=step,terminal_sha256=terminal_sha,
                      validation_outputs=outputs,hashes_unchanged=True,**resources(torch))
    except BaseException:
        result['error']=traceback.format_exc()
    result.update(finished_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-start)
    write(out/'result.json',result)
    print(json.dumps(result,allow_nan=False),flush=True)
    return 0 if result['completed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
