"""One preregistered 160-net full-carrier joint P/Q fit; no validation labels."""
from datetime import datetime,timezone
import gc
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import traceback
import numpy as np
import torch
from joint_prefix_final_data_v1 import (
    HERE,P2,STORE,TRAIN,VALID,PKEYS,QKEYS,ManifestFullCard,original_payloads,sha,weight_path,
)

OUT = HERE/'joint_prefix_final_fit_v1'
EPOCHS,BATCH = 6,4
UPDATES = EPOCHS*len(TRAIN)//BATCH


def utc():
    return datetime.now(timezone.utc).isoformat()


def main():
    assert 'threadpoolctl' not in sys.modules
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0')=='0'
    assert torch.backends.mps.is_available()
    assert not OUT.exists(), 'Preserve the entire existing run; no overwrite or automatic restart.'
    prerequisites = ('joint_prefix_final_gradients_v1.json',
                     'joint_prefix_final_finite_numpy_v1.json',
                     'joint_prefix_final_finite_mps_v1.json','joint_prefix_final_loader_v1.json')
    for filename in prerequisites:
        assert json.loads((HERE/filename).read_text())['passed'],filename
    loader_receipt = json.loads((HERE/prerequisites[-1]).read_text())
    assert loader_receipt['loader_sha256']==sha(HERE/'joint_prefix_final_data_v1.py')
    assert loader_receipt['protocol_sha256']==sha(HERE/'JOINT_PREFIX_FINAL_FIT_PROTOCOL_V1.md')
    assert json.loads((HERE/prerequisites[0]).read_text())['card_source_sha256']==sha(HERE/'centered_bank_autodiff_v3.py')
    torch.set_num_threads(3)
    torch.manual_seed(930696)
    rng = np.random.Generator(np.random.PCG64DXSM(930696))
    originals = dict(zip(('P','Q'),original_payloads()))
    bases = {ns:{k:torch.tensor(v,device='mps',dtype=torch.float32) for k,v in values.items()}
             for ns,values in originals.items()}
    scales,z = {},torch.nn.ParameterDict()
    for ns,keys in (('P',PKEYS),('Q',QKEYS)):
        for key in keys:
            name = ns+'_'+key
            scales[name] = max(float(np.sqrt(np.mean(originals[ns][key].astype(float)**2))),.001)
            z[name] = torch.nn.Parameter(torch.zeros_like(bases[ns][key]))
    assert sum(v.numel() for v in z.values())==78097
    # Path checks and lineage precede all target reads. Only training labels
    # are then loaded; the 40 validation labels are never accessed here.
    sources = []
    for rank in TRAIN:
        target = P2/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{rank:04d}.npz'
        assert target.is_file()
        sources.append(dict(rank=rank,weight_file=str(weight_path(rank)),target_file=str(target),
                            target_sha256=sha(target)))
    OUT.mkdir()
    started = time.time()
    meta = dict(protocol_sha256=sha(HERE/'JOINT_PREFIX_FINAL_FIT_PROTOCOL_V1.md'),
                script_sha256=sha(__file__),loader_sha256=sha(HERE/'joint_prefix_final_data_v1.py'),
                manifest_sha256=sha(STORE/'manifest.json'),
                prerequisites={f:sha(HERE/f) for f in prerequisites},
                original_P_sha256=sha(HERE/'payload_compact_feed_fullpair_v1.npz'),
                original_Q_sha256=sha(HERE/'payload_fullpair_final_head_v1.npz'),
                train_ranks=TRAIN,validation_ranks=VALID,epochs=EPOCHS,batch_nets=BATCH,
                updates=UPDATES,parameter_count=78097,group_scales=scales,
                seed=930696,utc_started=utc(),target_sources=sources,
                validation_targets_accessed=False,
                qualification='Fixed terminal checkpoint only; public-development split, not virgin generalization.')
    (OUT/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    progress = OUT/'progress.jsonl'

    def record(item):
        value = dict(utc=utc(),wall_seconds=time.time()-started,**item)
        with progress.open('a') as stream:
            stream.write(json.dumps(value)+'\n')
        print(json.dumps(value),flush=True)

    def payloads():
        result = {ns:dict(values) for ns,values in bases.items()}
        for ns,keys in (('P',PKEYS),('Q',QKEYS)):
            for key in keys:
                name = ns+'_'+key
                result[ns][key] = bases[ns][key]+scales[name]*z[name]
        return result['P'],result['Q']

    def snapshot(name):
        p,q = payloads()
        arrays = {ns+'_'+k:v.detach().cpu().numpy().copy()
                  for ns,values in (('P',p),('Q',q)) for k,v in values.items()}
        assert all(np.isfinite(v).all() for v in arrays.values())
        path = OUT/(name+'.npz')
        assert not path.exists()
        np.savez(path,**arrays)
        return dict(path=str(path),sha256=sha(path))

    try:
        targets = {}
        for source in sources:
            with np.load(source['target_file']) as data:
                target = data['mc_post'].copy()
            assert target.shape==(16,1024) and np.isfinite(target).all()
            targets[source['rank']] = torch.tensor(target,device='mps',dtype=torch.float32)
        optimizer = torch.optim.AdamW(z.parameters(),lr=.002,betas=(.9,.999),eps=1e-8,
                                     weight_decay=0.,foreach=False)
        step,projections,grad_clips = 0,0,0
        history,checkpoints = [],[]
        largest_gpu = int(torch.mps.driver_allocated_memory())
        record(dict(stage='training_start',training_targets=len(targets),validation_targets=0,
                    fixed_updates=UPDATES,parameter_count=78097))
        for epoch in range(EPOCHS):
            order = rng.permutation(TRAIN).tolist()
            epoch_losses,epoch_final,epoch_aux = [],[],[]
            for offset in range(0,len(order),BATCH):
                batch = order[offset:offset+BATCH]
                rate = .0002+.5*(.002-.0002)*(1+math.cos(math.pi*step/(UPDATES-1)))
                for group in optimizer.param_groups: group['lr']=rate
                optimizer.zero_grad(set_to_none=True)
                batch_metrics = []
                for rank in batch:
                    p,q = payloads()
                    card = ManifestFullCard(rank,p,q)
                    prediction,_ = card.run(P=p,training=True)
                    prediction = torch.where(prediction<0,torch.zeros_like(prediction),prediction)
                    error = prediction-targets[rank]
                    final = error[-1].square().mean()
                    aux = error[:-1].square().mean()
                    objective = (final+.05*aux)/2e-8
                    values = (float(objective.detach().cpu()),float(final.detach().cpu()),float(aux.detach().cpu()))
                    assert all(math.isfinite(v) for v in values), (rank,values)
                    (objective/len(batch)).backward()
                    batch_metrics.append(dict(rank=rank,objective=values[0],final_mse=values[1],aux_mse=values[2]))
                    epoch_losses.append(values[0]);epoch_final.append(values[1]);epoch_aux.append(values[2])
                    del card,p,q,prediction,error,final,aux,objective
                    gc.collect()
                norm = float(torch.nn.utils.clip_grad_norm_(z.parameters(),5.,error_if_nonfinite=True,foreach=False).cpu())
                grad_clips += int(norm>5.)
                optimizer.step()
                displacements = {}
                with torch.no_grad():
                    for name,value in z.items():
                        rms = float(value.square().mean().sqrt().cpu())
                        assert math.isfinite(rms), name
                        if rms>.10:
                            value.mul_(.10/rms);projections+=1;rms=.10
                        displacements[name] = rms
                step+=1
                torch.mps.synchronize()
                largest_gpu = max(largest_gpu,int(torch.mps.driver_allocated_memory()))
                record(dict(stage='batch',epoch=epoch+1,update=step,learning_rate=rate,
                            metrics=batch_metrics,gradient_norm=norm,gradient_clips=grad_clips,
                            group_projections=projections,group_relative_rms=displacements,
                            gpu_driver_bytes=int(torch.mps.driver_allocated_memory()),
                            process_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)))
                torch.mps.empty_cache()
            summary = dict(epoch=epoch+1,online_objective=float(np.mean(epoch_losses)),
                           online_final_mse=float(np.mean(epoch_final)),online_aux_mse=float(np.mean(epoch_aux)))
            history.append(summary)
            checkpoints.append(snapshot(f'epoch_{epoch+1:02d}_audit_only'))
            record(dict(stage='epoch_complete',**summary))
        assert step==UPDATES
        terminal = snapshot('terminal_payload')
        # The endpoint is already frozen before seeing ANY validation output.
        record(dict(stage='terminal_payload_written',**terminal))
        del targets,optimizer
        gc.collect();torch.mps.empty_cache()
        predictions = {}
        with torch.no_grad():
            p,q = payloads()
            for rank in VALID:
                card = ManifestFullCard(rank,p,q)
                pred,_ = card.run(P=p)
                pred = torch.where(pred<0,torch.zeros_like(pred),pred)
                predictions[f'rank_{rank:04d}'] = pred.cpu().double().numpy().copy()
                assert np.isfinite(predictions[f'rank_{rank:04d}']).all(),rank
                del card,pred
                gc.collect();torch.mps.empty_cache()
                record(dict(stage='validation_prediction_only',rank=rank,validation_targets=0))
        pred_path = OUT/'terminal_validation_mps.npz'
        np.savez(pred_path,**predictions)
        result = dict(completed=True,terminal=terminal,validation_predictions_sha256=sha(pred_path),
                      epochs=EPOCHS,updates=step,history=history,audit_checkpoints=checkpoints,
                      gradient_clips=grad_clips,group_projections=projections,
                      final_group_relative_rms=displacements,validation_targets_accessed=False,
                      accuracy_verdict=None,utc_finished=utc(),wall_seconds=time.time()-started,
                      largest_observed_gpu_driver_bytes=largest_gpu,
                      peak_process_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                      qualification='Training completed is not an accuracy pass. Only terminal_payload is eligible for independent NumPy judging; snapshots are not selection candidates.')
        (OUT/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
        record(dict(stage='completed',**result))
    except BaseException as exc:
        failure = dict(completed=False,utc_failed=utc(),wall_seconds=time.time()-started,
                       exception_type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),
                       validation_targets_accessed=False,qualification='Preserved failed fixed run; no automatic retry or hyperparameter substitution.')
        (OUT/'failure.json').write_text(json.dumps(failure,indent=2)+'\n')
        print(json.dumps(failure),flush=True)
        raise


if __name__=='__main__':
    main()
