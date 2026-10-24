"""Fixed five-epoch 910-parameter CPU full-carrier bank readout refinement."""
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch
from full_carrier_bank_autodiff_v1 import BankFullCard

HERE = Path(__file__).resolve().parent
OUT = HERE/'full_carrier_readout_pilot_v1.json'
PAYLOAD = HERE/'payload_full_carrier_readout_pilot_v1.npz'
PROGRESS = HERE/'full_carrier_readout_pilot_progress_v1'
assert not OUT.exists() and not PAYLOAD.exists() and not PROGRESS.exists()
assert json.loads((HERE/'full_carrier_bank_baseline_v1.json').read_text())['passed']
for rank in (35,100):
    assert json.loads((HERE/f'full_carrier_fixed_cell_r{rank}_v1.json').read_text())['branch_fidelity_pass']
PROGRESS.mkdir()
VALID = (37,39,42,48)
TRAIN = tuple(r for r in range(35,59) if r not in VALID)
SEED = 951212
torch.manual_seed(SEED)
torch.set_num_threads(3)
Pnp = dict(np.load(HERE/'payload_compact_feed_fullpair_v1.npz'))
original = torch.tensor(np.concatenate((Pnp['head'][1:15],Pnp['head_b'][1:15,None]),axis=1))
update = torch.nn.Parameter(torch.zeros_like(original))
radius = .20*torch.linalg.vector_norm(original,dim=1).clamp_min(1e-3)
labels = {r:torch.tensor(np.load(HERE.parent/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{r:04d}.npz')['mc_post'],dtype=torch.float64) for r in TRAIN+VALID}
opt = torch.optim.Adam([update],lr=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=5*len(TRAIN),eta_min=1e-4)
start = time.time()


def payload_numpy():
    p = {k:v.copy() for k,v in Pnp.items()}
    fitted = (original+update.detach()).numpy()
    p['head'][1:15] = fitted[:,:64]
    p['head_b'][1:15] = fitted[:,64]
    return p


def run(rank,training):
    card = BankFullCard(rank,'cpu')
    p = dict(card.P0)
    fitted = original+update
    p['head'] = torch.cat((p['head'][:1],fitted[:,:64],p['head'][15:]),dim=0)
    p['head_b'] = torch.cat((p['head_b'][:1],fitted[:,64],p['head_b'][15:]),dim=0)
    pred,_ = card.run(P=p,training=training)
    pred = pred.clamp_min(0)
    mse = ((pred-labels[rank])**2).mean(1)
    return mse[-1]/2e-8,mse,card


def evaluate():
    rows = []
    with torch.no_grad():
        for rank in VALID:
            loss,mse,card = run(rank,False)
            rows.append(dict(rank=rank,final_mse=float(mse[-1]),layer_mse=mse.tolist()))
            del loss,mse,card
    return rows


initial = evaluate()
baseline = {r['rank']:r['final_mse'] for r in initial}
config = dict(train=TRAIN,valid=VALID,epochs=5,parameters=update.numel(),seed=SEED,
              device='cpu',scalar_dtype='float64',normalizer=2e-8,readout_relative_radius=.20,
              initial=initial,wall_s=time.time()-start)
(PROGRESS/'initial.json').write_text(json.dumps(config,indent=2)+'\n')
print(json.dumps(config),flush=True)
history = []
for epoch in range(5):
    losses = []
    for step,rank in enumerate(np.random.default_rng(SEED+epoch).permutation(TRAIN),start=1):
        assert time.time()-start < 1800,'30-minute full-carrier pilot deadline'
        loss,mse,card = run(int(rank),True)
        assert torch.isfinite(loss).item(),(epoch,int(rank),'loss')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gradnorm = torch.nn.utils.clip_grad_norm_([update],5.)
        assert torch.isfinite(gradnorm).item(),(epoch,int(rank),'gradient')
        opt.step()
        with torch.no_grad():
            norms = torch.linalg.vector_norm(update,dim=1).clamp_min(1e-30)
            update.mul_((radius/norms).clamp(max=1)[:,None])
        sched.step()
        losses.append(float(loss.detach()))
        del loss,mse,card
        if step % 5 == 0:
            rec = dict(epoch=epoch+1,step=step,mean_train_normalized_loss=float(np.mean(losses)),
                       grad_norm=float(gradnorm),wall_s=time.time()-start,
                       max_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
            (PROGRESS/f'epoch_{epoch+1:02d}_step_{step:02d}.json').write_text(json.dumps(rec,indent=2)+'\n')
            print(json.dumps(rec),flush=True)
    valid = evaluate()
    for row in valid:
        row['parent_mse'] = baseline[row['rank']]
        row['ratio'] = row['final_mse']/row['parent_mse']
    rec = dict(epoch=epoch+1,train_normalized_loss=float(np.mean(losses)),valid=valid,
               update_relative_by_layer=(torch.linalg.vector_norm(update.detach(),dim=1)/(radius/.20)).tolist(),
               wall_s=time.time()-start)
    history.append(rec)
    np.savez(PROGRESS/f'epoch_{epoch+1:02d}_payload.npz',**payload_numpy())
    (PROGRESS/f'epoch_{epoch+1:02d}.json').write_text(json.dumps(rec,indent=2)+'\n')
    print(json.dumps(rec),flush=True)
np.savez(PAYLOAD,**payload_numpy())
receipt = dict(**{k:v for k,v in config.items() if k != 'wall_s'},history=history,
               wall_s=time.time()-start,max_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
               qualification='Fixed-epoch CPU full-carrier training with dynamic selections; holdouts are excluded from this pilot only. Independent NumPy actual-card validation and complete release gates remain required. Not a candidate score or submission.')
OUT.write_text(json.dumps(receipt,indent=2)+'\n')
print('FULL_CARRIER_READOUT_PILOT_TRAINED',flush=True)
