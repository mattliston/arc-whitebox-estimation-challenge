"""Actual finite-step CPU replay of the one frozen train-only payload."""
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch
from full_carrier_bank_autodiff_v1 import BankFullCard

HERE=Path(__file__).resolve().parent
OUT=HERE/'encoder_end_to_end_actual_torch_v1.json'
ART=OUT.with_suffix('.npz')
assert not OUT.exists() and not ART.exists()
probe=json.loads((HERE/'encoder_end_to_end_direction_v1.json').read_text())
pp=HERE/'payload_encoder_end_to_end_direction_v1.npz'
assert hashlib.sha256(pp.read_bytes()).hexdigest()==probe['payload_sha256']
Pnp=dict(np.load(pp));torch.set_num_threads(3)
start=time.time();rows=[];saves={}
with torch.no_grad():
    for rank in probe['valid']:
        card=BankFullCard(rank,'cpu')
        P={k:torch.tensor(v,dtype=torch.float32) for k,v in Pnp.items()}
        pred,trace=card.run(P=P,capture=True)
        pred=pred.clamp_min(0).numpy()
        label=np.load(HERE.parent/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{rank:04d}.npz')['mc_post']
        mse=np.mean((pred-label)**2,axis=1)
        row=dict(rank=rank,final_mse=float(mse[-1]),layer_mse=mse.tolist(),trace=trace,
                 wall_s=time.time()-start)
        rows.append(row);saves[f'mean_{rank}']=pred
        print(json.dumps({k:v for k,v in row.items() if k not in ('trace','layer_mse')}),flush=True)
        del card,P,pred
np.savez(ART,**saves)
record=dict(records=rows,wall_s=time.time()-start,
            peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            payload_sha256=probe['payload_sha256'],
            qualification='Dynamic full-carrier CPU finite-step replay, not a tangent score. Independent NumPy parity and pilot judge still required.')
OUT.write_text(json.dumps(record,indent=2)+'\n')
