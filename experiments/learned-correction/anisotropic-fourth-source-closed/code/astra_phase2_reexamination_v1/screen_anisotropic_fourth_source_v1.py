"""Predeclared current-host anisotropic K4/K3 source screen, no learned head."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from radial_anisotropic_fourth_source_v1 import predict

HERE=Path(__file__).resolve().parent
G22=HERE.parent/'public_g22_operator_features_v1'
STORE=HERE.parent/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
ap=argparse.ArgumentParser();ap.add_argument('--rank',type=int,choices=(35,100),required=True);args=ap.parse_args()
out=HERE/f'anisotropic_fourth_source_r{args.rank}_v1.json'
assert not out.exists() and not out.with_suffix('.npz').exists()
man=json.loads((STORE/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
W=np.load(STORE/Path(man[args.rank]['path']).parent/'weights.npy',mmap_mode='r')
truth=np.load(G22/'truthpost'/f'truthpost_rank{args.rank:04d}.npz')['mc_post'].astype(np.float64)
parent=np.load(HERE/f'compact_activeonly_r{args.rank}_v1.npz')['mean'][0]
start=time.time()
with threadpool_limits(limits=4):
    mean,states=predict(W,gate_floor=.01,collect_states=True)
assert np.all(np.isfinite(mean))
err=np.mean((np.maximum(mean,0)-truth)**2,axis=1)
base=np.mean((np.maximum(parent,0)-truth)**2,axis=1)
ratio=float(err[-1]/base[-1]);shift=max(r['column_shift_relative'] for r in states)
result=dict(rank=args.rank,layer_mse=err.tolist(),parent_layer_mse=base.tolist(),raw_ratio=ratio,
    max_column_shift_relative=shift,passes_predeclared_gate=ratio<=.85 and shift<.1,
    states=states,wall_s=time.time()-start,interpretation='Unlearned1e9-truth development screen, not a matched or metered candidate. Fourth-moment trace projection and quadratic birth remainder are approximations.')
out.write_text(json.dumps(result,indent=2)+'\n')
np.savez(out.with_suffix('.npz'),mean=mean)
print(json.dumps(result),flush=True)
