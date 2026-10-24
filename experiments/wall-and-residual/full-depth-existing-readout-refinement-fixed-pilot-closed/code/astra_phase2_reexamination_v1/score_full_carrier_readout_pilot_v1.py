"""Independent dynamic NumPy bank validation of the fixed fifth-epoch pilot."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from compact_fullpair_final_reference_v1 import predict
from radial_features import features
from radial_gnn_numpy import gnn_step
from compact_scalar_final_v1 import final_scalar_features
from compact_final_head_v1 import final_input,final_delta

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rank',type=int,choices=(37,39,42,48),required=True)
    args = ap.parse_args()
    out = HERE/f'full_carrier_readout_actual_r{args.rank}_v1.json'
    arrays = out.with_suffix('.npz')
    assert not out.exists() and not arrays.exists()
    training = json.loads((HERE/'full_carrier_readout_pilot_v1.json').read_text())
    assert training['epochs'] == 5 and len(training['history']) == 5
    pp = HERE/'payload_full_carrier_readout_pilot_v1.npz'
    P0 = dict(np.load(HERE/'payload_compact_feed_fullpair_v1.npz'))
    P1 = dict(np.load(pp))
    Q = dict(np.load(HERE/'payload_fullpair_final_head_v1.npz'))
    assert set(P0) == set(P1)
    for k in P0:
        if k not in ('head','head_b'):
            assert np.array_equal(P0[k],P1[k]),k
    for k in ('head','head_b'):
        assert np.array_equal(P0[k][[0,15]],P1[k][[0,15]]),k
    old = np.concatenate((P0['head'][1:15],P0['head_b'][1:15,None]),axis=1)
    new = np.concatenate((P1['head'][1:15],P1['head_b'][1:15,None]),axis=1)
    assert np.max(np.linalg.norm(new-old,axis=1)/np.linalg.norm(old,axis=1)) <= .200001
    folder = HERE/'rows_current_transport_shadow_v2'/f'row_{args.rank:04d}'
    W = np.load(folder/'weights.npy')
    predictions = []
    started = time.time()
    with threadpool_limits(limits=3):
        for P in (P0,P1):
            state = np.zeros((1024,64),np.float32)
            def hook(l,m,C,K21,aux,w):
                nonlocal state
                d,state = gnn_step(l,features(l,m,C,K21,aux,w),aux['sigma'],w,state,P)
                return d
            def final_hook(l,m,aux,w):
                h = final_input(final_scalar_features(l,m,aux,w),w,state,P)
                return final_delta(h,aux['sigma'],Q)
            means,_ = predict(W,hook=hook,final_hook=final_hook,collect_states=False)
            predictions.append(np.maximum(means,0))
    # Targets enter only after both actual estimator trajectories complete.
    truth = np.load(HERE.parent/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{args.rank:04d}.npz')['mc_post'].astype(np.float64)
    mses = [np.mean((m-truth)**2,axis=1) for m in predictions]
    tr = next(r for r in training['history'][-1]['valid'] if r['rank'] == args.rank)
    receipt = dict(rank=args.rank,parent_mse=float(mses[0][-1]),candidate_mse=float(mses[1][-1]),
                   ratio=float(mses[1][-1]/mses[0][-1]),
                   parent_layer_mse=mses[0].tolist(),candidate_layer_mse=mses[1].tolist(),
                   trained_torch_mse=tr['final_mse'],
                   torch_vs_numpy_mse_relative=abs(tr['final_mse']/float(mses[1][-1])-1),
                   payload_sha256=hashlib.sha256(pp.read_bytes()).hexdigest(),wall_s=time.time()-started,
                   qualification='Independent actual NumPy scalar-final bank reference on four new-fit holdouts, documented1e9 mean targets. Not the14-net panel, packaged candidate parity, metered bill, runtime gate or official score.')
    np.savez(arrays,parent=predictions[0],candidate=predictions[1])
    out.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__ == '__main__':
    main()
