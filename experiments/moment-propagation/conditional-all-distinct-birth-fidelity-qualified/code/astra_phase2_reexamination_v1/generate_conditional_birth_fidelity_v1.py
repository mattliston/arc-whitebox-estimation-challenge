"""Target-free NumPy zero-action parity and fixed coefficient secants."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from conditional_birth_bank_reference_v1 import predict
from conditional_distinct_birth_v1 import birth_scale
from radial_features import features
from radial_gnn_numpy import gnn_step
from compact_scalar_final_v1 import final_scalar_features
from compact_final_head_v1 import final_input,final_delta

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rank',type=int,choices=(35,100),required=True)
    args = ap.parse_args()
    out = HERE/f'conditional_birth_numpy_fidelity_r{args.rank}_v1.json'
    arrays = out.with_suffix('.npz')
    assert not out.exists() and not arrays.exists()
    folder = HERE/'rows_current_transport_shadow_v2'/f'row_{args.rank:04d}'
    W = np.load(folder/'weights.npy')
    original = np.load(HERE/f'fullpair_scalar_final_closedloop_r{args.rank}_v1.npz')['mean']
    P = dict(np.load(HERE/'payload_compact_feed_fullpair_v1.npz'))
    Q = dict(np.load(HERE/'payload_fullpair_final_head_v1.npz'))
    direction = np.random.default_rng(955311).normal(size=(3,64)).astype(np.float32)
    direction /= np.linalg.norm(direction)

    def run(eps):
        state = np.zeros((1024,64),np.float32)
        theta = np.float32(eps)*direction
        amplitudes = []
        def hook(l,m,C,K21,aux,w):
            nonlocal state
            d,state = gnn_step(l,features(l,m,C,K21,aux,w),aux['sigma'],w,state,P)
            return d
        def final_hook(l,m,aux,w):
            h = final_input(final_scalar_features(l,m,aux,w),w,state,P)
            return final_delta(h,aux['sigma'],Q)
        def birth_hook(l,birth,births):
            eta = birth_scale(l,state,birth,births,theta)
            amplitudes.append(dict(layer=l,minimum=float(eta.min()),maximum=float(eta.max())))
            return eta
        means,_ = predict(W,hook=hook,final_hook=final_hook,birth_hook=birth_hook,collect_states=False)
        return np.maximum(means,0),amplitudes

    t0 = time.time()
    records,saves = [],{}
    with threadpool_limits(limits=3):
        baseline,amps = run(0.)
        parity = float(np.max(np.abs(baseline-original)))
        assert parity == 0,parity
        assert all(a['minimum'] == 1 and a['maximum'] == 1 for a in amps)
        saves['baseline'] = baseline
        for name,eps in (('full',.05),('half',.025)):
            plus,pa = run(eps)
            minus,ma = run(-eps)
            response = (plus-minus)/(2*eps)
            prefix = max(float(np.max(np.abs(plus[:2]-baseline[:2]))),float(np.max(np.abs(minus[:2]-baseline[:2]))))
            assert prefix == 0,prefix
            saves[f'response_{name}'] = response
            saves[f'plus_{name}'],saves[f'minus_{name}'] = plus,minus
            row = dict(epsilon=eps,prefix_through_l1_max_abs=prefix,
                       final_response_rms=float(np.sqrt(np.mean(response[-1]**2))),
                       plus_amplitudes=pa,minus_amplitudes=ma)
            records.append(row)
            print(json.dumps(dict(rank=args.rank,epsilon=eps,final_response_rms=row['final_response_rms'])),flush=True)
    a,b = saves['response_full'][-1],saves['response_half'][-1]
    half_relative = float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))
    saves['direction'] = direction
    receipt = dict(rank=args.rank,zero_action_max_abs=parity,records=records,
                   finite_difference_half_relative=half_relative,wall_s=time.time()-t0,
                   qualification='Target-free NumPy bank-path zero-head parity and coefficient-direction secants. No ground truths, fit, raw score, bill or candidate claim.')
    np.savez(arrays,**saves)
    out.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(rank=args.rank,zero_action_max_abs=parity,finite_difference_half_relative=half_relative,wall_s=receipt['wall_s'])),flush=True)


if __name__ == '__main__':
    main()
