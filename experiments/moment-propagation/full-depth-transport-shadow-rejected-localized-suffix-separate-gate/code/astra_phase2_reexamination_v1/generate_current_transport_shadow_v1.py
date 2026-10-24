"""Target-free baseline tape and real-carrier response to fixed probes."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from radial_compact_feed_fullpair_v1 import predict
from radial_binary_compact import select
from radial_features import features
from radial_gnn_numpy import gnn_step

HERE = Path(__file__).resolve().parent
STORE = HERE.parent / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rank', type=int, required=True)
    ap.add_argument('--responses', action='store_true')
    args = ap.parse_args()
    out = HERE / 'rows_current_transport_shadow_v1' / f'row_{args.rank:04d}'
    assert not out.exists()
    man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
    W = np.load(STORE / Path(man[args.rank]['path']).parent / 'weights.npy')
    P = dict(np.load(HERE / 'payload_compact_feed_fullpair_v1.npz'))
    saved = {k: [] for k in ('pre_cov', 'k21', 'k4', 'mu', 'host', 'sigma', 'x', 'state_prev', 'delta', 'alive')}

    def run(probe_layer=None, direction=None, eps=0., capture=False):
        state = np.zeros((1024, 64), np.float32)
        def hook(l, m, C, K21, aux, w):
            nonlocal state
            x = features(l, m, C, K21, aux, w)
            previous = state.copy()
            delta, state = gnn_step(l, x, aux['sigma'], w, state, P)
            if capture:
                _, alive = select(aux['gate'], .01)
                for key, value in dict(pre_cov=aux['pre_cov'], k21=K21, k4=aux['pre_k4'],
                                      mu=aux['alpha']*aux['sigma'], host=m, sigma=aux['sigma'],
                                      x=x, state_prev=previous, delta=delta, alive=alive).items():
                    saved[key].append(np.array(value, dtype=np.float32, copy=True))
            if l == probe_layer:
                delta = delta.astype(np.float64) + eps*direction
            return delta
        return predict(W, hook=hook, gate_floor=.01, cov32=True, metric='covariance', collect_states=False)[0]

    started = time.time()
    with threadpool_limits(limits=4):
        baseline = run(capture=True)
        responses = []
        arrays = {}
        if args.responses:
            for layer in (4, 9, 13):
                rng = np.random.default_rng(131000 + args.rank*31 + layer)
                direction = rng.standard_normal(1024)
                direction /= np.sqrt(np.mean(direction**2))
                arrays[f'direction_{layer}'] = direction
                deriv = []
                for eps in (2e-4, 1e-4):
                    plus = run(layer, direction, eps)
                    minus = run(layer, direction, -eps)
                    deriv.append((plus-minus)/(2*eps))
                arrays[f'response_{layer}'] = deriv[0]
                arrays[f'response_half_{layer}'] = deriv[1]
                row = dict(layer=layer, epsilon=2e-4,
                           final_response_rms=float(np.sqrt(np.mean(deriv[0][-1]**2))),
                           fd_half_relative=float(np.linalg.norm(deriv[0][-1]-deriv[1][-1])/np.linalg.norm(deriv[1][-1])))
                responses.append(row)
                print(json.dumps(dict(rank=args.rank, **row)), flush=True)
    old = HERE / f'compact_feed_fullpair_closedloop_r{args.rank}_v1.npz'
    parity = None
    if old.exists():
        with np.load(old) as z:
            key = 'mean'
            parity = float(np.max(np.abs(baseline-z[key])))
        assert parity < 1e-12, parity
    out.mkdir(parents=True)
    for k, v in saved.items():
        np.save(out / f'{k}.npy', np.stack(v))
    np.save(out / 'weights.npy', W.astype(np.float32))
    np.save(out / 'baseline.npy', baseline)
    if args.responses:
        np.savez(out / 'probes.npz', **arrays)
    receipt = dict(rank=args.rank, target_access=False, responses=responses,
                   existing_parent_max_abs=parity, wall_s=time.time()-started)
    (out / 'result.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
