"""Target-free independent runner for the coherent centered-state action."""
from pathlib import Path
import numpy as np
import coherent_central_scale_v1 as parent
from radial_binary_compact import select
from radial_features import features
from radial_gnn_numpy import gnn_step
from compact_scalar_final_v1 import final_scalar_features
from compact_final_head_v1 import final_input, final_delta

HERE = Path(__file__).resolve().parent


class Reference:
    def __init__(self, rank):
        folder = HERE/'rows_current_transport_shadow_v2'/f'row_{rank:04d}'
        self.W = np.load(folder/'weights.npy')
        self.P = dict(np.load(HERE/'payload_compact_feed_fullpair_v1.npz'))
        self.Q = dict(np.load(HERE/'payload_fullpair_final_head_v1.npz'))

    def run(self, theta, fixed=None):
        selections, scales = [], []
        def selecting(score, floor):
            ix, mask = select(score, floor) if fixed is None else fixed[len(selections)]
            selections.append((ix.copy(), mask.copy()))
            return ix, mask
        env = dict(parent.env, select=selecting)
        exec(parent.source, env)
        state = np.zeros((self.W.shape[1], 64), np.float32)
        def hook(l, m, C, K21, aux, w):
            nonlocal state
            d, state = gnn_step(l, features(l,m,C,K21,aux,w), aux['sigma'], w, state, self.P)
            return d
        def final_hook(l, m, aux, w):
            h = final_input(final_scalar_features(l,m,aux,w), w, state, self.P)
            return final_delta(h, aux['sigma'], self.Q)
        def scale_hook(l, aux):
            s = parent.scaling(l, aux, theta)
            scales.append(s.copy())
            return s
        means, _ = env['predict'](self.W, hook=hook, final_hook=final_hook,
                                  scale_hook=scale_hook, collect_states=False)
        assert len(selections) == 30 and len(scales) == 15
        return np.maximum(means, 0), selections, np.stack(scales)


def selection_changes(base, plus, minus):
    changes = []
    for j, ((ix, mask), (pi, pm), (mi, mm)) in enumerate(zip(base, plus, minus)):
        pc, mc = int(np.count_nonzero(pm != mask)), int(np.count_nonzero(mm != mask))
        po, mo = not np.array_equal(ix, pi), not np.array_equal(ix, mi)
        if pc or mc or po or mo:
            changes.append(dict(layer=j//2, kind=('active','birth')[j%2],
                                plus_memberships_changed=pc, minus_memberships_changed=mc,
                                plus_order_changed=po, minus_order_changed=mo))
    return changes
