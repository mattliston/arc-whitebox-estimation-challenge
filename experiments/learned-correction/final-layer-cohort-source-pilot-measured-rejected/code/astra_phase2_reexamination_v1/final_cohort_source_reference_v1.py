"""Current frozen dynamic estimator, final source reads and no labels."""
from pathlib import Path
import numpy as np
import compact_fullpair_final_reference_v1 as parent
from radial_features import features
from radial_gnn_numpy import gnn_step
from compact_scalar_final_v1 import final_scalar_features
from compact_final_head_v1 import final_input, final_delta
from final_cohort_sources_v1 import split

HERE = Path(__file__).resolve().parent


def run(rank, collect=False):
    W = np.load(HERE/'rows_current_transport_shadow_v2'/f'row_{rank:04d}/weights.npy')
    P = dict(np.load(HERE/'payload_compact_feed_fullpair_v1.npz'))
    Q = dict(np.load(HERE/'payload_fullpair_final_head_v1.npz'))
    state = np.zeros((1024, 64), np.float32)
    saved, counts = {}, []

    def hook(l, m, C, K21, aux, w):
        nonlocal state
        dm, state = gnn_step(l, features(l, m, C, K21, aux, w), aux['sigma'], w, state, P)
        return dm

    def birth_probe(l, t):
        assert l == len(counts)
        counts.append(len(t['selected']))

    def final_probe(t):
        assert len(counts) == 15
        parts = split(t['pp'], t['lam'], counts)
        parts = np.column_stack((parts, t['star_scalar'], t['trace_scalar']))
        saved['source_sum_relative'] = float(np.linalg.norm(parts.sum(axis=1)-t['k3'])/np.linalg.norm(t['k3']))
        saved['sources'] = parts/t['aux']['sigma'][:, None]**3
        saved['birth_counts'] = np.asarray(counts)

    def final_hook(l, m, aux, w):
        h = final_input(final_scalar_features(l, m, aux, w), w, state, P)
        delta = final_delta(h, aux['sigma'], Q)
        saved.update(h=h.copy(), host=m.copy(), sigma=aux['sigma'].copy(), delta=delta.copy(),
                     variance=aux['pre_v'].copy(), k3=aux['pre_k3'].copy())
        return delta

    source = parent.source
    if collect:
        edits = (
            ('        previous_alive = alive', '        birth_probe(l,locals())\n        previous_alive = alive'),
            ('            k3 = k3+star_push(W[l],star,diagonal_only=True)+3*bv*bpre',
             '            star_scalar = star_push(W[l],star,diagonal_only=True)\n'
             '            trace_scalar = 3*bv*bpre\n'
             '            k3 = k3+star_scalar+trace_scalar'),
            ('            m,aux = nonlinear_scalar(mu,v,k3,radial*v*v)',
             '            m,aux = nonlinear_scalar(mu,v,k3,radial*v*v)\n            final_probe(locals())'),
        )
        for before, after in edits:
            assert source.count(before) == 1, before
            source = source.replace(before, after)
    env = dict(parent.env, birth_probe=birth_probe, final_probe=final_probe)
    exec(source, env)
    prediction, _ = env['predict'](W, hook=hook, final_hook=final_hook,
                                  gate_floor=.01, cov32=True, metric='covariance')
    saved['prediction'] = np.maximum(prediction, 0)
    return saved
