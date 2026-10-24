"""Fixed bank-P0 response, inherited CP legs only, exact declared repeats."""
from pathlib import Path
import numpy as np
import compact_fullpair_final_reference_v1 as final_parent
import radial_compact_feed_fullpair_v1 as pair_parent
from radial_features import features
from radial_gnn_numpy import gnn_step
from mean_response_feature_diagonal_v1 import feature_diagonal
from gnn_mean_response_dual_v1 import step

HERE = Path(__file__).resolve().parent


def compose(source):
    edits = (
        ('        host = m.copy()',
         '        response_observe(l, m, C, K21, aux, W[l], (mu,Cpre,K21,k4,k22), (m,C,target,radial_new,aux))\n        host = m.copy()'),
        ('        cp_post = (inherited+born)*live',
         '        q = response_gate(l, g)\n        response_inherited = q[:, None]**2*q[None, :]*cp_pre\n        cp_post = (response_inherited+born)*live'),
        ('g.astype(np.float32)[:, None]*pp,', 'q.astype(np.float32)[:, None]*pp,'),
        ('        previous_alive = alive', '        response_audit(l, locals())\n        previous_alive = alive'),
    )
    for before, after in edits:
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    return source


class ResponseCard:
    def __init__(self, enabled=True, final=True, audit=False):
        self.enabled, self.audit = enabled, audit
        self.P0 = dict(np.load(HERE/'payload_compact_feed_fullpair_v1.npz'))
        self.state = None
        self.records = []
        self.value_state_parity = []
        self.gates = {}
        self.response = None
        parent = final_parent if final else pair_parent
        source = compose(parent.source if final else parent.PREDICT_SOURCE)
        env = dict(parent.env, response_observe=self.observe,
                   response_gate=self.gate, response_audit=self.check)
        exec(source, env)
        self.predict = env['predict']

    def observe(self, l, m, C, K21, aux, W, inputs, output):
        if self.state is None:
            self.state = np.zeros((len(m), self.P0['head'].shape[1]), np.float32)
        if not self.enabled:
            return
        x = features(l, m, C, K21, aux, W)
        old = self.state
        if 8 <= l <= 14:
            dx, dm, _ = feature_diagonal(*inputs, output)
            delta, self.state, ddelta = step(l, x, dx, aux['sigma'], W, old, self.P0)
            self.response = np.clip(dm+ddelta.astype(np.float64), 0, 1)
            self.gates[l] = self.response.copy()
            if self.audit:
                od, os = gnn_step(l, x, aux['sigma'], W, old, self.P0)
                parity = np.array_equal(delta, od) and np.array_equal(self.state, os)
                self.value_state_parity.append(bool(parity))
                assert parity, l
        else:
            _, self.state = gnn_step(l, x, aux['sigma'], W, old, self.P0)

    def gate(self, l, gaussian):
        if self.enabled and 8 <= l <= 14:
            assert self.response is not None and l in self.gates
            return self.response
        return gaussian

    def check(self, l, t):
        if not self.audit:
            return
        star = t['star'].astype(np.float64)
        star_plane = star.T/3
        np.fill_diagonal(star_plane, np.diag(star))
        rebuilt = t['cp_post']+t['trace_post']+star_plane
        exact_star_plane = t['rest']
        intended = t['cp_post']+t['trace_post']+exact_star_plane
        scale = max(np.linalg.norm(t['target']), 1e-30)
        self.records.append(dict(layer=l,
            declared_repeat_relative=float(np.linalg.norm(intended-t['target'])/scale),
            rounded_star_repeat_relative=float(np.linalg.norm(rebuilt-t['target'])/scale),
            response_rms=float(np.sqrt(np.mean((t['q']-t['g'])**2))),
            birth_count=len(t['selected']), active_count=int(t['alive'].sum()),
            columns=int(t['p'].shape[1])))
