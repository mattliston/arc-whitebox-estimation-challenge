"""Private Torch physics port of original 95c RTT2/RTP and RE22.

No SDK, model, or Torch import on module load. Numerical kernel ordering remains
Torch's, so this is a physics port, not a native bit-parity claim. Frozen law
metadata is detached across checkpoint replay, exactly as the existing M card
freezes its carrier addends; the base bank and refined projection remain in graph.
"""
import math


def basis(t, a, b):
    m = int(a.numel())
    A = t.stack((a, b), dim=1).to(t.float64)
    q, r = t.linalg.qr(A, mode='reduced')
    r0, r01, r1 = (abs(float(r[0, 0].detach())),
                   abs(float(r[0, 1].detach())), abs(float(r[1, 1].detach())))
    assert all(math.isfinite(x) for x in (r0, r01, r1))
    eps = 32 * 2.220446049250313e-16 * max(m, 2)
    if r0 and r1 > eps * (r01 + r1):
        return q.to(t.float32)
    cols = []
    for x in (a, b):
        x = x.to(t.float64)
        z = x
        original = float(t.linalg.vector_norm(x).detach())
        for _ in range(2):
            for v in cols:
                z = z - v * (v * z).sum()
        norm = t.linalg.vector_norm(z) if cols else t.linalg.vector_norm(x)
        if float(norm.detach()) > 2.220446049250313e-16 * max(m, 2) * original:
            cols.append(z / norm)
    return (t.stack(cols, dim=1).to(t.float32) if cols else
            t.zeros((m, 0), dtype=t.float32, device=a.device))


def refine(t, prefix, Q, old_plane):
    rank = int(Q.shape[1])
    if not rank:
        return Q, False
    # Algebraic signed-prefix power; old plane subtraction is the same
    # operation as native's two factored old-inner products.
    V = prefix.T @ Q
    if old_plane is not None:
        V = V - old_plane.T @ Q
    U = prefix @ V
    if old_plane is not None:
        U = U - old_plane @ V
    second = U[:, 1] if rank == 2 else t.zeros_like(U[:, 0])
    fresh = basis(t, U[:, 0], second)
    fallback = int(fresh.shape[1]) < rank
    return (Q if fallback else fresh), fallback


def re22(t, prefix, Q, rows, prefix_cols, tail_cols, variance):
    """Original95c residual-squared rule, including its original old content."""
    v = variance.to(t.float32)
    residual = prefix - Q @ (Q.T @ prefix)
    diagonal = rows[:, None] == prefix_cols[None, :]
    residual = t.where(diagonal, t.zeros_like(residual), residual)
    vp = v[prefix_cols]
    normalized = residual / (v[rows, None] * vp.sqrt()[None, :])
    energy = (normalized * normalized).sum(dim=1)
    ratio = (1 / v[tail_cols]).sum() / (1 / vp).sum()
    return energy * ratio


def install_after_m(cd, push_module):
    t = cd.t
    assert not hasattr(cd, '_o18_rtt_port')
    assert hasattr(cd, 'native_push_context') and hasattr(cd, 'ncar6_state')
    st = dict(layer=None, pending=None, law_records={}, reads=[])
    cd._o18_rtt_port = st
    begin0 = cd._cutsim_begin
    def begin(layer, mean, cov):
        assert st['pending'] is None
        st['layer'] = int(layer)
        cd._o18_rtt_base_old = None
        cd._o18_rtt_law = None
        return begin0(layer, mean, cov)
    cd._cutsim_begin = begin
    read0 = cd.shared['read']

    def read(tt, pp, lam, response, live, U, F, old_columns):
        plane = read0(tt, pp, lam, response, live, U, F, old_columns)
        l = int(st['layer'])
        assert l == int(cd.call_layer)
        if not 1 <= l <= 14 or not int(pp.shape[1]):
            return plane
        S = cd.ncar6_state
        mu, vc = S['mu'], t.diagonal(S['C']).clamp_min(1e-12)
        gate = 0.5 * t.erfc(-mu / vc.sqrt() / math.sqrt(2))
        count = lambda z: min(int(live.numel()),
            ((int((gate.detach() >= 0.5 * math.erfc(z / math.sqrt(2))).sum()) + 31) // 32) * 32)
        c2, c15 = count(2.0), count(1.5)
        assert 0 < c15 <= c2 <= int(live.numel())
        if c15 == c2:
            return plane
        assert st['pending'] is None
        law = cd._o18_rtt_law
        if law is not None:
            assert law['layer'] == l and law['forms'].shape == plane.shape
            # No graph is retained across steps/replay. The frozen carrier
            # already has this approximation in the M wrapper.
            st['law_records'][l] = {k: (v.detach() if t.is_tensor(v) else v)
                                    for k, v in law.items()}
        law = st['law_records'].get(l)
        assert law is not None, ('missing same-layer frozen law metadata', l)
        forms = law['forms'].to(plane.dtype)
        assert forms.shape == plane.shape
        old_current = old_native = None
        if old_columns:
            base = cd._o18_rtt_base_old
            assert base is not None and base[0] == l and base[1].shape == plane.shape
            old_current = base[1] + law['old_fold'].to(plane.dtype)
            context = cd.native_push_context
            assert context['layer'] == l and context['old_columns'] == old_columns
            left = pp[response, :old_columns]
            effective_lam = base[2]
            weighted = left * left * effective_lam[:old_columns][None, :]
            law_left = (left.to(t.float64) * law['old_w'][None, :] +
                        pp[response][:, law['old_partner']].to(t.float64) * law['old_x'][None, :])
            weighted = weighted + (law['factor'][:, None] * law_left).to(weighted.dtype)
            parts = []
            for b in context['blocks']:
                w = weighted[:, b['lo']:b['hi']]
                if context['truncated']:
                    x = push_module.uv_inner(t, w, t.ones_like(lam[b['lo']:b['hi']]),
                                             b['F'], b['pairs'], context['ru'])
                else:
                    x = w @ b['F'].T
                parts.append(x)
            inner = parts[0] if context['mapping'] is None else parts[0] @ context['mapping'].T + parts[1]
            old_native = inner @ U[live].T
        young = plane - forms
        if old_current is not None:
            young = young - old_current
        bank = young if old_native is None else young + old_native
        prefix = bank[:, :c15].to(t.float32)
        old_prefix = None if old_native is None else old_native[:, :c15].to(t.float32)
        Q0 = basis(t, vc[response], mu[response])
        Q, fallback = refine(t, prefix, Q0, old_prefix)
        tail = Q @ (Q.T @ young[:, c15:c2].to(t.float32))
        if old_native is not None:
            tail = tail + old_native[:, c15:c2].to(t.float32)
        block = t.cat((prefix, tail), dim=1).to(plane.dtype) + forms[:, :c2]
        output = t.cat((block, plane[:, c2:]), dim=1)
        energy = re22(t, prefix, Q, response, live[:c15], live[c15:c2], vc)
        st['pending'] = (l, response, energy)
        st['reads'].append(dict(layer=l, m=int(response.numel()), c15=c15, c2=c2,
                                old=int(old_columns), rank=int(Q.shape[1]), fallback=fallback))
        cd._o18_rtt_base_old = None
        cd._o18_rtt_law = None
        return output
    cd.shared['read'] = read

    features0 = cd.env['features']
    def features(layer, *args, **kwargs):
        out = features0(layer, *args, **kwargs)
        pending = st['pending']
        if pending is None:
            return out
        l, rows, energy = pending
        assert int(layer) == l
        prior = out[rows, 22]
        restored = (prior * prior + energy.to(prior.dtype)).clamp_min(0).sqrt()
        correction = t.zeros_like(out[:, 22]).index_copy(0, rows, restored - prior)
        basis22 = t.nn.functional.one_hot(t.tensor(22, device=out.device), num_classes=out.shape[1]).to(out.dtype)
        answer = out + correction[:, None] * basis22[None, :]
        st['pending'] = None
        return answer
    cd.env['features'] = features
    return cd
