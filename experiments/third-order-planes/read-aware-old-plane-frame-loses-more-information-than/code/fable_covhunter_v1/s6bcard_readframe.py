"""TAG fable-covhunter (CH-51): READ-AWARE old-tier frame (lead's L3 test). The analyst's s6bcard with the every3 tier push re-sketched: at
each join the rank-R frame is QR(A Omega) with Omega = W[prev, S], S = the R current-layer units with the largest response score
min(gate, 1-gate) (the rows the covariance plane read sees), instead of W[prev, :R] (the first R units). gate = Phi(mu / sqrt(diag(W^T C W)))
comes from the layer's input state before the push (hooks state_in + mu; in native the covariance push can simply run first). The push is
s2card.make_push's source with only the Omega line replaced (exact-once). RANK sets covstep.TIER_RANK; MODE 'read' or 'first' (= the card's
own sketch; with RANK 384 it must be bit-exact with s6bcard). Drop-in card module (payload/model/card) for refit6b_aux.py --card."""
import inspect, math
import s6bcard as S6
import s2card
import covstep as cs
RANK, MODE = 320, 'read'
model, weights, truthpost, payload, PARAMS = S6.model, S6.weights, S6.truthpost, S6.payload, S6.PARAMS
_OLD = "Omega = w32[previous_indices, :rank]"
_NEW = "Omega = w32[previous_indices][:, _STASH['S'][layer][:rank]] if layer in _STASH['S'] else w32[previous_indices, :rank]"
_SRC = inspect.getsource(s2card.make_push)
assert _SRC.count(_OLD) == 1


def card(torch, w, p, a, c, e, fine_ckpt=False, **kw):
    cs.TIER_RANK = RANK
    cd = S6.card(torch, w, p, a, c, e, fine_ckpt=False, **kw)
    stash = {'S': {}}
    ns = {'_STASH': stash}; exec(_SRC.replace(_OLD, _NEW), ns)
    cd.shared['push'] = ns['make_push'](cd.shared['_qr'], s2card.EVERY3)

    def state_in(layer, st, ctx):
        stash['C'] = st[1]; return st

    def mu_hook(layer, mu, ctx):
        if MODE == 'read' and layer in s2card.EVERY3:
            W = cd.w[layer]
            vv = ((stash['C'].to(torch.float32) @ W) * W).sum(0).to(torch.float64)
            gate = torch.erfc(-(mu / vv.clamp_min(1e-10).sqrt()) / math.sqrt(2)) / 2
            stash['S'][layer] = torch.argsort(-torch.minimum(gate, 1 - gate).detach(), stable=True)
        return mu
    cd.hooks['state_in'], cd.hooks['mu'] = state_in, mu_hook
    if fine_ckpt:
        for k in ('_nonlinear_pair_batch', '_bd_update', '_flms_star'): cd.env[k] = s2card._ckpt(torch, cd.env[k])
        for k in ('push', 'read'): cd.shared[k] = s2card._ckpt(torch, cd.shared[k])
    return cd
