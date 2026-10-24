# BEGIN OPUS-ENGINEER LAYER-15 COPY OF THE LEARNED CORRECTION (a45 l15 v1, 2026-10-10; OFF unless _L15_W is given: then bit-identical).
# The learned mean correction's encoder (enc0 / enc2 / enc4, weights and biases) and its message weights (msg_w, msg_b) are shared
# by layers 1-15; its centres, scales, rms and the 64 -> 1 readout (head, head_b, gain) are per layer. At layer 15 the correction
# touches nothing downstream, so a copy for layer 15 alone can be fitted offline on dumped layer-15 inputs (lead 10-10 03:05Z item 3,
# 03:08Z). With _L15_W given, gnn_step at l == 15 takes the copy's arrays where it has them and the shared ones elsewhere: the same
# operations on arrays of the same shapes, so no bill and no request beyond the copy's one unpack per process (on its first use).
#   _L15_W  None, or a dict like _GNN_LITERAL ({name: (shape, base64 of zlib of the float32 bytes)}) with any of
#           enc0_w (128, 126) enc0_b (128,) enc2_w (128, 128) enc2_b (128,) enc4_w (64, 128) enc4_b (64,) msg_w (48, 64) msg_b (48,)
#           and layer 15's own rows: centers (30,) scales (30,) rms () head (64,) head_b () gain (1,) (gain raw, as P['gain'][15:16]).
# The head columns of layer 15 (fragment_hcf, lever HCF@) add to the bias gnn_step uses at layer 15, the copy's enc0_b when given.
_L15_W = None
_L15_CACHE = {}


def _l15_arrays():
    got = _L15_CACHE.get('w')
    if got is None:
        got = _L15_CACHE['w'] = _unpack_static_fp32(_L15_W)
    return got


def _l15_enc(k, l, P):
    """enc0_w', enc0_b, enc2_w', enc2_b, enc4_w', enc4_b for layer l: layer 15's copy where given, else the shared arrays."""
    if l != 15 or _L15_W is None:
        return k['enc0_wT'], P['enc0_b'], k['enc2_wT'], P['enc2_b'], k['enc4_wT'], P['enc4_b']
    got = _L15_CACHE.get('enc')
    if got is None or got[0] is not k:
        W = _l15_arrays()
        got = _L15_CACHE['enc'] = (k, tuple((W[name].T if name in W else k[name + 'T']) if name.endswith('_w') else W.get(name, P[name])
                                            for name in ('enc0_w', 'enc0_b', 'enc2_w', 'enc2_b', 'enc4_w', 'enc4_b')))
    return got[1]


def _l15_msgk(k, l):
    """The constants _e2_messages reads (msg_wT, P['msg_b']): layer 15's copy where given, else k itself."""
    if l != 15 or _L15_W is None:
        return k
    W = _l15_arrays()
    if 'msg_w' not in W and 'msg_b' not in W:
        return k
    got = _L15_CACHE.get('msgk')
    if got is None or got[0] is not k:
        P = k['P']
        got = _L15_CACHE['msgk'] = (k, dict(k, msg_wT=W['msg_w'].T if 'msg_w' in W else k['msg_wT'], P=dict(P, msg_b=W.get('msg_b', P['msg_b']))))
    return got[1]


_L15_PARENT_LAYER = _e2_gnn_layer


def _e2_gnn_layer(k, l):
    c = _L15_PARENT_LAYER(k, l)
    if l != 15 or _L15_W is None:
        return c
    got = _L15_CACHE.get('layer')
    if got is None or got[0] is not c:
        W = _l15_arrays()
        own = dict(c)
        for name in ('centers', 'scales', 'rms', 'head', 'head_b'):
            if name in W:
                own[name] = W[name]
        if 'gain' in W:
            own['gain'] = 1 + 0.25 * fnp.tanh(W['gain'])
        got = _L15_CACHE['layer'] = (c, own)
    return got[1]
# END OPUS-ENGINEER LAYER-15 COPY OF THE LEARNED CORRECTION
