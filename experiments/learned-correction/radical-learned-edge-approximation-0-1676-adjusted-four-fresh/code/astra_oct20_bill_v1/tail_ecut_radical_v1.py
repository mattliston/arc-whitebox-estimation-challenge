# [astra-bill] 2026-10-20T19:15:43Z: six-op clipped radical edge approximation.
# Constants fitted only to the original scalar function on a fixed grid.
# This changes numeric outputs; no accuracy gain or identity is claimed.
assert _ECUT_C1 == 0.165 and _ECUT_C2 == 0.0098
_O20_ECUT_RADICAL_ON = True
_O20_ECUT_RADICAL_A = -0.2684473340048872
_O20_ECUT_RADICAL_B = 1.5215732663431163
_O20_ECUT_RADICAL_PARENT = _ecut_raw

def _ecut_raw(left, right, t, k, pf, gf, hold):
    """raw (n, n) float32 of the message edge: T_ij pdf_i gate_j on the kept rows x columns, 0 elsewhere.
    left, right: the two (n, 16) operands; t: edge parameters; k: the edge's persistent R, S planes and weight rows."""
    if not _O20_ECUT_RADICAL_ON or not _ECUT_STANDIN:
        return _O20_ECUT_RADICAL_PARENT(left, right, t, k, pf, gf, hold)
    _e2_H = _E2Hold(); _e2_append_inline = _e2_H.append; _e2_value_inline = None   # temporaries of this call, released together
    L = _E2_LAYER
    I, J = L['ecut_rows'], L['ecut_cols']
    n, ni, nj = int(pf.shape[0]), int(I.shape[0]), int(J.shape[0])
    R, S = k['R'], k['S']
    Ic = (_e2_append_inline((_e2_value_inline := I[:, None])) or _e2_value_inline)
    Rij = (_e2_append_inline((_e2_value_inline := R[Ic, J])) or _e2_value_inline)
    Sij = (_e2_append_inline((_e2_value_inline := S[Ic, J])) or _e2_value_inline)
    Sjit = (_e2_append_inline((_e2_value_inline := S[J, Ic])) or _e2_value_inline)                            # element (i, j) = S[J[j], I[i]]: the transposed block
    h = (_e2_append_inline((_e2_value_inline := (_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := left[I])) or _e2_value_inline) + t['bias']))) or _e2_value_inline)[:, None, :])) or _e2_value_inline) + (_e2_append_inline((_e2_value_inline := right[J])) or _e2_value_inline)
    hold.append(fnp.add(h, (_e2_append_inline((_e2_value_inline := fnp.einsum('ij,c->ijc', Rij, k['e0'], optimize=False))) or _e2_value_inline), out=h))
    hold.append(fnp.add(h, (_e2_append_inline((_e2_value_inline := fnp.einsum('ij,c->ijc', Sij, k['e1'], optimize=False))) or _e2_value_inline), out=h))
    hold.append(fnp.add(h, (_e2_append_inline((_e2_value_inline := fnp.einsum('ij,c->ijc', Sjit, k['e2'], optimize=False))) or _e2_value_inline), out=h))
    # Six metered elementwise operations, replacing the ten-op stand-in.
    w = fnp.empty((ni, nj, int(h.shape[2])), dtype=fnp.float32)
    hold.append(fnp.multiply(h, h, out=w))
    hold.append(fnp.add(w, _O20_ECUT_RADICAL_B, out=w))
    hold.append(fnp.sqrt(w, out=w))
    hold.append(fnp.add(w, _O20_ECUT_RADICAL_A, out=w))
    hold.append(fnp.divide(h, w, out=h))
    hold.append(fnp.clip(h, -1.0, 1.0, out=h))
    block = fnp.einsum('ij,i,j->ij', (_e2_append_inline((_e2_value_inline := (h @ t['output']))) or _e2_value_inline), (_e2_append_inline((_e2_value_inline := pf[I])) or _e2_value_inline), (_e2_append_inline((_e2_value_inline := gf[J])) or _e2_value_inline), optimize=False)
    raw = fnp.zeros((n, n), dtype=fnp.float32)
    (_e2_append_inline((_e2_value_inline := fnp.put(raw, _e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := (Ic * n))) or _e2_value_inline) + J))) or _e2_value_inline, block))) or _e2_value_inline)
    _ECUT_STATS.update(layers=_ECUT_STATS['layers'] + 1, kept=_ECUT_STATS['kept'] + ni * nj, full=_ECUT_STATS['full'] + n * n)
    return raw
