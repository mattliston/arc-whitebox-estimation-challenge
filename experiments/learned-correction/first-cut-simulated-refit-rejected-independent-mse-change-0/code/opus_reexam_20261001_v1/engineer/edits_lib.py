"""TAG opus_engineer source edits shared by the stack builds; the texts are those of a08_f32_core/build.py (checked against its
build records when this file was made).
a08_edits(traces, core): (T) float32 tail traces, (C) one-solve Nystrom core (q64 = float64 solve, q32 = float32 solve).
rows32_edits(): the three row cuts call _rc_round (hzs/fragment_rc_v1.py) instead of rounding their kept count up to 64."""

OTT_OLD = '''def _ott_trace(U, F, lam, tail):
    U64 = U.astype(fnp.float64)
    F64 = F.astype(fnp.float64)
    norm2 = fnp.sum(F64 * F64, axis=0)
    _OTT_DATA.setdefault('blocks', []).append((U64[tail, :], F64, norm2, tail))
    rho = fnp.sum(U64 * U64) / (int(U.shape[0]) * int(U.shape[1]))
    vector = F64 @ (lam.astype(fnp.float64) * norm2)
    return (U64[tail, :] @ vector) * (3 * rho)
'''
OTT_NEW = '''def _ott_trace(U, F, lam, tail):
    # opus-engineer a08 (rounding-level): U, F and lam are float32 and this first-chaos tail estimate needs no more; only the repair widens
    norm2 = fnp.sum(F * F, axis=0)
    Ut = U[tail, :]
    _OTT_DATA.setdefault('blocks', []).append((Ut, F, norm2, tail))
    rho = fnp.sum(U * U) / (int(U.shape[0]) * int(U.shape[1]))
    vector = F @ (lam * norm2)
    return ((Ut @ vector) * (3 * rho)).astype(fnp.float64)
'''
OMIT_OLD = '''            sums.append((Ut.T @ beta_tail) @ F)
'''
OMIT_NEW = '''            sums.append((Ut.T @ beta_tail.astype(fnp.float32)) @ F)   # opus-engineer a08: Ut and F are float32
'''
YNT_OLD = '''    q, selected, base, t, s = newborn
    q, births = int(q), int(selected.shape[0])
    weights = coefficient.astype(fnp.float64)
    B = base.astype(fnp.float64)
    t64, s64 = t.astype(fnp.float64), s.astype(fnp.float64)
    normB = fnp.sum(B * B, axis=0)
    ct = weights[q:q + births] * (normB + t64 * t64)
    cs = weights[q + births:] * (normB + s64 * s64)
    tau = B @ (ct + cs)
    if q:
        P = p[:, :q].astype(fnp.float64)
        carried = P @ (weights[:q] * fnp.sum(P * P, axis=0))
        fnp.put(tau, previous, tau[previous] + carried)
    fnp.put(tau, selected, tau[selected] + t64 * ct + s64 * cs)
    # Ambient weight dimension is 1024, including rows outside previous.
    wt = w[:, tail].astype(fnp.float64)
    normw = fnp.sum(wt * wt, axis=0)
    return (3 * normw / (int(w.shape[0]) + 2)) * (wt.T @ tau)
'''
YNT_NEW = '''    # opus-engineer a08 (rounding-level): every operand is float32 and this first-chaos tail estimate needs no more; only the repair widens
    q, selected, base, t, s = newborn
    q, births = int(q), int(selected.shape[0])
    normB = fnp.sum(base * base, axis=0)
    ct = coefficient[q:q + births] * (normB + t * t)
    cs = coefficient[q + births:] * (normB + s * s)
    tau = base @ (ct + cs)
    if q:
        P = p[:, :q]
        carried = P @ (coefficient[:q] * fnp.sum(P * P, axis=0))
        fnp.put(tau, previous, tau[previous] + carried)
    fnp.put(tau, selected, tau[selected] + t * ct + s * cs)
    # Ambient weight dimension is 1024, including rows outside previous.
    wt = w[:, tail]
    normw = fnp.sum(wt * wt, axis=0)
    return ((3 * normw / (int(w.shape[0]) + 2)) * (wt.T @ tau)).astype(fnp.float64)
'''
CORE_OLD = '''        K = Y[_jn_rows, :].astype(fnp.float64) if _need_core else None
        Qn, unused_R = fnp.linalg.qr(Y, mode='reduced')
        core = None
        if _need_core:
            K = (K + K.T) * 0.5
            R64 = unused_R.astype(fnp.float64)
            core = (R64 @ fnp.linalg.solve(K, R64.T)).astype(fnp.float32)
            core = (core + core.T) * 0.5
            del R64
'''
CORE_NEW = '''        K = None
        Qn, unused_R = fnp.linalg.qr(Y, mode='reduced')
        core = None
        if _need_core:   # opus-engineer a08: R K^-1 R' with K = Y[rows] = Qn[rows] R is Qn[rows]^-1 R' (one solve, no product)
            R64 = unused_R.astype(fnp.%s)
            core = fnp.linalg.solve(Qn[_jn_rows, :].astype(fnp.%s), R64.T).astype(fnp.float32)
            core = (core + core.T) * 0.5
            del R64
'''


def a08_edits(traces, core):
    assert traces in '01' and core in ('none', 'q64', 'q32')
    edits = []
    if traces == '1':
        edits += [(OTT_OLD, OTT_NEW, 1), (OMIT_OLD, OMIT_NEW, 1), (YNT_OLD, YNT_NEW, 1)]
    if core != 'none':
        dt = 'float64' if core == 'q64' else 'float32'
        edits.append((CORE_OLD, CORE_NEW % (dt, dt), 1))
    return edits


def rows32_edits():
    return [
        ("    count = min(n, (int(fnp.nonzero(score >= 0.0)[0].shape[0]) + 63) // 64 * 64)\n",
         "    count = _rc_round(int(fnp.nonzero(score >= 0.0)[0].shape[0]), n, int(previous.shape[0]))   # opus-engineer rc v1 (was: up to 64)\n", 1),
        ("    kept = min(m, (int(fnp.nonzero(score >= 0.0)[0].shape[0]) + 63) // 64 * 64)\n",
         "    kept = _rc_round(int(fnp.nonzero(score >= 0.0)[0].shape[0]), m)   # opus-engineer rc v1 (was: up to 64)\n", 2),
        ("    kept = min(m, (int(fnp.nonzero(~negative)[0].shape[0]) + 63) // 64 * 64)\n    if m - kept < 64:\n",
         "    kept = _rc_round(int(fnp.nonzero(~negative)[0].shape[0]), m, int(previous.shape[0]))   # opus-engineer rc v1 (was: up to 64)\n    if m - kept < _RC_STEP:\n", 1),
    ]


def features_f32_edits(parent_text):
    # (F) the mean-correction net's nine per-layer plane reductions accumulate float32 planes in float64 (billed 2x) and are then
    # stacked into a float32 feature matrix; same reductions with float32 accumulation. Rounding level.
    lines = [line for line in parent_text.split('\n') if line.startswith("    vals = (c[0], c[1], c[2], aux[")]
    assert len(lines) == 1 and lines[0].count(', dtype=fnp.float64') == 9, (len(lines), [x.count(', dtype=fnp.float64') for x in lines])
    old = lines[0] + '\n'
    return [(old, old.replace(', dtype=fnp.float64', ''), 1)]
