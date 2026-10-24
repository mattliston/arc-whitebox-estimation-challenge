# PRIVATE research port of95c JSK-lite; not submission code.
# Native old-tier UV layout and all full content factors are preserved.
# Original module documentation below describes its historical parent;
# effective ranks are384/384/384 and sketch widths2048/1536/1536.
"""Research-only native S46 old-tier push/read simulator v2 (2026-10-10).

No torch import or work at module import.  Install with ``install(cd, torch)``
BEFORE the row-cut simulator and trainer.otr_m.install.  This is not a submission.
It needs forward/native parity and checkpoint/backward checks before training.

Audited source is quick_s46s_fresh_a.json's candidate, sha prefix 63f89e8d:
  native 8390-8495: index-sorted Nystrom join; ranks 384/320/384; Nystrom core;
  native 3556-3604: materialize L10; retain F10 + map + Fj lazily at L13;
  native 8140-8212, 8237-8278: local pair-mean truncation, reconstruction/read;
  native 8831-8897: r_u=256 at L10-15; tail traces use FULL uncut blocks.

Important: s6bcard_loop5js.make_push uses energy-ordered QR coordinates and a
384-column QR followed by truncation at L10.  Native chooses/index-sorts exactly
320 rows at L10, so truncating the original torch F is not the native cut.

Checkpoint state: F is always the full, untruncated canonical tensor.  At L13
native no longer needs a Gram core (there is no later join), so the outgoing
core SLOT stores cat((F10, overlap.T), dim=1).  L14/15 recover the immutable lazy
blocks from that incoming tensor.  All persistent differentiable state therefore
travels through ordinary checkpoint arguments; no layer-keyed cache retains it.
The small ``cd.native_push_context`` dictionary is replaced on each push and
used only by reads/row repairs in that SAME step.  It never supplies carried
state.  This core-slot convention is private to this installed push.

Row-cut integration: context['blocks'] contains dicts with lo, hi, U, F, pairs,
where U/F are FULL factors in their OWN block coordinates (320 for the L10
prefix after L13; 384 for L13's recent block).  Use these full blocks for tail
trace and omitted-radial repair; do not use the locally truncated factors.
The parent row-cut wrapper may zero pp rows but should leave U/F/core untouched.

Tensor footprint: after L13 the packed core adds 320*(prefix_columns+384)
float32 elements (roughly 13-17 MB per retained boundary for normal rosters).
There are only two propagated boundaries after this final join.  No dense
rank-by-column copies are cached separately by the context.
"""


JOINS = {7: (0, 2), 10: (3, 5), 13: (6, 8)}
RANKS = {7: 384, 10: 384, 13: 384}
JSK_KC = {7: 2048, 10: 1536, 13: 1536}


def pair_blocks(stops, lo, hi):
    """Relative (offset, half-width) entries of intact cohorts in [lo, hi)."""
    blocks = []
    start = 0
    covered = 0
    for stop in stops:
        stop = int(stop)
        width = stop - start
        if width and start >= lo and stop <= hi:
            assert width % 2 == 0, (start, stop)
            blocks.append((start - lo, width // 2))
            covered += width
        start = stop
    assert covered == hi - lo, (lo, hi, covered)
    return tuple(blocks)


def uv_split(t, F, pairs, ru):
    ru = min(int(ru), int(F.shape[0]))
    means = [(F[:ru, a:a+h] + F[:ru, a+h:a+2*h]) * 0.5
             for a, h in pairs]
    gaps = [(F[:, a:a+h] - F[:, a+h:a+2*h]) * 0.5
            for a, h in pairs]
    return t.cat(means, 1), t.cat(gaps, 1), ru


def uv_reconstruct(t, U, F, pairs, ru):
    """U[:, :ru] Fmean +/- U Fgap, restored to cohort roster order."""
    mean, gap, ru = uv_split(t, F, pairs, ru)
    pu, pv = U[:, :ru] @ mean, U @ gap
    pieces, b = [], 0
    for _, h in pairs:
        pieces.extend((pu[:, b:b+h] + pv[:, b:b+h],
                       pu[:, b:b+h] - pv[:, b:b+h]))
        b += h
    return t.cat(pieces, 1)


def uv_inner(t, left_sq, lam, F, pairs, ru):
    """(left_sq*lam) F_local.T without materializing the modified F."""
    mean, gap, ru = uv_split(t, F, pairs, ru)
    weighted = left_sq * lam[None, :]
    sums = [weighted[:, a:a+h] + weighted[:, a+h:a+2*h] for a, h in pairs]
    diffs = [weighted[:, a:a+h] - weighted[:, a+h:a+2*h] for a, h in pairs]
    inner = t.cat(diffs, 1) @ gap.T
    common = t.cat(sums, 1) @ mean.T
    # Out-of-place, so a differentiable simulator does not mutate saved inputs.
    return t.cat((inner[:, :ru] + common, inner[:, ru:]), 1)


def install(cd, t, ru=256):
    """Replace base tier push/read before the trainer's physics wrappers."""
    assert not hasattr(cd, 'native_push_context'), 'install once per card'
    assert int(ru) > 0
    read0 = cd.shared['read']
    context = {}
    cd.native_push_context = context

    def push(tt, layer, w, p, lam, previous, Q, F, core, stops, rank=384):
        assert tt is t
        context.clear()
        assert len(stops) == layer
        wp = w[previous, :].T
        old_columns = (int(stops[8]) if layer > 13 else
                       int(stops[5]) if layer > 10 else
                       int(stops[2]) if layer > 7 else 0)
        if layer < 7:
            return wp @ p[previous, :], None, F, core, 0
        Qp = Q[previous, :] if old_columns else None
        lazy = None
        if layer in JOINS:
            r = RANKS[layer]
            assert len(previous) >= r, (layer, len(previous), r)
            start, stop = old_columns, int(stops[JOINS[layer][1]])
            J = p[previous, start:stop]
            weight = lam[start:stop] * lam[start:stop]
            d = (J * J) @ weight
            if old_columns:
                d = d + ((Qp @ core) * Qp).sum(1)
            rows = t.sort(t.argsort(-d.to(t.float64), stable=True)[:r]).values
            # Current95c: full energy chooses rows and heaviest columns.
            # All content factors below still use every joined column.
            column_energy = (J * J).sum(0) * weight
            columns = t.argsort(-column_energy.to(t.float64), stable=True)[:JSK_KC[layer]]
            Jsk = J[:, columns]
            Y = Jsk @ (weight[columns, None] * Jsk[rows, :].T)
            if old_columns:
                Y = Y + Qp @ (core @ Qp[rows, :].T)
            # Match the card/native QR precision convention; R is needed by the
            # native Nystrom-core shortcut, not the original torch Gram update.
            q64, r64 = t.linalg.qr(Y.to(t.float64), mode='reduced')
            Qn, R = q64.to(w.dtype), r64.to(w.dtype)
            Fj = Qn.T @ J
            if old_columns:
                overlap = Qn.T @ Qp
                Fprev = F
                F = t.cat((overlap @ Fprev, Fj), 1)
            else:
                F = Fj
            if layer < 13:
                # numpy.linalg (behind fnp.linalg.solve) computes even f32
                # inputs in float64, then returns the input result dtype.
                core = t.linalg.solve(Qn[rows, :].to(t.float64),
                                      R.T.to(t.float64)).to(w.dtype)
                core = (core + core.T) * 0.5
                # Match current95c literally: joined-factor trace, including
                # its use at the second join. Do not add old-core trace here.
                full_joined_trace = ((Fj * Fj) @ weight).sum()
                core = core * (full_joined_trace / t.trace(core))
            else:
                assert old_columns == int(stops[5])
                assert tuple(overlap.shape) == (RANKS[13], RANKS[10])
                # No further join ever consumes a Gram core.  Store the exact
                # lazy prefix and overlap as normal differentiable state.
                core = t.cat((Fprev, overlap.T), 1)
                lazy = (Fprev, overlap, Fj)
            basis, old_columns = Qn, stop
        else:
            basis = Qp
            if layer > 13:
                split = int(stops[5])
                assert tuple(core.shape) == (RANKS[10], split + RANKS[13]), tuple(core.shape)
                lazy = (core[:, :split], core[:, split:].T, F[:, split:])
        U = wp @ basis
        if lazy is None:
            blocks = (dict(lo=0, hi=old_columns, U=U, F=F,
                           pairs=pair_blocks(stops, 0, old_columns)),)
            mapping = None
        else:
            base, mapping, recent = lazy
            split = int(stops[5])
            blocks = (dict(lo=0, hi=split, U=U @ mapping, F=base,
                           pairs=pair_blocks(stops, 0, split)),
                      dict(lo=split, hi=old_columns, U=U, F=recent,
                           pairs=pair_blocks(stops, split, old_columns)))
        use_uv = layer >= 10
        old = t.cat([uv_reconstruct(t, b['U'], b['F'], b['pairs'], ru)
                     if use_uv else b['U'] @ b['F'] for b in blocks], 1)
        pp = t.cat((old, wp @ p[previous, old_columns:]), 1)
        context.update(layer=layer, U=U, F=F, core=core,
                       old_columns=old_columns, blocks=blocks, mapping=mapping,
                       ru=int(ru), truncated=use_uv)
        return pp, U, F, core, old_columns

    def read(tt, pp, lam, response, live, U, F, old_columns):
        if not old_columns or not context.get('truncated'):
            return read0(tt, pp, lam, response, live, U, F, old_columns)
        # This transient context is rebuilt by every (also recomputed) push.
        # It must never be used to recover a previous layer's carried state.
        assert context['layer'] == cd.call_layer
        assert context['U'] is U and context['F'] is F
        assert context['old_columns'] == old_columns
        left, right = pp[response], pp[live]
        inners = [uv_inner(t, left[:, b['lo']:b['hi']] ** 2,
                           lam[b['lo']:b['hi']], b['F'], b['pairs'], ru)
                  for b in context['blocks']]
        if context['mapping'] is None:
            inner = inners[0]
        else:
            inner = inners[0] @ context['mapping'].T + inners[1]
        old_plane = inner @ U[live].T
        ly, ry = left[:, old_columns:], right[:, old_columns:]
        young_plane = (ly * ly) @ (ry * lam[old_columns:]).T
        return young_plane + old_plane

    cd.shared['push'] = push
    cd.shared['read'] = read
    return cd
