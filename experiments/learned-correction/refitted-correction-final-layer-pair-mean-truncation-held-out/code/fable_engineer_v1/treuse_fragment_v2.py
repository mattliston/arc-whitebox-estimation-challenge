

# BEGIN FABLE-ENGINEER TRANSFORM REUSE v2 (E-73 = E-71 group (b), 2026-09-25; bit-identical, bill down; v2 allows _D6X_OLD_PPC, the ab-era
# dispatcher (ab tree kernel, else _AB_OLD_PPC, which still clears): v1 cleared on it and refused 11-16 audited reuses per net): the tree kernel's forward-A transform
# of an operand sits in the arena's A region until the next forward-A. When the SAME operand object comes back with the same (L, m, k) in
# the very next forward-A, e.g. the two column packets of a push, or the covariance chain's products sharing a left factor, the
# transform is skipped and the resident tree is returned. Proof (e73_reuse_audit.py, S24c in process, nets 35/100/37/39): at all 76
# such events the normal recomputation left the A region bit-identical, so skipping it changes nothing. Runtime guard, conservative: any
# arena-touching function other than the audited dispatchers (listed below) and the tree views, any alias that reaches an older kernel
# path, and every prediction start clear the resident key, so a reuse happens only in the audited call patterns.
_TR_LAST = [None]          # (operand object, (L, m, k)): the transform resident in the A region (the reference keeps the id unique)
_TR_STATS = {'reused': 0, 'computed': 0}
_TR_CLEAR = ('_predict_binary', '_pb_predict', '_w5_port_new', '_bb_product', '_mm_batch', '_packed_read5', '_packed_read6', '_push6_padded',
             '_packed_product_3', '_packed_product_6', '_nonlinear_pair_batch', '_e2_pair_arena', '_shared_old384_mm', '_slr_join_factors',
             '_ipt_r6_side', '_read6_frames', '_read5_view', '_read5_child_frame', '_read5_packet_frames', '_bh_fwd', '_bh_extent', '_bh_leafx',
             '_bh_o_frame', '_packed_frame', '_a_frame', '_b_frame', '_o_frame', '_cp_a_frame', '_bb_o_frame', '_bb_views', '_exact_view',
             '_ul_mode', '_ul_view', '_p1_d6_832_bind', '_cp3_bind_arena_views', '_cp3_winograd_arena_depth4', '_bind_push6',
             # aliases that reach older kernel paths (frames, the old tree kernel, the batched paths)
             '_AB_OLD_BB', '_IPT_OLD_BB', '_AB_OLD_READ5', '_IPT_OLD_READ5', '_IPT_R6_OLD', '_BB6_OLD_MM_BATCH', '_CAP_BH_FWD', '_EV5_BH_FWD',
             '_EV5_BH_LEAFX', '_EV5_BH_O_FRAME', '_CAP_PACKED_FRAME', '_CAP_READ5_FRAMES', '_UL_A_FULL', '_UL_B_FULL', '_AB_OLD_PPC',
             '_IPT_OLD_PPC', '_AB_OLD_CP', '_IPT_OLD_CP', '_AB_OLD_W5', '_IPT_OLD_W5', '_D6X_OLD_W5P')
_TR_WRAPPED = []


def _tr_clearing(f):
    def w(*a, **k):
        _TR_LAST[0] = None
        return f(*a, **k)
    w.__name__ = getattr(f, '__name__', 'w'); w.__wrapped__ = f
    return w


for _tr_n in _TR_CLEAR:
    if callable(globals().get(_tr_n)):
        globals()[_tr_n] = _tr_clearing(globals()[_tr_n]); _TR_WRAPPED.append(_tr_n)
_TR_OLD_FWD_A = _ipt_forward_a_ab


def _ipt_forward_a_ab(a, L, m, k):
    last = _TR_LAST[0]
    if last is not None and last[0] is a and last[1] == (L, m, k):
        _TR_STATS['reused'] += 1
        return _ipt_side(0, L, m, k)[0]
    TA = _TR_OLD_FWD_A(a, L, m, k)
    _TR_LAST[0] = (a, (L, m, k))
    _TR_STATS['computed'] += 1
    return TA
# END FABLE-ENGINEER TRANSFORM REUSE v2
