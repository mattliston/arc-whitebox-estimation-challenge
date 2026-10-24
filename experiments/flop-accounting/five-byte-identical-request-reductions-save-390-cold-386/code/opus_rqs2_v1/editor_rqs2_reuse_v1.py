"""[opus-lowres] RQS round 2, piece 5 (byte-identical request diet; editor: apply(text) -> text). Edits stay on their own lines;
one helper block is appended LAST. Composes after strips_v3 -> fhmdiag_v2 -> pairdiag_v1 -> misc_v1.
 R1 _xsg_inverse / R2 _unique_rows_via_scatter: put(inverse, idx, arange(k)) -> put(inverse, idx, AR) with AR one process-lifetime
    int64 arange of length >= k (numpy put takes values[i] for indices[i] when values are longer; flopscope bills put by numel(indices),
    so the bill drops by the arange's k); one arange request per call saved
 R3 _re22_energy: v[prefix_cols] was gathered twice (scale and the tail/prefix ratio): gathered once
 R4 _re22_capture: cols[:cut], cols[cut:] (two getitem views) -> one fnp.split(cols, [cut]) (the same two views)"""


def _live_def(text, head, anchors):
    hits = []
    i = text.find(head)
    while i >= 0:
        j = text.find('\ndef ', i + len(head)); j = len(text) if j < 0 else j
        if all(text[i:j].count(a) == 1 for a in anchors):
            hits.append((i, j))
        i = text.find(head, i + 1)
    assert len(hits) == 1, (head, len(hits))
    return hits[0]


EDITS = [
 ('def _xsg_inverse(', [
  ("    fnp.put(inverse, idx, fnp.arange(k, dtype=fnp.int64))\n",
   "    fnp.put(inverse, idx, _rqs2_ar(k))   # RQS2 R1: a kept arange (put uses its first k values)\n")]),
 ('def _unique_rows_via_scatter(', [
  ("    fnp.put(inverse, rows, fnp.arange(k, dtype=fnp.int64))\n",
   "    fnp.put(inverse, rows, _rqs2_ar(k))   # RQS2 R2\n")]),
 ('def _re22_energy(', [
  ("    scale=hold(fnp.sqrt(hold(v[prefix_cols])))\n",
   "    _rqs2_vp=hold(v[prefix_cols]);scale=hold(fnp.sqrt(_rqs2_vp))   # RQS2 R3: gathered once\n"),
  ("hold(fnp.sum(hold(1/hold(v[prefix_cols]))))", "hold(fnp.sum(hold(1/_rqs2_vp)))")]),
 ('def _re22_capture(', [
  ("    energy=_re22_energy(prefix,Q,rows,hold(cols[:cut]),hold(cols[cut:]),_E2_LAYER['vc'],hold)\n",
   "    energy=_re22_energy(prefix,Q,rows,*hold(fnp.split(cols,[cut])),_E2_LAYER['vc'],hold)   # RQS2 R4: one split for both views\n")]),
]

HELPER = '''

# BEGIN [opus-lowres] RQS2 R1/R2 (append LAST): one process-lifetime int64 arange (length >= every k seen; regrown by doubling) for the
# two inverse-map scatters that made fnp.arange(k) afresh per call
_RQS2_AR = []


def _rqs2_ar(k):
    if not _RQS2_AR or int(_RQS2_AR[0].shape[0]) < k:
        n = 1024
        while n < k:
            n *= 2
        _RQS2_AR[:] = [fnp.arange(n, dtype=fnp.int64)]
    return _RQS2_AR[0]
# END [opus-lowres] RQS2 R1/R2
'''


def apply(text):
    for head, R in EDITS:
        i, j = _live_def(text, head, [a for a, _ in R])
        body = text[i:j]
        for a, b in R:
            body = body.replace(a, b)
        text = text[:i] + body + text[j:]
    return text + HELPER
