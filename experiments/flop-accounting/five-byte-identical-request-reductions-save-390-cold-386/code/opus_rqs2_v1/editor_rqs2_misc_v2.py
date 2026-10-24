"""[opus-lowres] RQS round 2, piece 4, v2 (F3 zero made by fnp.zeros((), float32): release STEP6 native parity found v1's asarray(np.float32(0)) billed 1 flop in-process, 0 via the client) (byte-identical request diet; editor: apply(text) -> text). Edits stay on their own lines;
one helper block is appended LAST (after every fragment). Composes after strips_v3 -> fhmdiag_v2 -> pairdiag_v1.
 F1 _astra_fhm_effective: the changed-row COUNT (greater + sum + int fetch = 3 requests per layer) fed only
    _ASTRA_FHM_STATS['changed_rows'][i][1], which nothing reads (the end-of-predict check reads the layer index x[0] only):
    changed = -1, both stats appends kept.
 F2 _rtp_qr_basis: the three scalar reads of the 2x2 reduced-QR R (three __getitem__ requests) become one fetch (r.tolist());
    the same float64 values, the same Python-side tests.
 F3 _re22_energy / _re22_features: fnp.float32(0) (a fresh create_from_data upload per call) -> one process-lifetime 0-d float32
    zero made at first use (the same server object type and value: 0-d float32 +0.0)."""


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
 ('def _astra_fhm_effective(', [
  ("    changed = int(hold(fnp.sum(hold(fnp.greater(effective, delta)))))\n",
   "    changed = -1   # RQS2 F1: the count was a diagnostic only (the stats check reads the layer index); 3 requests per layer\n")]),
 ('def _rtp_qr_basis(', [
  ("    r0,r01,r1=abs(float(r[0,0])),abs(float(r[0,1])),abs(float(r[1,1]))\n",
   "    _rqs2_r=r.tolist();r0,r01,r1=abs(float(_rqs2_r[0][0])),abs(float(_rqs2_r[0][1])),abs(float(_rqs2_r[1][1]))   # RQS2 F2: one fetch\n")]),
 ('def _re22_energy(', [
  ("fnp.put(residual,hold(hold(ir*p)+ic),fnp.float32(0))", "fnp.put(residual,hold(hold(ir*p)+ic),_rqs2_z32())")]),
 ('def _re22_features(', [
  ("fnp.maximum(hold(hold(prior*prior)+energy),fnp.float32(0))", "fnp.maximum(hold(hold(prior*prior)+energy),_rqs2_z32())")]),
]

HELPER = '''

# BEGIN [opus-lowres] RQS2 F3 (append LAST): one process-lifetime 0-d float32 zero for the two re22 sites that uploaded
# fnp.float32(0) afresh per call (create_from_data); made at first use, then reused (same dtype, same +0.0)
_RQS2_Z32 = []


def _rqs2_z32():
    if not _RQS2_Z32:
        _RQS2_Z32.append(fnp.zeros((), dtype=fnp.float32))   # v2: zeros bills 0 in-process and through the client (asarray of a numpy scalar billed 1 flop in-process only: STEP6 parity)
    return _RQS2_Z32[0]
# END [opus-lowres] RQS2 F3
'''


def apply(text):
    for head, R in EDITS:
        i, j = _live_def(text, head, [a for a, _ in R])
        body = text[i:j]
        for a, b in R:
            body = body.replace(a, b)
        text = text[:i] + body + text[j:]
    return text + HELPER
