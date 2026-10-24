"""[opus-lowres] RQS round 2, piece 2 (byte-identical request diet; editor: apply(text) -> text). Inside the LIVE _astra_fhm_core
(the def holding every anchor exactly once), six diagonal re-reads become reuses of values the layer already holds:
 E1 _ps_kdiag: fnp.diag(Bpre) -> _ps_bv (the line above defines _ps_bv = vv if Bpre is Cpre else fnp.diag(Bpre): same array)
 E2 _ps_Cdiag0: fnp.diag(C) -> the pair map's own var (aux['post_v']) when C is still the pair map's cov object (_nonlinear_pair_batch
    ends with fill_diagonal(cov, var) and publishes L['pair'] = dict(cov=cov, aux=aux)); else fnp.diag(C) as before; .copy() kept
 E3 _ps_Bdiag: fnp.diag(Bpost) -> _ps_Cdiag when Bpost is the C the hook branch just filled (marker _E2_LAYER['post_cov32'][0] is Bpost,
    set right after fill_diagonal(C, _ps_Cdiag) under _FULL_LOWER_MOMENTS, where Bpost = C); no write to C's diagonal in between
 E4/E5 diag(target): extracted once (innovation) and reused for _ps_rest_diag (target is not written in between; _t31_commit is None)
 E6 _ps_cpdiag: fnp.diag(live) -> alive (live = fnp.outer(alive, alive), boolean: its diagonal IS alive)
Same values, dtypes and operation order downstream; one diag request (and its span) saved per site per layer. v2: every edit stays on its own line (line count unchanged, so line-number
edits composed after it still land)."""


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


R = [
 ("            _ps_kdiag = (_plane_cpdiag64 + fnp.diag(star_pre) + _ps_bv * bpre) + fnp.diag(Bpre) * (2 * bpre)\n",
  "            _ps_kdiag = (_plane_cpdiag64 + fnp.diag(star_pre) + _ps_bv * bpre) + _ps_bv * (2 * bpre)   # RQS2 E1: _ps_bv is diag(Bpre)\n"),
 ("                _ps_Cdiag0 = fnp.diag(C).copy()\n",
  "                _rqs2_pair = _E2_LAYER.get('pair'); _ps_Cdiag0 = (_rqs2_pair['aux']['post_v'] if (_rqs2_pair is not None and _rqs2_pair['cov'] is C) else fnp.diag(C)).copy()   # RQS2 E2\n"),
 ("            _ps_Bdiag = fnp.diag(Bpost)\n",
  "            _rqs2_pc = _E2_LAYER.get(\"post_cov32\"); _ps_Bdiag = _ps_Cdiag if (_rqs2_pc is not None and _rqs2_pc[0] is Bpost) else fnp.diag(Bpost)   # RQS2 E3\n"),
 ("            innovation = _e2_H(_e2_H(fnp.diag(target)) - _ps_inherited_diag) - _ps_trace_diag\n",
  "            _rqs2_tdiag = fnp.diag(target); innovation = _e2_H(_e2_H(_rqs2_tdiag) - _ps_inherited_diag) - _ps_trace_diag   # RQS2 E4 (reused at E5)\n"),
 ("            _ps_cpdiag = (_ps_inherited_diag + _ps_born_diag) * fnp.diag(live)\n",
  "            _ps_cpdiag = (_ps_inherited_diag + _ps_born_diag) * alive   # RQS2 E6: diag(outer(alive, alive)) is alive\n"),
 ("            _ps_rest_diag = (fnp.diag(target) - _ps_cpdiag) - _ps_trace_diag\n",
  "            _ps_rest_diag = (_rqs2_tdiag - _ps_cpdiag) - _ps_trace_diag   # RQS2 E5\n"),
]


def apply(text):
    i, j = _live_def(text, 'def _astra_fhm_core(', [a for a, _ in R])
    body = text[i:j]
    for a, b in R:
        body = body.replace(a, b)
    return text[:i] + body + text[j:]
