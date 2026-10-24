"""[opus-lowres] RQS round 2, piece 3 (byte-identical request diet; editor: apply(text) -> text; line-preserving). Composes after
editor_rqs2_strips_v3 and editor_rqs2_fhmdiag_v2.
 Live _astra_fhm_core (the def holding every core anchor exactly once):
  E7  fill_diagonal(_ps_K2132, _ps32(_ps_kdiag)) -> fill_diagonal(_ps_K2132, _ps_kdiag)
  E8  fill_diagonal(k22, _ps32(_ps_k22diag))     -> fill_diagonal(k22, _ps_k22diag)
      (a float32 target takes float64 values through the same round-to-nearest cast asarray(..., float32) applies; the pattern
      fill_diagonal(_ps_post_cov32, _ps_Cdiag) already runs on the platform; no later _ps32 call reads these conversions)
  E12 the layer publication also carries rqs2_vv = vv (= diag(Cpre), a read-only view of Cpre's diagonal)
  E13 the K21 publication also carries rqs2_kdiag = _ps_kdiag (the values fill_diagonal just wrote into K21's diagonal)
 Pair map _nonlinear_pair_batch (the def at line 844; co_firstlineno unchanged):
  E10 v = diag(C)    -> the published vv when C is the published Cpre (both are views of the same diagonal memory)
  E11 k3 = diag(K21) -> the published kdiag when K21 is the published ps_K21 (K21's diagonal is written once, from kdiag, before
      the pair map, and neither is written afterwards in the live code; every k3 consumer reads)
  E9  fill_diagonal(_ps_target, _ps32(post_k3)) -> fill_diagonal(_ps_target, post_k3) (as E7)
 Side effect: aux['pre_v'] is the core's vv, so _gn_dcov's _ps32(v) finds the core's own conversion (_ps32(_ps_bv), _ps_bv is vv).
Same values, dtypes and operation order everywhere; only requests disappear."""


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


CORE = [
 ("            fnp.fill_diagonal(_ps_K2132, _ps32(_ps_kdiag))\n",
  "            fnp.fill_diagonal(_ps_K2132, _ps_kdiag)   # RQS2 E7: the float32 target casts as _ps32 would\n"),
 ("            fnp.fill_diagonal(k22, _ps32(_ps_k22diag))\n",
  "            fnp.fill_diagonal(k22, _ps_k22diag)   # RQS2 E8\n"),
 ("            _E2_LAYER.update(mu=mu, Cpre=Cpre, vc=_e2_vc, s=_e2_s, a=_e2_a, g=response_gate)\n",
  "            _E2_LAYER.update(mu=mu, Cpre=Cpre, vc=_e2_vc, s=_e2_s, a=_e2_a, g=response_gate, rqs2_vv=vv)   # RQS2 E12\n"),
 ("            _E2_LAYER.update(ps_K21=K21, ps_K2132=_ps_K2132)\n",
  "            _E2_LAYER.update(ps_K21=K21, ps_K2132=_ps_K2132, rqs2_kdiag=_ps_kdiag)   # RQS2 E13\n"),
]
PAIR = [
 ("    v = fnp.diag(C)\n",
  "    v = _E2_LAYER['rqs2_vv'] if (_E2_LAYER.get('Cpre') is C and _E2_LAYER.get('rqs2_vv') is not None) else fnp.diag(C)   # RQS2 E10\n"),
 ("    k3 = fnp.diag(K21)\n",
  "    k3 = _E2_LAYER['rqs2_kdiag'] if (_E2_LAYER.get('ps_K21') is K21 and _E2_LAYER.get('rqs2_kdiag') is not None) else fnp.diag(K21)   # RQS2 E11\n"),
 ("    fnp.fill_diagonal(_ps_target, _ps32(post_k3))\n",
  "    fnp.fill_diagonal(_ps_target, post_k3)   # RQS2 E9\n"),
]


def apply(text):
    for head, R in (('def _astra_fhm_core(', CORE), ('def _nonlinear_pair_batch(', PAIR)):
        i, j = _live_def(text, head, [a for a, _ in R])
        body = text[i:j]
        for a, b in R:
            body = body.replace(a, b)
        text = text[:i] + body + text[j:]
    return text
