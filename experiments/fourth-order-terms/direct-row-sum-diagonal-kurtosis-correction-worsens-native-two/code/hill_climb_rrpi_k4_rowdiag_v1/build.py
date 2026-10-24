"""Source-only full14 K4 row/diagonal projection; no imports of numerical code."""
from pathlib import Path
import ast, hashlib, json

D = Path(__file__).resolve().parent
P = D.parent/'hill_climb_pair_rr_inverse_v1/candidate.py'
EXPECTED = 'a3efb3a1066506e5bc2bc5a5da668056277ba65ea6cdd88c1de3be636fd419f3'
raw = P.read_bytes()
assert hashlib.sha256(raw).hexdigest() == EXPECTED
original = raw.decode()
edits = [
    ("        _PHYS_HANDLES.clear()\n        for l in range(depth):",
     "        _PHYS_HANDLES.clear()\n        _k4r_previous = None   # prediction-local vectors from outgoing layers1-14\n        for l in range(depth):"),
    ("post_v=var, post_k3=post_k3, post_k4=post_k4)\n    L['pair'] = dict(cov=cov, den=den, aux=aux, d=d, ss=ss, ss32=_ps_ss32, mgs=mgs)",
     "post_v=var, post_k3=post_k3, post_k4=post_k4, post22=post22)\n    L['pair'] = dict(cov=cov, den=den, aux=aux, d=d, ss=ss, ss32=_ps_ss32, mgs=mgs)"),
    ("                m, aux = nonlinear_scalar(mu, v, k3, _e2_H(_e2_H(radial * v) * v))\n                m = m + _e2_H(final_hook(l, m, aux, _e2_H(W[l])))",
     "                if _K4R_ALPHA != 0.0 and l in _K4R_LAYERS:\n"
     "                    if _k4r_previous is None or _k4r_previous[0] != l - 1:\n"
     "                        raise RuntimeError('K4 row/diagonal terminal predecessor ownership')\n"
     "                    _k4r_terminal = _e2_H(_e2_H(radial * v) * v)\n"
     "                    _k4r_terminal = _k4r_terminal + _K4R_ALPHA * _k4r_delta(w32, _k4r_previous[1])\n"
     "                    _k4r_previous = None\n"
     "                    m, aux = nonlinear_scalar(mu, v, k3, _k4r_terminal)\n"
     "                else:\n"
     "                    m, aux = nonlinear_scalar(mu, v, k3, _e2_H(_e2_H(radial * v) * v))\n"
     "                m = m + _e2_H(final_hook(l, m, aux, _e2_H(W[l])))"),
    ("            _e2_VV = _e2_H(_e2_H(_e2_pair_state(n))['SVVv'])[2]",
     "            if _K4R_ALPHA != 0.0 and l in _K4R_LAYERS:\n"
     "                if _k4r_previous is None or _k4r_previous[0] != l - 1:\n"
     "                    raise RuntimeError('K4 row/diagonal hidden predecessor ownership')\n"
     "                k4 = k4 + _K4R_ALPHA * _k4r_delta(w32, _k4r_previous[1])\n"
     "            _k4r_previous = None\n"
     "            _e2_VV = _e2_H(_e2_H(_e2_pair_state(n))['SVVv'])[2]"),
    ("            previous, previous_alive = (live_indices, alive)\n            radial = radial_new\n        return fnp.asarray",
     "            previous, previous_alive = (live_indices, alive)\n            radial = radial_new\n"
     "            if _K4R_ALPHA != 0.0 and l + 1 in _K4R_LAYERS:\n"
     "                _k4r_previous = (l, _k4r_state(aux['post_k4'], aux['post22'], C, radial))\n"
     "        return fnp.asarray"),
]
body = original
for before, after in edits:
    assert body.count(before) == 1 and after not in body
    body = body.replace(before, after, 1)
inverse = body
for before, after in reversed(edits):
    assert inverse.count(after) == 1
    inverse = inverse.replace(after, before, 1)
assert inverse == original
fragment = (D/'fragment.py').read_text()
tree = ast.parse(fragment)
assert not any(isinstance(x,(ast.Import,ast.ImportFrom,ast.AugAssign)) for x in ast.walk(tree))
candidate = body+'\n\n'+fragment
compile(candidate,str(D/'candidate.py'),'exec')
assert candidate.count('_K4R_ALPHA = 1.0') == 1
zero = candidate.replace('_K4R_ALPHA = 1.0','_K4R_ALPHA = 0.0',1)
for name,value in [('candidate.py',candidate),('candidate_zero.py',zero)]:
    with (D/name).open('x') as f:f.write(value)
def pin(p):
    b=p.read_bytes();return dict(path=str(p.resolve()),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
record=dict(parent=pin(P),candidate=pin(D/'candidate.py'),zero=pin(D/'candidate_zero.py'),fragment=pin(D/'fragment.py'),
            replacements=[dict(before=a,after=b,count=1) for a,b in edits],
            body_inverse_whole_source_exact=True,append_separator='\n\n',compiled=True,
            alpha=1.0,layers=list(range(2,16)),capture_layers=list(range(1,15)),additive_after_existing_hidden_k4_band=True,
            terminal_adds_to_existing_radial_v_squared=True,zero_branch_no_new_array_arithmetic=True,
            precision='FP64 residual, symmetric row/diagonal projection, squared exact FP32 weights widened to FP64, two vector-matrix products.',
            source_cost_upper_bound_giga_flops=0.25,
            scope='CR-12 repeated-face M approximated by matrix matching symmetric rows and diagonal; vectors u/v plus scalar carried prediction-locally.',
            no_coefficient_search=True,no_model_or_array_job=True,inactive_until_passive_read_information=True,
            acquisition_only_no_score_gate=True,coadaptation_not_yet_implemented=True)
with (D/'source_delta.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
print(json.dumps({k:record[k] for k in ('candidate','zero','fragment')}))
