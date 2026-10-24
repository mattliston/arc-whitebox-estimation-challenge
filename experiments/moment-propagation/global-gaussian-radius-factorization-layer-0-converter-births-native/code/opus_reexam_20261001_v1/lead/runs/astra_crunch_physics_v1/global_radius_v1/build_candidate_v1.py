"""Assemble fixed global-radius v18c experiment with L0-only consistency hooks."""
from pathlib import Path
import ast,hashlib,json
H=Path(__file__).resolve().parent;R=H.parents[1]
P=R/'astra_crunch_runtime_v1/failure336631_v1/candidate_setup_repair_v18c.py'
text=P.read_text();assert hashlib.sha256(P.read_bytes()).hexdigest()=='3e333d59c518f434a11699e168ca9110b17b779480a5b287c0887cd88ef90ce4'
tree=ast.parse(text);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_astra_fhm_core')
lines=text.splitlines(keepends=True);body=''.join(lines[node.lineno-1:node.end_lineno]);edits=[]
def edit(old,new):
    global body
    assert body.count(old)==1,(old,body.count(old));body=body.replace(old,new);edits.append(old)
edit('        radial = 0.0\n','        radial = _GR_INPUT_RADIAL\n')
edit("            _e2_beta = 2 * _e2_H(aux['gaussian_mean'] - _e2_H(host * aux['gate']))\n",
     "            _e2_beta = _gr_l0_beta(aux,host) if l==0 else 2 * _e2_H(aux['gaussian_mean'] - _e2_H(host * aux['gate']))\n")
edit("            birth = _bsk(birth, aux)   # opus-engineer bsk: the kink density to first order in the unit's skewness\n",
     "            birth = _gr_l0_birth(aux) if l==0 else _bsk(birth, aux)   # exact spherical mixed moment at L0\n")
old="            b = _e2_H(gq * bpre) + _e2_H(_e2_H(_e2_H(radial * d) * vv) / _S19_BSRC_DIV) if _FULL_LOWER_MOMENTS else _e2_H(_e2_H(gq * bpre) + _e2_H(_e2_H(_e2_H(radial * d) * vv) / _S19_BSRC_DIV)) * alive\n"
edit(old,"            b = _gr_l0_bsource(aux,host) if l==0 else ("+old.strip()[4:]+")\n")
text=''.join(lines[:node.lineno-1])+body+''.join(lines[node.end_lineno:])
conv=(H/'fragment_converter_v2.py.txt').read_text()
conv=conv.replace("gr_raw2=r2,gr_raw3=r3,gr_raw4=r4,gr_gaussian_mean=mg,","gr_raw2=r2,gr_raw3=r3,gr_raw4=r4,gr_norm_diagonal=fnp.sum(post22,axis=1),gr_gaussian_mean=mg,")
(H/'fragment_converter_v3.py.txt').write_text(conv)
birth_src=R/'astra_crunch_radius_birth_v1/fragment_k4l_l0_v1.py.txt'
birth=birth_src.read_text().replace('_math.','_rc_math.')
(H/'fragment_k4l_l0_integrated_v1.py.txt').write_text(birth)
npf4=(H/'fragment_np_f4_l0_v1.py.txt').read_text()
text+='\n\n'+conv+'\n'+birth+'\n'+npf4
ast.parse(text);out=H/'candidate_global_radius_v1.py';out.write_text(text)
receipt=dict(parent=str(P),parent_sha256=hashlib.sha256(P.read_bytes()).hexdigest(),candidate=str(out),
    candidate_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),core_edit_count=len(edits),core_anchors=edits,
    fragments={name:hashlib.sha256((H/name).read_bytes()).hexdigest() for name in ('fragment_converter_v3.py.txt','fragment_k4l_l0_integrated_v1.py.txt','fragment_np_f4_l0_v1.py.txt')},
    status='await fit review/tiny then fixed35,100 currentSDK pilot; no tunable coefficient')
(H/'BUILD_CANDIDATE_v1.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
