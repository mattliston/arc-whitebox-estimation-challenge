import ast, hashlib, json
from pathlib import Path
D=Path(__file__).resolve().parent
P=D.parent/'opus_fix336687_v1/candidate_v61j.py'
s=P.read_text(); sha=hashlib.sha256(s.encode()).hexdigest();assert sha.startswith('ac93b10f'),sha
tree=ast.parse(s); fs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_astra_bcc_push_impl'];assert len(fs)==1
n=fs[0]; lines=s.splitlines(keepends=True); block=''.join(lines[n.lineno-1:n.end_lineno])
anchor="        Fj = _shared_old384_mm((_e2_append_inline((_e2_value_inline := Qn.T)) or _e2_value_inline), J)\n"
assert block.count(anchor)==1
patched=block.replace(anchor,anchor+"        if layer == 13:   # ASTRA ephemeral join-left repair, consumed by the existing read\n            _jlr_prepare(w32, previous, J, Qn, Fj, start, stop, weight)\n")
base=''.join(lines[:n.lineno-1])+patched+''.join(lines[n.end_lineno:])
fragment=(D/'fragment_join_left_repair_v1.py').read_text()
out=[]
for mode in ('exact256','residual64'):
    text=base+'\n\n'+fragment.replace("_JLR_MODE = 'exact256'",f"_JLR_MODE = {mode!r}",1)
    p=D/f'candidate_jlr_{mode}_HELD_v1.py';compile(text,str(p),'exec')
    with p.open('x') as f:f.write(text)
    out.append(dict(mode=mode,path=str(p),sha256=hashlib.sha256(text.encode()).hexdigest()))
res=dict(parent=str(P),parent_sha256=sha,candidates=out,status='HELD source only; no native model or release qualification',mutation='Only active _astra_bcc_push_impl inserts one helper after Fj, plus appended hook; source signatures unchanged')
(D/'BUILD_JLR_v1.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
