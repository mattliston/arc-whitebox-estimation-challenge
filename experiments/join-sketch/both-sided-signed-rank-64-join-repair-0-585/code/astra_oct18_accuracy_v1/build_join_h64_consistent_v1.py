import ast, hashlib, json
from pathlib import Path
D=Path(__file__).resolve().parent
P=D.parent/'opus_fix336687_v1/candidate_v61j.py'
s=P.read_text();sha=hashlib.sha256(s.encode()).hexdigest();assert sha.startswith('ac93b10f')
tree=ast.parse(s);nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_astra_bcc_push_impl'];assert len(nodes)==1
n=nodes[0];lines=s.splitlines(keepends=True);block=''.join(lines[n.lineno-1:n.end_lineno])
a="        Fj = _shared_old384_mm((_e2_append_inline((_e2_value_inline := Qn.T)) or _e2_value_inline), J)\n"
b="    answer = _astra_bcc_join(old, young, state)\n"
assert block.count(a)==block.count(b)==1
patched=block.replace(a,a+"        if layer == 13:\n            _jhr_prepare(w32, previous, J, Qn, Fj, start, stop, weight, state)\n")
patched=patched.replace(b,b+"    if layer == 13:\n        answer = _jhr_apply(answer)\n")
t=''.join(lines[:n.lineno-1])+patched+''.join(lines[n.end_lineno:])
pa,ca=ast.parse(s),ast.parse(t)
changed=[x.name for x,y in zip(pa.body,ca.body) if ast.dump(x)!=ast.dump(y)]
assert changed==['_astra_bcc_push_impl']
t+='\n\n'+(D/'fragment_join_h64_consistent_v1.py').read_text()
out=D/'candidate_jhr64_HELD_v1.py';assert not out.exists();compile(t,str(out),'exec');out.write_text(t)
meta=dict(parent=str(P),parent_sha256=sha,candidate=str(out),sha256=hashlib.sha256(t.encode()).hexdigest(),status='HELD source only, no tiny/native qualification',current_layer='Both CP sides restored; corrected pp feeds NP/K3 and other existing consumers. Existing omitted-tail/deadcut rows unchanged.',future_bank='Q/F/core unmodified; corrected current physical moments and NP coefficient update may affect future recursion; no residual factor is carried.',products='8 extra products including second projection, one QR64; halfwidth signed h coefficient; no future rank growth')
(D/'BUILD_JHR64_v1.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
