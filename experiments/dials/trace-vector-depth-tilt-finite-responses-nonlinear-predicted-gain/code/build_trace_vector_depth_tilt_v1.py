from pathlib import Path
import ast,hashlib,json
HERE=Path(__file__).resolve().parent
parent=HERE/'candidate_old_both_tail_trace_bounded_v6.py'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(parent)=='5d43930820d144db00aac86ace9281fe0232240e321f53701cfdd8fe1ec98f26'
source=parent.read_text()
tree=ast.parse(source)
bs=[ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
    and any(isinstance(t,ast.Name) and t.id=='_S19_BSCALE' for t in n.targets)]
assert bs==[1.0383394097164294]
old='                b = b * _S19_BSCALE'
new='                b = b * _BTD_SCALE[l]'
assert source.count(old)==1
q=tuple((l-7)/7 for l in range(15))
assert abs(sum(q))<1e-15
records=[]
for name,theta in [('plus',.02),('minus',-.02)]:
    scales=tuple(bs[0]*(1+theta*x) for x in q)
    tail='\n\n# Fixed trace-vector depth contrast; only metadata constants.\n_BTD_SCALE = '+repr(scales)+'\n'
    body=source.replace(old,new)+tail
    assert body[:-len(tail)].replace(new,old)==source
    ast.parse(body)
    out=HERE/f'candidate_trace_vector_depth_tilt_{name}_v1.py'
    with out.open('x') as f:f.write(body)
    records.append(dict(arm=name,theta=theta,path=str(out),sha256=sha(out),scales=scales))
record=dict(parent_sha256=sha(parent),parent_bscale=bs[0],profile=q,
    arms=records,whole_source_inverse_exact=True,scope='Two fixed TRAIN2 information probes; no held-out data or new fitted source')
with (HERE/'trace_vector_depth_tilt_manifest_v1.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record))
