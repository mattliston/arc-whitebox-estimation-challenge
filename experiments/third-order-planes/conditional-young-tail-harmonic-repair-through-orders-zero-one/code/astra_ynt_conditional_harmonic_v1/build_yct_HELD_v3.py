from pathlib import Path
import ast,hashlib,json
D=Path(__file__).resolve().parent
P=D.parent/'astra_hermite_group_view_repair_v1/candidate_hermite_group_views_QR_HELD_v2.py'
H='1c913b92d6fd231593224d423dd0278be66390583bd68d6eb6c4b89eb1f79c9a'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(P)==H
s=P.read_text();tree=ast.parse(s);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_astra_fhm_core')
lines=s.splitlines(keepends=True);changed=[]
for i in range(node.lineno-1,node.end_lineno):
 if lines[i].strip()=='mu = wT @ m':
  lines[i]=lines[i].replace('mu = wT @ m','mu = wT @ m; _YCT_PRE_MEAN[0] = m')
  changed.append(i+1)
assert changed==[17313],changed
body=''.join(lines);fragment=D/'fragment_yct_HELD_v2.py';f=fragment.read_text()
out=D/'candidate_yct_ON_HELD_v2.py';outoff=D/'candidate_yct_OFF_HELD_v2.py'
for p,t in ((out,body+'\n'+f),(outoff,body+'\n'+f.replace('_YCT_ON = True','_YCT_ON = False',1))):
 compile(t,str(p),'exec')
 with p.open('x')as fp:fp.write(t)
assert body.replace('mu = wT @ m; _YCT_PRE_MEAN[0] = m','mu = wT @ m',1)==s
v=dict(parent=dict(path=str(P),sha256=H),candidate=dict(path=str(out),sha256=sha(out)),off=dict(path=str(outoff),sha256=sha(outoff)),fragment=dict(path=str(fragment),sha256=sha(fragment)),changed_parent_lines=changed,mean_lifetime='Exact previous mean published on the same active-core line as mu; tensor trace only reads it; outer predict clears in finally on success or exception.',fixed_scope='Full H0+H1 at existing YNT calls only, original rosters/bands/coefficients/trace preserved, no tuning.',models=0)
with(D/'SOURCE_MANIFEST_HELD_v2.json').open('x')as fp:json.dump(v,fp,indent=2);fp.write('\n')
print(json.dumps(v))
