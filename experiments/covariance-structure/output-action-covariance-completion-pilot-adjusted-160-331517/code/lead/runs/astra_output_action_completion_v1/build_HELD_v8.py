from pathlib import Path
import ast,hashlib,json
D=Path(__file__).resolve().parent;S=D.parent/'astra_hermite_group_view_repair_v1/candidate_hermite_group_views_QR_HELD_v2.py'
s=S.read_text();assert hashlib.sha256(s.encode()).hexdigest()=='1c913b92d6fd231593224d423dd0278be66390583bd68d6eb6c4b89eb1f79c9a'
t=ast.parse(s);ls=s.splitlines(keepends=True);changes=[]
for node in t.body:
 if isinstance(node,ast.FunctionDef)and node.name in ('features','_astra_compute_left_v1','_astra_fhm_core'):
  old=''.join(ls[node.lineno-1:node.end_lineno]);new=old
  if node.name=='features':
   for item in ast.walk(node):
    if isinstance(item,ast.Assign) and any(isinstance(x,ast.Name)and x.id=='vals'for x in item.targets):
     for j in (13,14,15,26):
      orig=item.value.elts[j];rep=ast.parse(f"_OAC_STATE['features'][{j}] if _OAC_STATE.get('Cpre') is aux['pre_cov'] else 0",mode='eval').body;rep.orelse=orig;item.value.elts[j]=rep
   vals=next(x for x in ast.walk(node) if isinstance(x,ast.Assign) and any(isinstance(y,ast.Name)and y.id=='vals'for y in x.targets))
   lines=old.splitlines(keepends=True)
   for i,line in enumerate(lines):
    if line.startswith('    vals = '):lines[i]='    '+ast.unparse(vals)+'\n'
   new=''.join(lines).replace('return fnp.stack(vals, axis=1, dtype=fnp.float32)','return _oac_features(l, aux, vals)')
  elif node.name=='_astra_compute_left_v1':
   assert old.count('hold(cov @ mu)')==1;new=old.replace('hold(cov @ mu)','hold(_oac_cmu(cov, mu))')
  else:
   needle='_dc_cov_star(w32, C, star, response_gate, gate_floor, retain32=True)';assert old.count(needle)==1;new=old.replace(needle,'_oac_cov_star(w32, C, star, response_gate, gate_floor, mu, m, retain32=True, layer=l)')
  changes.append((node.lineno-1,node.end_lineno,new))
assert len(changes)==3
for a,b,new in sorted(changes,reverse=True):ls[a:b]=[new]
out=''.join(ls)+'\n'+(D/'fragment_output_action_HELD_v8.py').read_text();ast.parse(out)
p=D/'candidate_output_action_HELD_v8.py';p.open('x').write(out)
manifest=dict(parent=str(S),parent_sha256=hashlib.sha256(s.encode()).hexdigest(),candidate=str(p),candidate_sha256=hashlib.sha256(out.encode()).hexdigest(),threshold=.05,rank=3,changed_functions=['features','_astra_compute_left_v1','_astra_fhm_core'],models=0)
(D/'SOURCE_v8.json').open('x').write(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))
