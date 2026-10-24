from pathlib import Path
import ast,hashlib,json
D=Path(__file__).resolve().parent
P=Path("/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_hermite_group_view_repair_v1/candidate_hermite_group_views_QR_HELD_v2.py")
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(P)=='1c913b92d6fd231593224d423dd0278be66390583bd68d6eb6c4b89eb1f79c9a'
F=D/'fragment_cct_HELD_v1.py';s=P.read_text();tree=ast.parse(s);lines=s.splitlines(keepends=True);edits=[]
for name in ('_flms_cov_star','_dc_cov_star'):
 node=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name][-1]
 old=''.join(lines[node.lineno-1:node.end_lineno]);new=old
 if name=='_flms_cov_star':
  a='    first = _mm(C32, w)'
  b='    _cct_input, _cct_aux = _cct_center(C32, w, hold); first = _mm(_cct_input, w)'
  assert new.count(a)==1;new=new.replace(a,b)
 else:
  a='        first = _mm(hold(C32[fnp.ix_(live, live)]), wa)'
  b='        _cct_input, _cct_aux = _cct_center(hold(C32[fnp.ix_(live, live)]), wa, hold); first = _mm(_cct_input, wa)'
  assert new.count(a)==2;new=new.replace(a,b)
  a='        first = _mm(hold(C32[fnp.ix_(roster, roster)]), wa)'
  b='        _cct_input, _cct_aux = _cct_center(hold(C32[fnp.ix_(roster, roster)]), wa, hold); first = _mm(_cct_input, wa)'
  assert new.count(a)==1;new=new.replace(a,b)
 a='    Cpre = cov.astype(fnp.float64)'
 b='    cov = _cct_restore(cov, _cct_aux, hold); Cpre = cov.astype(fnp.float64)'
 assert new.count(a)==1;new=new.replace(a,b)
 assert old.count('\n')==new.count('\n')
 edits.append((node.lineno-1,node.end_lineno,old,new))
for start,end,old,new in sorted(edits,reverse=True):lines[start:end]=[new]
core=''.join(lines);candidate=core+'\n\n'+F.read_text()
inverse=core
for _,_,old,new in edits:assert inverse.count(new)==1;inverse=inverse.replace(new,old)
assert inverse==s and core.count('\n')==s.count('\n')
C=D/'candidate_cct_HELD_v1.py';C.open('x').write(candidate)
M=dict(parent_path=str(P),parent_sha256=sha(P),candidate_path=str(C),candidate_sha256=sha(C),fragment_path=str(F),fragment_sha256=sha(F),source_inverse_byte_exact=True,line_count_preserved=True,changed_existing_functions=['_flms_cov_star','_dc_cov_star'],scope='Hidden1..14 only from existingW5scope; L0andL15unchanged.',covariance_dtype='float32',correction='cov+(R+R.T), R=(r@W-.5*mean(r)*sum(W,0))[:,None]*sum(W,0)[None,:]',original_dispatch_star_and_roster_unchanged=True,new_setup_keys=0,new_einsum_signatures=0)
(D/'SOURCE_MANIFEST_HELD_v1.json').open('x').write(json.dumps(M,indent=2)+'\n')
print(json.dumps(M))

