from pathlib import Path
import ast,json,hashlib
D=Path(__file__).resolve().parent;P=D.parent/'astra_rtt2_prefix_power_source_v1/candidate_prefix_power_QR_ON_HELD_v3.py';b=P.read_bytes();s=b.decode()
assert hashlib.sha256(b).hexdigest()=='4d7739a5ad4444be8c20bdd5c5bc86f1cee5536a2c21e956d86dd95e6c83d8d9'
n=[n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)and n.name=='_rtt_read'][-1];lines=s.splitlines(keepends=True);a=sum(map(len,lines[:n.lineno-1]));z=sum(map(len,lines[:n.end_lineno]));body=s[a:z]
old='floor=.5*_rc_math.erfc(1.5/_rc_math.sqrt(2.0))';new='floor=.5*_rc_math.erfc(1.0/_rc_math.sqrt(2.0))';assert body.count(old)==1
body2=body.replace(old,new);out=s[:a]+body2+s[z:];assert out[:a]+out[a:a+len(body2)].replace(new,old)+out[a+len(body2):]==s
p=D/'candidate_QR_prefix_z1_ON_HELD_v1.py';p.write_text(out);ast.parse(out)
r=dict(status='FROZEN_SOURCE_ONLY_SINGLE_THRESHOLD',candidate=str(p),candidate_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),parent=str(P),parent_sha256=hashlib.sha256(b).hexdigest(),source_edit='Only active final _rtt_read prefix cutoff1.5 to1.0. Existing variables retain c15 names as prefix-count fields.',exact_inverse_edit_recovers_parent=True,original_c2_mask_unchanged=True,QR_refinement_unchanged=True,one_old_inner_unchanged=True,old_tail_exact=True,all_existing_context_guards_unchanged=True,no_models=True,no_parameter_sweep=True)
q=D/'SOURCE_READY_HELD_v1.json';q.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r));print('ready',hashlib.sha256(q.read_bytes()).hexdigest())
