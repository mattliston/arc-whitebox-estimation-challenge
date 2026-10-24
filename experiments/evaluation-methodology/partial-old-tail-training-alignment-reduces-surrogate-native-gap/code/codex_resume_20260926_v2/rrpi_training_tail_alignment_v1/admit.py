from pathlib import Path
import hashlib,json
H=Path("/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2");J=H/'rrpi_training_join_alignment_v1';D=H/'rrpi_training_tail_alignment_v1'
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def main():
 p=json.loads((J/'protocol.json').read_text())
 p['kind']='target-free-old-vs-partial-old-tail-alignment'
 p['pins']=[r for r in p['pins'] if not r['path'].startswith(str(J))]
 p['pins'] += [pin(D/'tail_adapter.py'),pin(D/'probe.py'),pin(J/'result.json'),pin(J/'guard.json'),pin(J/'protocol.json')]
 for r in p['pins']:assert pin(r['path'])==r
 for row in p['data']:row['old']=pin(J/('old_'+str(row['rank'])+'.npy'))
 p['prospective_next']='Only after pooledgap<=.8andbothlower, samefixedS33warm2 8ranks/2updates withtailadapter andoriginaljoins; no variants. Native raw+adjustedN2 then decides endpoint.'
 p['known_partial_scope']='FullU,F tailestimate mirrors originalnonlazyblock; currentnative lazyblocktrace, latecommon256, youngYNT/deadcutrows andprecisionremainunmatched. Failedjoinpatchnotincluded.'
 with (D/'protocol.json').open('x') as f:json.dump(p,f,indent=2);f.write('\n')
 print(json.dumps(pin(D/'protocol.json')))
if __name__=='__main__':main()
