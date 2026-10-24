"""Build a separately sealed diagonal-reduction candidate."""
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent;H=HERE.parent
parent=H/'candidate_old_both_tail_trace_bounded_v6.py'
src=parent.read_text();assert hashlib.sha256(parent.read_bytes()).hexdigest()=='5d43930820d144db00aac86ace9281fe0232240e321f53701cfdd8fe1ec98f26'
lines=src.splitlines(keepends=True);tree=ast.parse(src)
nodes={name:[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name][-1]
       for name in ['_predict_binary','_flms_cov_star']}
for name,node in sorted(nodes.items(),key=lambda x:x[1].lineno,reverse=True):
 body=''.join(lines[node.lineno-1:node.end_lineno])
 if name=='_predict_binary':
  old='''                Cpre = _e2_H(_resident_gram512(_e2_H(w32.T), w32)).astype(fnp.float64)
                Bpre = Cpre'''
  new='''                Cpre = _e2_H(_resident_gram512(_e2_H(w32.T), w32)).astype(fnp.float64)
                # Exact input variance reduction; all Gram off-diagonals retained.
                fnp.fill_diagonal(Cpre, _e2_H(fnp.sum(_e2_H(w * w), axis=0)))
                Bpre = Cpre'''
 else:
  old='''    cov = _sym_upper(wt, first)
    Cpre = cov.astype(fnp.float64)'''
  new='''    cov = _sym_upper(wt, first)
    Cpre = cov.astype(fnp.float64)
    # Reuse the paid first product; remove only outer-diagonal FP32 rounding.
    _v_w64 = w.astype(fnp.float64)
    _v_first64 = first.astype(fnp.float64)
    _v_products = _v_w64 * _v_first64
    _v_diagonal = fnp.sum(_v_products, axis=0)
    fnp.fill_diagonal(Cpre, _v_diagonal)
    flops.free([_v_w64, _v_first64, _v_products, _v_diagonal])
    del _v_w64, _v_first64, _v_products, _v_diagonal'''
 assert body.count(old)==1,(name,body.count(old))
 lines[node.lineno-1:node.end_lineno]=[body.replace(old,new)]
candidate=HERE/'candidate.py';newsrc=''.join(lines);newtree=ast.parse(newsrc)
changed=[getattr(a,'name',type(a).__name__) for a,b in zip(tree.body,newtree.body) if ast.dump(a)!=ast.dump(b)]
assert len(tree.body)==len(newtree.body) and changed==['_predict_binary','_flms_cov_star']
with candidate.open('x') as f:f.write(newsrc)
files=[HERE/n for n in ['PROTOCOL.md','build.py','unit.py','assess.py','candidate.py']]
files += [parent,H/'native_eval.py',H/'both_tail_train2_v6/rows.jsonl',H/'both_tail_train6_v6_routing_control/rows.jsonl']
seal=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),candidate_evaluated=False,changed_functions=changed,
 candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
with (HERE/'seal.json').open('x') as f:json.dump(seal,f,indent=2);f.write('\n')
print(json.dumps(dict(candidate_sha256=seal['candidate_sha256'],changed_functions=changed)))
