"""Reuse the star-plane contraction for its diagonal after the s1 term is gone.

For P=W.T@Q and T=(W.T**2)@B.T,
  diag_old[i] = sum_j ((P@B)[i,j] * W[j,i]**2)
              = sum_a P[i,a] * T[i,a].
T is already needed by s2=T@P.T. Avoid forming the unused P@B matrix.
This changes floating-point association but not the mathematical estimator.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
base=Path(sys.argv[1])
tag=sys.argv[2]
src=base.read_text()
node=[n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='_flms_cov_star'][-1]
fn='\n'.join(src.splitlines()[node.lineno-1:node.end_lineno])+'\n'
assert 'sub = s2 / 3' in fn and 's1 =' not in fn
old='    pushed = P1 @ B\n    pw = pushed * wt\n'
assert fn.count(old)==1
fn=fn.replace(old,'')
old='    s2 = (wt2 @ B.T) @ P1.T'
assert fn.count(old)==1
fn=fn.replace(old,'    T2 = wt2 @ B.T\n    s2 = T2 @ P1.T')
old='    diagonal = fnp.sum(pw * wt, axis=1, dtype=fnp.float64)'
assert fn.count(old)==1
fn=fn.replace(old,'    diagonal = fnp.sum(P1 * T2, axis=1, dtype=fnp.float64)')
fragment='\n\n# Reuse the s2 inner product for the star diagonal, 2026-09-26.\n'+fn
out=src+fragment
ast.parse(out)
for path,value in [(HERE/(tag+'.py'),out),(HERE/(tag+'_fragment.py'),fragment)]:
    with path.open('x') as f:f.write(value)
manifest=dict(base=str(base),base_sha256=hashlib.sha256(src.encode()).hexdigest(),candidate_sha256=hashlib.sha256(out.encode()).hexdigest(),
    identity='sum_j (P B)_ij (W.T)^2_ij = sum_a P_ia ((W.T)^2 B.T)_ia',scope='exact algebra; floating-point reassociation')
with (HERE/(tag+'_manifest.json')).open('x') as f:json.dump(manifest,f,indent=2)
print(json.dumps(manifest))
