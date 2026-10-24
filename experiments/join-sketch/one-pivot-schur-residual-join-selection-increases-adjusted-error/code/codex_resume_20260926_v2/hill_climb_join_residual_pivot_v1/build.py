"""Build one fixed residual-pivot old-join selector; stdlib/source only."""
from pathlib import Path
import ast, difflib, hashlib, json
D=Path(__file__).resolve().parent; H=D.parent
BASE=H/'hill_climb_pair_rr_inverse_v1/candidate.py'
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def once(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
assert pin(BASE)['sha256']=='a3efb3a1066506e5bc2bc5a5da668056277ba65ea6cdd88c1de3be636fd419f3'
s=BASE.read_text(); tree=ast.parse(s)
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_shared_old384_push' and 'K = Y[_jn_rows' in ast.get_source_segment(s,n)]
node=nodes[-1]; before=ast.get_source_segment(s,node)
anchor="            _jn_rows = fnp.sort(_e2_H(fnp.argsort(_e2_H(-_e2_H(_jn_d.astype(fnp.float64))), kind='stable'))[:rank])"
assert before.count(anchor)==1
replacement="""            # One Schur-complement pivot: retain the largest diagonal row,
            # then choose the remaining rows by unexplained diagonal energy.
            _jn_d64 = _jn_d.astype(fnp.float64)
            _jn_pivot = fnp.argsort(-_jn_d64, kind='stable')[:1]
            _jn_col = fnp.matmul(J, weight[:, None] * J[_jn_pivot, :].T)
            if old_columns:
                _jn_col = _jn_col + fnp.matmul(Qp, fnp.matmul(state['core'], Qp[_jn_pivot, :].T))
            _jn_col64 = _jn_col.astype(fnp.float64)
            _jn_dp = _jn_d64[_jn_pivot]
            _jn_score = _jn_d64 - fnp.where(_jn_dp > 0,
                fnp.sum(_jn_col64 * _jn_col64, axis=1) / fnp.maximum(_jn_dp, 1e-30), 0.0)
            fnp.put(_jn_score, _jn_pivot, -float('inf'))
            _jn_rest = fnp.argsort(-_jn_score, kind='stable')[:rank - 1]
            _jn_rows = fnp.sort(fnp.concatenate([_jn_pivot, _jn_rest]))
            del _jn_d64, _jn_pivot, _jn_col, _jn_col64, _jn_dp, _jn_score, _jn_rest"""
after=before.replace(anchor,replacement,1)
candidate=s.replace(before,after,1)
assert candidate.replace(after,before,1)==s
ct=ast.parse(candidate);compile(candidate,str(D/'candidate.py'),'exec')
assert len(tree.body)==len(ct.body)
changed=[(i,a.name) for i,(a,b) in enumerate(zip(tree.body,ct.body)) if ast.dump(a)!=ast.dump(b)]
assert len(changed)==1 and changed[0][1]=='_shared_old384_push'
with (D/'candidate.py').open('x') as f:f.write(candidate)
with (D/'source.diff').open('x') as f:f.write(''.join(difflib.unified_diff(s.splitlines(True),candidate.splitlines(True),fromfile=str(BASE),tofile=str(D/'candidate.py'))))
once(D/'source_inverse.json',dict(parent=pin(BASE),candidate=pin(D/'candidate.py'),before=before,after=after,exact_inverse=True,changed_top_level_nodes=changed))
once(D/'protocol.json',dict(schema='old-join-one-residual-pivot-v1',root_selected=True,parent=pin(BASE),candidate=pin(D/'candidate.py'),
 intervention='Only row selection in active old-join push, all3 joins7/10/13, same ranks/cohorts/full Y/QR/F/core/projection and downstream learned parameters.',
 formula='G=J diag(lam^2) J.T + Q core Q.T when old exists. p=stableargmax diagG. g=G[:,p] by one thin column. score_i=d_i-g_i^2/d_p ifd_p>0, else d_i. Exclude p; choose largest rank-1 scores plusp, then physically sort indices.',
 rationale='Reduce redundant selected rows after removing one dominant Gram direction, at fixed retained rank. This is a new empirical hypothesis, with no measured gain or theorem of endpoint improvement.',
 arithmetic='Extra column FP32 ordinary thin matmul; residual score FP64; denominator clamped1e-30 with nonpositive fallback. Approximate rounded old core need not bePSD, and separatelycomputed d_p may differ from g_p. No clipping of negative residualscores.',
 ownership='All runtime arithmetic/indexing uses public meteredFNP. No host array data extraction, persistent state, cache, setup mutation, or new arena. Temporary selectors retired before ordinary Y construction.',
 forecast=dict(extra_flops_saved_geometry=30600000,network_reference_flops=263000000000,estimated_bill_pct=100*30600000/263000000000,conservative_bill_pct=0.02,not_measured=True,actual_billing_required=True),
 dedup='Bounded oldjoin PT40/current_nystrom_origin/selector review found no residual-pivot equivalent. Value-weightedL10 ranking and energy-order-only failures preserved; closed star-greedy selection is different carrier.',
 parity_layers=list(range(7)),native2_only=True,gates='raw<=-.1%,adjusted<=-.1%,>=1adjustedwin,maxadjustedregression<=1%; new ownboundary/native6 needed before retention. Bank>=1% only after fullqualification.',
 no_parameter_fit=True,no_pivot_layer_threshold_sweep=True,model_executed=False))
once(D/'build.json',dict(builder=pin(__file__),parent=pin(BASE),candidate=pin(D/'candidate.py'),protocol=pin(D/'protocol.json'),source_inverse=pin(D/'source_inverse.json'),source_diff=pin(D/'source.diff'),source_inverse_exact=True,changed_top_level_nodes=changed,model_executed=False))
print(json.dumps({k:pin(D/k) for k in ('candidate.py','protocol.json','build.json','source_inverse.json')}))
