"""One fixed analytic young-trace proposal; stdlib source preparation only."""
from pathlib import Path
import ast, copy, difflib, hashlib, json

D = Path(__file__).resolve().parent
H = D.parent
P = H/'hill_climb_e1_ynt_psdb_d5_v1'
T = H/'hill_climb_d5_ynt275_v1'


def pin(p):
    p=Path(p).resolve(); raw=p.read_bytes()
    return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def once(p, x):
    with p.open('x') as f: json.dump(x,f,indent=2,allow_nan=False);f.write('\n')


def main():
    parent=pin(P/'candidate.py');assert parent['sha256']=='42e50cdc80f49a562428f018afb3c20ae4bf42f4b48b1f63025bb384decd08eb'
    base=(P/'candidate.py').read_text(); lines=base.splitlines(True)
    nodes=[n for n in ast.parse(base).body if isinstance(n,ast.FunctionDef) and n.name=='_ynt_trace'];assert len(nodes)==1
    old=''.join(lines[nodes[0].lineno-1:nodes[0].end_lineno]);new=(D/'fragment.py').read_text()
    capture='_E2_LAYER.update(mu=mu, Cpre=Cpre, vc=_e2_vc, s=_e2_s, a=_e2_a, g=response_gate)'
    cleanup="        _YNT_STATS['retired'] = True\n"
    edits=[(old,new),(capture,capture[:-1]+', ynt_previous_mean=m)'),
        (cleanup,cleanup+"        _E2_LAYER.pop('ynt_previous_mean', None)\n")]
    child=base
    for before,after in edits:
        assert child.count(before)==1 and after not in child
        child=child.replace(before,after,1)
    inverse=child
    for before,after in reversed(edits):
        assert inverse.count(after)==1;inverse=inverse.replace(after,before,1)
    assert inverse==base;ast.parse(child)
    assert "negative = (ctx['a'] < -2.5) & (~ctx['live'])" in child
    assert "score = fnp.where(negative, ctx['a'] + 2.5, 0.0)" in child
    with (D/'candidate.py').open('x') as f:f.write(child)
    with (D/'source_diff.patch').open('x') as f:f.write(''.join(difflib.unified_diff(base.splitlines(True),child.splitlines(True),fromfile='S34/candidate.py',tofile=D.name+'/candidate.py')))
    candidate=pin(D/'candidate.py')
    protocol=dict(root_selected=True,proposal='Young-only spherical zonal degree3 correction, analytic coefficient1; no fit or threshold changes.',
        parent=parent,current_bank=pin(H/'official_bank_S34_v1.json'),candidate=candidate,
        formula='u=m_previous/||m_previous||; z=w.u; A=sum(lambda*(p.u)^3); B=u.tau; add (((n+2)*A-3*B)/(n-1))*(z^3-3*||w||^2*z/(n+2)) to existing 3*||w||^2*(w.tau)/(n+2).',
        interpretation='Conditional mean of the missing cubic spherical harmonic under rotations around the previous-mean axis; not exact prediction of an individual cubic or of a trained network.',
        fixed_coefficient=1,fixed_threshold=2.5,layers=list(range(7,15)),parity_layers=list(range(7)),
        prior_failures='Total-CP Gaussian conditional-mean repair8697a46e failed; inherited-young-only and YNT2.75 are closed. This changes only young spherical diagonal reconstruction; no prior failure is relabeled.',
        radial_exclusion='YNT tail is outside actual alive; both omitted=alive&~response and beta*=alive vanish there, so a direct omitted-radial repair is identically zero.',
        capture='Exact previous-layer mean captured in the unique E2 update before nonlinear m update; removed in YNT prediction finally. Projection uses wt64.T@u, not differently rounded preactivation mu.',
        primitive='Root-only small signed-atom/full-ambient, spherical quadrature, pure-mean, pure-trace and cleanup fixture; no estimator import.',
        fixed_train2=[35,100],gates=dict(adjusted_pct_at_most=-.1,wins_min=1,max_per_network_regression_pct=1),
        native_limits=dict(seconds=150,rss_bytes=7500000000,added_swap_bytes=0),
        added_work='About2*(n*births+len(previous)*q+n*tail)+vector work; existing B/P/wt/tau reused. Added RPCs must be measured; no gain forecast.',
        prospective_scope='native2_only',no_neighbor_sweep=True,no_result_based_retuning=True,later_authority=False,bank_authority=False)
    once(D/'protocol.json',protocol)
    once(D/'build.json',dict(parent=parent,candidate=candidate,builder=pin(__file__),fragment=pin(D/'fragment.py'),
        primitive=pin(D/'probe.py'),protocol=pin(D/'protocol.json'),diff=pin(D/'source_diff.patch'),
        source_inverse_exact=True,edits=[dict(before=a,after=b) for a,b in edits],
        wrapper_captures_unchanged=True,setup_counts=dict(bound=682,lvfuse=649,psdb_fused=9),
        numerical_execution=False))
    nc=copy.deepcopy(json.loads((T/'native_screen.json').read_text()))
    assert nc['arms']['base']['source']==parent and nc['arms']['base']==nc['arms']['parity_parent']
    nc['arms']['candidate']['source']=candidate
    for size,b in nc['arms']['candidate']['blocks'].items():
        b['tag']=D.name+'_train'+size+'_v1';b['guard']=str(D/('native'+size+'_guard.json'))
        assert 'artifacts' not in b
    for size,stage in nc['stages'].items():
        stage['result']=str(D/('native'+size+'_assessment.json'));stage['assessment_guard']=str(D/('native'+size+'_assess_guard.json'))
    req=nc['boundary_before8'];req['receipt']=str(D/'boundary_not_admitted.json');req['assessment_guard']=str(D/'boundary_guard_not_admitted.json')
    req['assessment_command'][3]=str(D/'boundary_config_not_admitted.json');req['assessment_command'][4]='0'*64
    nc['provenance']=[pin(__file__),pin(D/'protocol.json'),pin(D/'build.json'),pin(D/'fragment.py'),pin(D/'probe.py'),
        pin(D/'source_diff.patch'),parent,pin(T/'native_screen.json'),pin(P/'native_screen.json'),
        pin(P/'native8_official_limit_extension.json'),pin(P/'native8_official_limit_extension_guard.json'),pin(H/'official_bank_S34_v1.json')]
    nc['interpretation']='One fixed analytic zonal young-trace N2 pilot versus officialS34. Same cutoff2.5, old paths, learned values and fixed gates. L0-6 exact. No boundary/additional6/broad/bank authority from this config.'
    nc['prospective_scope']='native2_only'
    once(D/'native_screen.json',nc)
    print(json.dumps(dict(candidate=candidate,config=pin(D/'native_screen.json'),build=pin(D/'build.json'),probe=pin(D/'probe.py'),protocol=pin(D/'protocol.json'))))


if __name__=='__main__':main()
