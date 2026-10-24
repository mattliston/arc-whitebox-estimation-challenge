"""Source-only projected-diagonal core congruence, fixed joins7/10 and rank."""
from pathlib import Path
import ast,hashlib,json
D=Path(__file__).resolve().parent
H=D.parent
P=H/'hill_climb_pair_rr_inverse_v1/candidate.py'
PARENT_SHA='a3efb3a1066506e5bc2bc5a5da668056277ba65ea6cdd88c1de3be636fd419f3'
SUFFIX='''

# Fixed diagonal congruence of the existing approximate core; not a PSD repair.
_CDM_ENABLED = True


def _cdm_join_diag(Fj, weight):
    hold = _E2Hold()
    F64 = Fj.astype(fnp.float64)
    return hold(F64 * F64) @ hold(weight.astype(fnp.float64))


def _cdm_old_diag(overlap, old_core):
    hold = _E2Hold()
    O64 = overlap.astype(fnp.float64)
    projected = O64 @ hold(old_core.astype(fnp.float64))
    return fnp.sum(hold(projected * O64), axis=1)


def _cdm_scale(core, target):
    hold = _E2Hold()
    C64 = core.astype(fnp.float64)
    current = fnp.diag(C64)
    if target.shape != current.shape or str(target.dtype) != 'float64':
        raise RuntimeError('Core diagonal matching shape or precision')
    if (not bool(fnp.all(hold(fnp.isfinite(target)))) or not bool(fnp.all(hold(fnp.isfinite(current))))
            or not bool(fnp.all(hold(target > 0.0))) or not bool(fnp.all(hold(current > 0.0)))):
        raise RuntimeError('Core diagonal matching requires positive finite diagonals')
    scale = fnp.sqrt(hold(target / current))
    scaled = hold(C64 * hold(scale[:, None])) * hold(scale[None, :])
    symmetric = hold(scaled + hold(scaled.T)) * 0.5
    return symmetric.astype(fnp.float32)
'''


def pin(p):
    p=Path(p).resolve();b=p.read_bytes()
    return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))


def source():
    assert pin(P)['sha256']==PARENT_SHA
    base=P.read_text();tree=ast.parse(base)
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_shared_old384_push']
    node=next(n for n in reversed(nodes) if '_need_core = layer < max(_JOIN_SCHEDULE)' in ast.get_source_segment(base,n))
    before=ast.get_source_segment(base,node);assert base.count(before)==1
    local=[
        ("        Fj = _shared_old384_mm(_e2_H(Qn.T), J)\n        joined_core = None",
         "        Fj = _shared_old384_mm(_e2_H(Qn.T), J)\n"
         "        if _CDM_ENABLED and _need_core:\n"
         "            _cdm_target = _cdm_join_diag(Fj, weight)\n"
         "        joined_core = None"),
        ("            overlap = _shared_old384_mm(_e2_H(Qn.T), Qp)\n            factors = _slr_join_factors(layer, overlap, state, Fj)",
         "            overlap = _shared_old384_mm(_e2_H(Qn.T), Qp)\n"
         "            if _CDM_ENABLED and _need_core:\n"
         "                _cdm_target = _cdm_target + _cdm_old_diag(overlap, state['core'])\n"
         "            factors = _slr_join_factors(layer, overlap, state, Fj)"),
        ("        old_columns = stop\n        state['columns'] = old_columns",
         "        if _CDM_ENABLED and _need_core:\n"
         "            core = _cdm_scale(core, _cdm_target)\n"
         "            del _cdm_target\n"
         "        old_columns = stop\n        state['columns'] = old_columns")]
    after=before
    for old,new in local:
        assert after.count(old)==1;after=after.replace(old,new,1)
    inverse=after
    for old,new in reversed(local):
        assert inverse.count(new)==1;inverse=inverse.replace(new,old,1)
    assert inverse==before
    edits=[dict(before=before,after=after)]
    body=base.replace(before,after,1);candidate=body+SUFFIX
    assert body.replace(after,before,1)==base and candidate.count('_CDM_ENABLED = True')==1
    identity=candidate.replace('_CDM_ENABLED = True','_CDM_ENABLED = False',1)
    compile(candidate,str(D/'candidate.py'),'exec');compile(identity,str(D/'candidate_identity.py'),'exec')
    return candidate,identity,edits


def main():
    candidate,identity,edits=source()
    prior=pin(H/'hill_climb_combo_exact_core7_v1/closeout.json');r=json.loads(Path(prior['path']).read_text())
    assert r['status']=='closed_native2_score_failure'
    for name,value in [('candidate.py',candidate),('candidate_identity.py',identity)]:
        with (D/name).open('x') as f:f.write(value)
    record=dict(parent=pin(P),candidate=pin(D/'candidate.py'),identity=pin(D/'candidate_identity.py'),build=pin(__file__),
        edits=edits,suffix=SUFFIX,whole_source_inverse_exact=True,compiled=True,enabled=True,
        core_joins=[7,10],ranks=[384,320,384],joins=[7,10,13],common_rank=256,first_changed_prediction_layer=10,
        formula='target=rowsum(Fj^2*weight)+rowsum((overlap@old_core)*overlap); d=sqrt(target/diag(approx_core)); newcore=sym(diag(d)@approx_core@diag(d)).float32',
        semantics='Congruence preserves PSD only if the input approximate core is PSD in real arithmetic. It is not a PSD repair, exact full Gram, or claim of native PSD after rounding.',
        precision='FP64 target reductions, existing-overlap contraction, ratios and congruence; final FP64 symmetry average thenFP32.',
        strict_positive_finite_diagonals=True,fallback=False,fitting_or_clipping=False,
        unchanged='Current joining bank, lambda weights, row selection, Y, QR, Qn, Fj, overlap, F/factors, current ranks and all other source. Only stored core after joins7/10 changes; future joins can then differ.',
        identity_original_array_operations_exact_by_source=True,
        cost='One FP64 320x384 by384x384 product atL10, reused existing overlap; O(rank*cohortwidth) weighted square reductions atL7/L10; O(rank^2) scaling. Actual native bill/resources required.',
        prior_exact_core_failure=prior,prior_exact_core7_parent_note='Prior2212448 preS35 retained READ7/packed-pair composition, not currentRRPI.',
        prior_metrics={k:r[k] for k in ('raw_pct','adjusted_pct','bill_pct','wins')},
        source_only=True,no_model_or_array_jobs=True)
    with (D/'source_delta.json').open('x') as f:json.dump(record,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:record[k] for k in ('candidate','identity','build')}))


if __name__=='__main__':main()
