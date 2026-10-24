"""Create one prospective rank224 pilot from fully qualified pending D5."""
from pathlib import Path
import ast, copy, hashlib, json
D=Path(__file__).resolve().parent
H=D.parent
P=H/'hill_climb_e1_ynt_psdb_d5_v1'
W=H/'s33_two_update_warm_proposal_v1'
def pin(p):
    p=Path(p).resolve(); b=p.read_bytes()
    return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def once(p,x):
    with p.open('x') as f: json.dump(x,f,indent=2,allow_nan=False); f.write('\n')
def main():
    parent=pin(P/'candidate.py')
    assert parent['sha256']=='42e50cdc80f49a562428f018afb3c20ae4bf42f4b48b1f63025bb384decd08eb'
    base=(P/'candidate.py').read_text()
    assert '_LCR_LAYERS = (10, 11, 12, 13, 14)' in base
    assert 'ru = _LCR_RU if _UVP_CUR[0] in _LCR_LAYERS else 256' in base
    suffix='\n# Fixed aligned late-only common-rank pilot.\n_LCR_RU = 224\n'
    child=base+suffix; ast.parse(child)
    assert child[:-len(suffix)]==base
    with (D/'candidate.py').open('x') as f: f.write(child)
    candidate=pin(D/'candidate.py')
    protocol=dict(root_admitted=True,proposal='Single untested aligned late common rank224, layers10-14; terminal15 stays256.',
        basis=pin(H/'teammate_late_common_neighbors_v1.md'),parent=parent,candidate=candidate,
        parent_status='Fully local-qualified submission333050, official grade pending; research comparator only until grade.',
        comparator=pin(P/'native8_official_limit_extension.json'),qualification=pin(P/'FIRE_CHECKS_v1.json'),
        fixed_train2=[35,100],fixed_additional6=[36,45,52,58,64,70],
        stage2_gates=dict(adjusted_pct_at_most=-.1,wins_min=1,max_per_network_regression_pct=1),
        parity_layers=list(range(10)),cheap_pilot_only=True,no_neighbor_sweep=True,no_result_based_retuning=True,
        later_work='Only if pilot passes: own boundary under documented official0.4 residual limit, unchanged105s/4.1s/7.5GB/zeroSwap; fixed8 consistency. No broad or bank without >=1% combined gain and full serving.',
        approximate_bill_forecast_pct=-.42,forecast_is_not_measured=True,future_only=True,reused_training_not_pristine=True)
    once(D/'protocol.json',protocol)
    once(D/'build.json',dict(parent=parent,candidate=candidate,builder=pin(__file__),protocol=pin(D/'protocol.json'),exact_parent_prefix=True,sole_override='_LCR_RU = 224',lines_appended=suffix.splitlines()))
    original=json.loads((P/'native_screen.json').read_text())
    prior=json.loads((P/'native8_official_limit_extension.json').read_text())
    assert prior['passed_integrity'] and prior['admitted'] and prior['candidate']==parent
    bc=copy.deepcopy(original['arms']['candidate'])
    for b in prior['blocks']: bc['blocks'][str(b['size'])]['artifacts']=b['arms']['candidate']
    nc=copy.deepcopy(json.loads((W/'native_screen.json').read_text()))
    nc['arms']=dict(candidate=copy.deepcopy(bc),base=copy.deepcopy(bc),parity_parent=copy.deepcopy(bc))
    nc['arms']['candidate']['source']=candidate
    for key,b in nc['arms']['candidate']['blocks'].items():
        b.pop('artifacts',None); b['tag']=D.name+'_train'+key+'_v1'; b['guard']=str(D/('native'+key+'_guard.json'))
    nc['parity_layers']=list(range(10))
    for key,b in nc['stages'].items():
        b['result']=str(D/('native'+key+'_assessment.json'));b['assessment_guard']=str(D/('native'+key+'_assess_guard.json'))
    assert nc['stages']['2']['gates']==dict(adjusted_pct_at_most=-.1,strict_adjusted=False,wins_min=1,ci_upper_below=None,max_per_network_regression_pct=1,additional6_gain_required=False)
    # The unexecuted later-stage requirement stays fail-closed and is not a passed parent boundary.
    req=nc['boundary_before8'];req['receipt']=str(D/'boundary_assessment.json');req['assessment_guard']=str(D/'boundary_assess_guard.json')
    req['assessment_command'][3]=str(D/'boundary_config_not_yet_admitted.json')
    nc['provenance']=[pin(__file__),pin(D/'protocol.json'),pin(D/'build.json'),pin(H/'teammate_late_common_neighbors_v1.md'),parent,
        pin(P/'native_screen.json'),pin(P/'native8_official_limit_extension.json'),pin(P/'native8_official_limit_extension_guard.json'),
        pin(P/'FIRE_CHECKS_v1.json'),pin(P/'submission_receipt_v1.json')]
    nc['interpretation']='Prospective one-candidate N2 pilot against pending locally-qualified D5; no future model run admitted by this configuration. Original engine and fixed cheap gates; exactL0-9 parity. Later resource/score qualification is mandatory and requires its own admission.'
    nc['prospective_scope']='native2_only'
    once(D/'native_screen.json',nc)
    print(json.dumps(dict(candidate=candidate,config=pin(D/'native_screen.json'),driver=nc['arms']['candidate']['blocks']['2']['driver'])))
if __name__=='__main__': main()

