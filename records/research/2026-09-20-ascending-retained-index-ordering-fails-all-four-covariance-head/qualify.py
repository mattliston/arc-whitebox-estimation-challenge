import sys
import numpy as np
from common import HERE,OLD,read,write,pin,verify,seal_check

seal_check();build=read(HERE/'build.json');assert read(HERE/'unit.json')['passed']
def packet(path):
    r=read(path)
    for key in ['source','trace','prediction','weights']:verify(r[key])
    assert not r['truth_accessed'] and r['selector_calls']==45 and r['extra_capture_arithmetic']==0
    assert r['numeric_cache_retired'] and r['includes_initialization']
    y=np.load(r['prediction']['path'],allow_pickle=False)
    with np.load(r['trace']['path'],allow_pickle=False) as z:t={k:z[k].copy() for k in z.files}
    assert y.shape==(16,1024) and y.dtype==np.float32 and np.isfinite(y).all() and (y>=0).all()
    return r,y,t

def differences(a,b):
    result=[]
    for i in range(45):
        ca,cb=int(a['counts'][i]),int(b['counts'][i])
        members=int(np.count_nonzero(a['masks'][i]!=b['masks'][i]))
        order=int(np.count_nonzero(a['indices'][i]!=b['indices'][i]))
        result.append(dict(layer=i//3,role=['live','response','birth'][i%3],count_a=ca,count_b=cb,
            membership_difference=members,index_position_difference=order,
            pure_order_change=members==0 and order>0,
            score_max_absolute_difference=float(np.max(np.abs(a['scores'][i]-b['scores'][i])))))
    changed=[x for x in result if x['index_position_difference']]
    return dict(rows=result,first_changed_selector=changed[0] if changed else None,
        pure_order_cells=sum(x['pure_order_change'] for x in result),
        membership_change_cells=sum(x['membership_difference']>0 for x in result),
        count_change_cells=sum(x['count_a']!=x['count_b'] for x in result))

if sys.argv[1]=='capture':
    rows=[];traces={}
    for expected in build['original36_controls']:
        label=expected['label'];verify(expected['receipt']);verify(expected['output'])
        r,y,t=packet(HERE/f'original36_{label}/result.json')
        old=np.load(expected['output']['path'],allow_pickle=False)
        assert y.tobytes()==old.tobytes() and r['flops']==expected['call']['flops']
        for i in range(45):
            count=int(t['counts'][i]);expected_order=np.argsort(-t['scores'][i],kind='stable')[:count]
            assert np.array_equal(t['indices'][i,:count],expected_order)
        traces[label]=t
        rows.append(dict(label=label,all16_bytes_exact=True,complete_bill_exact=True,receipt=pin(HERE/f'original36_{label}/result.json')))
    write(HERE/'capture_qualified.json',dict(passed=True,rows=rows,original36_head0_changes={
        label:differences(traces['base'],traces[label]) for label in ['plus','minus','double']},
        targets_accessed=False,extra_parent_arithmetic=0))
    print(dict(passed=True,original_replays=4),flush=True)
elif sys.argv[1]=='response':
    assert read(HERE/'capture_qualified.json')['passed']
    sealed=read(HERE/'predictions_sealed.json');assert len(sealed['rows'])==15
    packets={};rows=[]
    for row in sealed['rows']:
        verify(row['receipt']);r,y,t=packet(row['receipt']['path'])
        for i in range(45):
            count=int(t['counts'][i]);assert np.all(np.diff(t['indices'][i,:count])>0)
        key=(row['rank'],row['coordinate'],row['label'],row['repeat'])
        packets[key]=(r,y,t)
        rows.append(dict(**row,flops=r['flops'],prediction=r['prediction'],trace=r['trace']))
    repeated=packets[(36,0,'plus',True)];original=packets[(36,0,'plus',False)]
    assert repeated[0]['flops']==original[0]['flops'] and repeated[1].tobytes()==original[1].tobytes()
    assert all(repeated[2][k].tobytes()==original[2][k].tobytes() for k in repeated[2])
    old=read(OLD/'response_qualified.json');cells=[];within=[]
    for rank in [36,38]:
        baseline,base_y,base_trace=packets[(rank,None,'base',False)];base=base_y[-1].astype(np.float64)
        for coordinate in [0,8]:
            values={}
            for label in ['plus','minus','double']:
                r,y,t=packets[(rank,coordinate,label,False)]
                assert y[:8].tobytes()==base_y[:8].tobytes()
                values[label]=y[-1]
                within.append(dict(rank=rank,coordinate=coordinate,label=label,
                    delta=differences(base_trace,t)))
            plus,minus,double=(values[k].astype(np.float64) for k in ['plus','minus','double'])
            centered=(plus-minus)/2;forward=plus-base
            spacing=np.sqrt(np.sum(np.spacing(np.abs(values['plus'])).astype(np.float64)**2)+
                            np.sum(np.spacing(np.abs(values['minus'])).astype(np.float64)**2))
            snr=float(np.linalg.norm(plus-minus)/max(spacing,1e-30))
            asym=float(np.linalg.norm(forward-centered)/max(np.linalg.norm(centered),1e-30))
            curvature=float(np.linalg.norm(double-2*plus+base)/max(np.linalg.norm(forward),1e-30))
            old_cell=next(x for x in old['cells'] if x['rank']==rank and x['coordinate']==coordinate)
            cells.append(dict(rank=rank,coordinate=coordinate,centered_response_rms=float(np.sqrt(np.mean(centered**2))),
                spacing_snr=snr,forward_vs_centered_relative=asym,double_curvature_relative=curvature,
                original_asymmetry=old_cell['forward_vs_centered_relative'],original_curvature=old_cell['double_curvature_relative'],
                asymmetry_over_original=asym/old_cell['forward_vs_centered_relative'],curvature_over_original=curvature/old_cell['double_curvature_relative'],
                passed=snr>=20 and asym<=.10 and curvature<=.10))
    cross=[]
    for label in ['base','plus','minus','double']:
        a,ya,ta=packet(HERE/f'original36_{label}/result.json')
        b,yb,tb=packets[(36,None if label=='base' else 0,label,False)]
        cross.append(dict(rank=36,label=label,comparison=differences(ta,tb),
            output_difference_rms_by_layer=np.sqrt(np.mean((yb.astype(np.float64)-ya.astype(np.float64))**2,axis=1)).tolist(),
            canonical_over_original_flops=b['flops']/a['flops']))
    write(HERE/'selector_differences.json',dict(within_canonical=within,matched_original_vs_canonical=cross,
        original_within=read(HERE/'capture_qualified.json')['original36_head0_changes'],targets_accessed=False))
    passed=all(c['passed'] for c in cells)
    write(HERE/'response_result.json',dict(passed=passed,cells=cells,rows=rows,repeat_output_bill_trace_exact=True,
        first8_own_base_exact=True,truth_accessed=False,fit_steps=0,full_native_calls=19,
        original_response=build['old_response'],selector_comparison=pin(HERE/'selector_differences.json'),
        scope='Four sampled native secants under a retained-index ordering intervention; not a complete derivative or accuracy result.'))
    write(HERE/'outcome.json',dict(stage='response',passed=passed,study_closed=not passed,
        new_fit_design_licensed=passed,accuracy_scored=False,fitted=False,uploaded=False))
    print(dict(passed=passed,cells=cells),flush=True)
else:raise ValueError(sys.argv[1])
