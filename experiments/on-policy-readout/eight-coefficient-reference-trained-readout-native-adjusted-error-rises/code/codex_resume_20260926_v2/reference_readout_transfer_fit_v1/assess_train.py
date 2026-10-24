from common import *
verify();guard('native_parent');guard('native_fit')
for e in read(HERE/'native_seal.json')['sources']:assert pin(e['path'])==e
parent=read(HERE/'native_parent/complete.json');candidate=read(HERE/'native_fit/complete.json')
assert parent['actual_fast_path'] and candidate['actual_fast_path'] and not candidate['training_reference_deployed']
assert tuple(r['mini'] for r in parent['rows'])==TRAIN and tuple(r['mini'] for r in candidate['rows'])==TRAIN
t=packet(read(HERE/'targets.json')['packet']);assert tuple(t['train'])==TRAIN
rows=[]
for i,(b,c) in enumerate(zip(parent['rows'],candidate['rows'])):
    assert b['mini']==c['mini'];bp=packet(b['packet'])['prediction'];cp=packet(c['packet'])['prediction']
    assert bp[:9].tobytes()==cp[:9].tobytes()
    bm=math.fsum(float(x)**2 for x in bp[-1]-t['targets'][i])/1024
    cm=math.fsum(float(x)**2 for x in cp[-1]-t['targets'][i])/1024
    assert math.isclose(bm,b['mse'],rel_tol=1e-14) and math.isclose(cm,c['mse'],rel_tol=1e-14)
    raw=cm/bm;cost=c['flops']/b['flops'];adj=(cm*c['flops'])/(bm*b['flops'])
    rows.append(dict(mini=c['mini'],raw_ratio=raw,bill_ratio=cost,adjusted_ratio=adj,win=adj<1))
raw=sum(r['mse'] for r in candidate['rows'])/sum(r['mse'] for r in parent['rows'])
adjusted=sum(r['mse']*r['flops'] for r in candidate['rows'])/sum(r['mse']*r['flops'] for r in parent['rows'])
wins=sum(r['win'] for r in rows)
gates=dict(pooled_adjusted=adjusted<=.98,at_least_four_wins=wins>=4,maximum_regression=max(r['adjusted_ratio'] for r in rows)<=1.05)
result=dict(passed=all(gates.values()),gates=gates,rows=rows,pooled_raw_ratio=raw,pooled_adjusted_ratio=adjusted,wins=wins,
    actual_fast_estimator=True,complete_cost_included=True,linear_fit_prediction_is_not_score=True,
    check_targets_read=False,candidate=read(HERE/'fit_result.json')['candidate'])
once(HERE/'native_train_result.json',result)
print(json.dumps(result),flush=True)
