"""Root-only fixed output-average diagnostic. No fitting, estimator, or parameter export."""
from pathlib import Path
import hashlib
import importlib.util
import io
import json
import sys
import numpy as np
D=Path(__file__).resolve().parent

def pin(p):
    p=Path(p).resolve();b=p.read_bytes()
    return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def checked(r):
    assert pin(r['path'])==r,r['path']
    return Path(r['path']).read_bytes()
def load(r):return json.loads(checked(r))
def prediction(r):
    a=np.load(io.BytesIO(checked(r)),allow_pickle=False)
    assert a.shape==(16,1024) and a.dtype==np.dtype('float64') and np.isfinite(a).all()
    return a

def main():
    assert len(sys.argv)==3
    pp=pin(sys.argv[1]);assert pp['sha256']==sys.argv[2] and Path(pp['path'])==D/'forecast_plan.json'
    p=load(pp);assert p['schema']=='fixed-half-output-average-diagnostic-v1' and p['coefficient']==.5
    assert p['bill_proxy']=='D16 per-rank billed FLOPs; no actual ensemble or parameter-average cost claim'
    assert p['script']==pin(__file__) and sys.executable==p['runtime'] and np.__version__=='2.4.6'
    for r in p['pins']:checked(r)
    spec=importlib.util.spec_from_file_location('average_frozen_validation_common',p['common']['path'])
    common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
    c,index,base=common.configuration(Path(p['config16']['path']),p['config16']['sha256'])
    r16=load(p['assessment16']);r8=load(p['assessment8']);f8=load(p['protocol8'])
    assert r16['passed'] is False and r16['admitted'] is False and r16['stage']=='VALID8'
    assert r8['passed'] is False and r8['retained_evidence_passed'] is False and r8['original_admitted'] is False
    assert r16['receipt_checks_passed'] and r8['receipt_checks_passed']
    assert r16['config']==p['config16'] and r8['protocol']==p['protocol8']
    assert r16['candidate']==c['candidate'] and r8['candidate']==f8['candidate']
    assert r16['baseline']==r8['baseline']==c['baseline']
    assert r16['baseline_index']==r8['baseline_index']==c['baseline_index']
    original=load(r8['original_assessment']);assert original['passed'] is False and original['admitted'] is False
    assert c['ranks']==f8['ranks']==p['ranks']==[37,39,42,48,59,63,66,67]
    rows16=[json.loads(x) for x in checked(r16['artifacts']['rows.jsonl']).decode().splitlines()]
    rows8=[json.loads(x) for x in checked(r8['artifacts']['rows.jsonl']).decode().splitlines()]
    assert [r['rank'] for r in rows16]==[r['rank'] for r in rows8]==p['ranks']
    for receipt in (r8,r16):
        for r in list(receipt['artifacts'].values())+list(receipt['predictions'].values()):checked(r)
        g=load(receipt['native_guard'])
        assert g['completed'] and g['whole_group_retired'] and g['remaining_rows']==[]
        assert g['exit_code']==0 and g['error'] is None and g['max_swap_growth_bytes']==0
        assert 0<g['elapsed_s']<=300 and 0<g['max_rss_bytes']<g['rss_trigger_bytes']==7500000000
    result_rows=[];diagnostics=[]
    for i,rank in enumerate(p['ranks']):
        a=prediction(r8['predictions'][str(rank)]);b=prediction(r16['predictions'][str(rank)])
        parent=prediction(index['data'][i]['prediction'])
        with np.load(io.BytesIO(checked(c['data'][i]['truth'])),allow_pickle=False) as z:
            truth=z['mc_post'];assert truth.shape==(16,1024) and np.isfinite(truth).all()
            truth=truth[15].astype(np.float64)
        assert a[0].tobytes()==b[0].tobytes()==parent[0].tobytes()
        mse8=float(np.mean((a[15]-truth)**2));mse16=float(np.mean((b[15]-truth)**2))
        assert mse8==rows8[i]['mse'] and mse16==rows16[i]['mse']
        assert float(np.mean((parent[15]-truth)**2))==base[i]['mse']
        average=.5*(a+b)
        assert np.isfinite(average).all() and average[0].tobytes()==parent[0].tobytes()
        mse=float(np.mean((average[15]-truth)**2))
        row=dict(rows16[i]);row['mse']=mse;row['adjusted']=mse*max(.1,row['util'])
        result_rows.append(row)
        diagnostics.append(dict(rank=rank,baseline_mse=base[i]['mse'],update8_mse=mse8,update16_mse=mse16,
            fixed_half_output_average_mse=mse,error_cross_moment=float(np.mean((a[15]-truth)*(b[15]-truth))),
            d16_flops_proxy=row['flops'],d16_util_proxy=row['util'],adjusted_proxy=row['adjusted'],all16_finite=True,layer0_exact=True))
    limits=dict(adjusted_pct_max=-.65,raw_pct_strict_max=0.,wins_min=6,max_regression_pct=2.)
    assert p['forecast_gates']==limits
    m=common.metrics(base,result_rows)
    gate=dict(adjusted=m['adjusted_pct']<=-.65,raw=m['raw_pct']<0,
              wins=m['adjusted_wins']>=6,max_regression=m['max_per_network_adjusted_regression_pct']<=2)
    result=dict(schema='fixed-half-output-average-diagnostic-result-v1',completed=True,integrity_passed=True,
        plan=pp,script=pin(__file__),common=p['common'],coefficient=.5,input_pins=p['pins'],
        forecast_gates=gate,forecast_gate_passed=all(gate.values()),metrics=m,metrics_update8=common.metrics(base,rows8),metrics_update16=common.metrics(base,rows16),
        per_network=diagnostics,bill_proxy=p['bill_proxy'],cpu_proxy='D16 copied only for common metric compatibility; no measured ensemble runtime',
        original8_and_followup8_and_update16_failures_preserved=True,
        exact_saved_prediction_average_diagnostic=True,parameter_average_predictions_not_guaranteed=True,
        adaptive_reused_validation=True,bootstrap_intervals_descriptive_not_fixed_design=True,
        coefficient_fitted=False,search_performed=False,model_executed=False,candidate_exported=False,
        admitted=False,qualified=False,bank_qualified=False,submitted=False)
    with (D/'forecast_result.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(completed=True,metrics=result['metrics'],output=pin(D/'forecast_result.json'))))

if __name__=='__main__':main()
