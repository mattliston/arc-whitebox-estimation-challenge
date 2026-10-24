"""Metadata-only stage receipts; never decodes arrays or changes scientific code."""
import argparse,datetime,json
from pathlib import Path
from common import HERE,TRAIN,read,write,pin,sha,verify

def guard(name):
    g=read(HERE/name)
    assert g['completed'] and g['whole_group_retired'] and g['exit_code']==0 and g['error'] is None,name
    assert g['max_swap_growth_bytes']==0 and g['max_rss_bytes']<8_000_000_000,name
    return g

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['captures','fit_inputs','fit_terminal']);a=ap.parse_args()
    verify();verify('learning_sources_sealed.json')
    files=[HERE/'sources_sealed.json',HERE/'learning_sources_sealed.json'];extra={}
    if a.stage=='captures':
        total=0;flops=0;rows=[]
        for rank in [None]+TRAIN:
            stage='qualification' if rank is None else str(rank)
            wn='capture_'+stage+'_watch.json';g=guard(wn)
            folder=HERE/'capture_qualification' if rank is None else HERE/f'captures/rank{rank:04d}'
            rp=folder/'result.json';r=read(rp)
            assert r['completed'] and not r['truth_accessed'] and r['extra_parent_numerical_operations']==0
            assert r['source_proof']['inverse_byte_exact'] and r['peak_rss_bytes']<8_000_000_000
            assert r['full_estimator_calls']==(3 if rank is None else 1)
            if rank is None:assert r['repeat_all_capture_bytes_exact']
            for v in r['records']:
                assert v['all16_parent_bytes_and_bill_exact'] and v['numeric_cache_retired']
                assert v['peak_rss_bytes']<8_000_000_000
                if rank is None:assert v['old_physical_neural_common_fields_exact']
                assert sha(v['capture']['path'])==v['capture']['sha256']
                files.append(Path(v['capture']['path']));flops+=v['flops']
            total+=r['full_estimator_calls'];rows.append(dict(stage=stage,result=pin(rp),guard=pin(HERE/wn)))
            files.extend([rp,HERE/wn,HERE/('capture_'+stage+'.log')])
        assert total==9
        extra=dict(full_estimator_calls=total,total_parent_flops=flops,all16_output_and_bills_exact=True,
                   truth_arrays_accessed=False,capture_stages=rows)
    elif a.stage=='fit_inputs':
        verify('captures_sealed.json');guard('prepare_watch.json')
        p=read(HERE/'preparation.json')
        assert p['completed'] and p['train_ranks']==TRAIN and p['rows']==49152 and p['source_cells']==48
        assert not p['check_target_access'] and not p['ordinary_1B_access']
        assert sha(p['data']['path'])==p['data']['sha256']
        for row in p['proofs']:
            assert len(row['layers'])==8
            assert all(max(c['affine_mean_relative'],c['affine_covariance_relative'])<=1e-4 for c in row['layers'])
        for unit in ['fnp','torch']:
            guard('unit_'+unit+'_watch.json');assert read(HERE/('unit_'+unit+'.json'))['passed']
        files += [HERE/n for n in ['captures_sealed.json','preparation.json','train_data.npz','prepare_watch.json',
                  'unit_fnp.json','unit_torch.json','unit_fixture.npz','unit_fnp_watch.json','unit_torch_watch.json']]
        extra=dict(train_ranks=TRAIN,check_labels_accessed=False,fit_started=False)
    else:
        guard('fit_watch.json');r=read(HERE/'fit_result.json')
        assert r['completed'] and r['passed'] and r['epochs']==100 and r['updates']==2400
        assert r['optimizer_and_leaves_retired'] and r['reloaded_nontrainable']
        assert not r['check_targets_accessed'] and not r['check_forecasts_started']
        files += [HERE/n for n in ['fit_inputs_sealed.json','fit_result.json','terminal.npz','fit_schedule.npy',
                  'train_source_predictions.npy','fit_watch.json']]
        extra=dict(train_ratio=r['train_ratio'],train_gate_passed=True,check_labels_accessed=False)
    pins=[pin(p) for p in sorted(set(files))]
    r=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage=a.stage,pins=pins,**extra)
    out=HERE/(a.stage+'_sealed.json');write(out,r)
    print(json.dumps(dict(seal=pin(out),pins=len(pins),**extra)),flush=True)

if __name__=='__main__':main()
