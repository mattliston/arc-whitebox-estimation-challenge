import argparse,runpy,sys,json
from common import HERE,D,read,write,sha,verify
def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['null','dev','cold100']);stage=parser.parse_args().stage
    verify('native_sources_sealed.json');assert read(HERE/'unit.json')['passed']
    previous='unit' if stage=='null' else ('null' if stage=='dev' else 'dev')
    g=read(HERE/(previous+'_watch.json'));assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
    if stage!='null':assert read(HERE/(previous+'_validation.json'))['passed']
    candidate=HERE/('null.py' if stage=='null' else 'candidate.py');out=HERE/stage;ranks=[100] if stage=='cold100' else [35,100,35]
    runner=D/'weight5_reuse_v1/native.py';sys.argv=[str(runner),str(candidate),str(out),'--ranks',','.join(map(str,ranks))]
    runpy.run_path(str(runner),run_name='__main__')
    import numpy as np
    receipt=read(out/'result.json');assert receipt['candidate_sha256']==sha(candidate) and receipt['runner_sha256']==sha(runner) and not receipt['truth_accessed']
    records=receipt['records'];assert [r['rank'] for r in records]==ranks
    parents=read(read(HERE/'inputs.json')['parent_dev']['path'])['records'];proofs=[]
    for i,r in enumerate(records):
        p=parents[1 if stage=='cold100' else i]
        assert r['weights_sha256']==p['weights_sha256'] and sha(r['output'])==r['output_sha256']
        assert r['max_rss_bytes']<8_000_000_000 and r['weight5_cache']['completed'] and r['weight5_cache']['numeric_cache_retired']
        assert r['input_flops_separate']==0 and r['includes_initialization']==(i==0)
        a=np.load(r['output']);b=np.load(p['output']);assert a.shape==(16,1024) and a.dtype==np.float32 and np.isfinite(a).all() and (a>=0).all()
        assert a[0].tobytes()==b[0].tobytes()
        if stage=='null':assert a.tobytes()==b.tobytes() and r['output_sha256']==p['output_sha256'] and r['flops']==p['flops']
        proofs.append(dict(rank=r['rank'],flops=r['flops'],first_row_exact=True,all_parent_bytes_and_bill_exact=stage=='null',cache_retired=True))
    if stage in ['null','dev']:assert records[0]['output_sha256']==records[2]['output_sha256']
    else:assert records[0]['output_sha256']==read(HERE/'dev/result.json')['records'][1]['output_sha256']
    write(HERE/(stage+'_validation.json'),dict(passed=True,stage=stage,proofs=proofs,repeat_or_warm_cold_exact=True,truth_accessed=False,result_sha256=sha(out/'result.json')))
    print(json.dumps(dict(stage=stage,passed=True,flops=[r['flops'] for r in records])),flush=True)
if __name__=='__main__':main()
