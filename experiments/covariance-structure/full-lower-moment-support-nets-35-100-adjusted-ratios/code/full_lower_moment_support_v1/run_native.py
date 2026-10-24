"""Serial target-free native qualification through the unchanged metered runner."""
import argparse,runpy,sys
from common import HERE,D,sha,read,write,seal_check,guarded

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['null','dev','cold100'],required=True)
    args=parser.parse_args()
    seal_check()
    assert read(HERE/'unit_result.json')['passed']
    guarded('unit')
    if args.stage!='null':
        assert read(HERE/'null_validation.json')['passed']
        guarded('null')
    if args.stage=='cold100':
        assert read(HERE/'dev_validation.json')['passed']
        guarded('dev')
    meta=read(HERE/'inputs.json')
    candidate=HERE/('null.py' if args.stage=='null' else 'candidate.py')
    out=HERE/args.stage
    ranks=[100] if args.stage=='cold100' else [35,100,35]
    runner=D/'weight5_reuse_v1/native.py'
    sys.argv=[str(runner),str(candidate),str(out),'--ranks',','.join(map(str,ranks))]
    runpy.run_path(str(runner),run_name='__main__')
    import numpy as np
    receipt=read(out/'result.json')
    assert receipt['candidate_sha256']==sha(candidate)
    assert receipt['runner_sha256']==sha(runner)
    assert receipt['truth_accessed'] is False
    records=receipt['records']
    assert [r['rank'] for r in records]==ranks
    parents=read(meta['parent_dev']['path'])['records']
    proofs=[]
    for i,r in enumerate(records):
        parent=parents[1 if args.stage=='cold100' else i]
        assert r['weights_sha256']==parent['weights_sha256']
        assert sha(r['output'])==r['output_sha256']
        assert r['max_rss_bytes']<8_000_000_000
        assert r['weight5_cache']['completed'] and r['weight5_cache']['numeric_cache_retired']
        assert r['input_flops_separate']==0
        assert r['includes_initialization']==(i==0)
        assert r['multiplier']==max(.1,r['flops']/2**41)
        a=np.load(r['output']);b=np.load(parent['output'])
        assert a.shape==(16,1024) and a.dtype==np.float32
        assert np.isfinite(a).all() and (a>=0).all()
        first_four=a[:4].tobytes()==b[:4].tobytes()
        assert first_four
        proof=dict(rank=r['rank'],output_sha256=r['output_sha256'],complete_flops=r['flops'],
                   first_four_parent_rows_exact=first_four,cache_retired=True)
        if args.stage=='null':
            assert a.tobytes()==b.tobytes() and r['output_sha256']==parent['output_sha256']
            assert r['flops']==parent['flops']
            proof.update(all_sixteen_parent_rows_exact=True,complete_parent_bill_exact=True)
        proofs.append(proof)
    repeat=None
    if args.stage in ('null','dev'):
        repeat=records[0]['output_sha256']==records[2]['output_sha256']
        assert repeat
    cold_match=None
    if args.stage=='cold100':
        warm=read(HERE/'dev/result.json')['records'][1]
        cold_match=warm['output_sha256']==records[0]['output_sha256']
        assert cold_match
    write(HERE/(args.stage+'_validation.json'),dict(passed=True,stage=args.stage,proofs=proofs,
        repeat35_exact=repeat,warm_cold100_exact=cold_match,truth_accessed=False,
        result_sha256=sha(out/'result.json'),source_seal_sha256=sha(HERE/'sources_sealed.json')))
    print('FLMS '+args.stage+' qualification passed',flush=True)

if __name__=='__main__':main()

