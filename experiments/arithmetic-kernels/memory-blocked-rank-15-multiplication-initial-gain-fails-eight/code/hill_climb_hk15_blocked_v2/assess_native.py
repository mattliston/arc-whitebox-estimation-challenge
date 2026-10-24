"""Verify fixed native panel receipts, then apply prospective score gates."""
from pathlib import Path
import hashlib
import json
import math
import sys
import numpy as np

D=Path(__file__).resolve().parent; H2=D.parent
NATIVE=H2.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
S32='f226a45750e6c3851e50967ba96f1742bb7bf563d43c1adfc2988fd674041aef'
BASE_PINS={
 'hill_climb_star_projection32_train2_v1/manifest.json':'4327a61916482e1b699de367ca8c9a79ca9ff25c7467f1e37f4b54ce68285324',
 'hill_climb_star_projection32_train2_v1/rows.jsonl':'9dfa3606493dd151cc3c2086dd6b35012912b4a9db2eb043ce7847e8be62e510',
 'hill_climb_star_projection32_v1/train2_guard.json':'3eaaa150076f8f29fee64725dffe5f890e991239fc5d37829d8035eb72a94a04',
 'hill_climb_star_projection32_train6_v1/manifest.json':'728875ac00795f06ef06e5d3a32ca452496fd41f9e0b333e60643cd174d9ee29',
 'hill_climb_star_projection32_train6_v1/rows.jsonl':'2a4fbdd048be56be4fd62e6f7bf5c0e4905fc8a3545c32f0f37ec9b1828356f3',
 'hill_climb_star_projection32_v1/train6_guard.json':'ab5be94d347c6dd32e9da1854567c84614fc6ae92ddb82116e258138d39df5f0'}
BLOCKS=([35,100],[36,45,52,58,64,70])


def pin(p):
    b=p.read_bytes();return dict(path=str(p.resolve()),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))


def read(p):return json.loads(p.read_text())


def guard_ok(path):
    g=read(path)
    assert g['completed'] and g['whole_group_retired'] and g['exit_code']==0
    assert g['error'] is None and not g['remaining_rows']
    assert g['max_rss_bytes']<7500000000 and g['max_swap_growth_bytes']==0
    return g


def main():
    assert len(sys.argv)==2 and sys.argv[1] in ('2','8')
    count=int(sys.argv[1]);blocks=1 if count==2 else 2
    seal=read(D/'source_seal.json')
    for name,sha in seal['files'].items():assert pin(D/name)['sha256']==sha,name
    child=pin(D/'candidate.py')
    assert read(D/'build.json')['candidate']==child
    fixtures=read(D/'shared_fixtures.json')
    assert fixtures['passed'] and fixtures['source_seal_sha256']==pin(D/'source_seal.json')['sha256']
    equivalence=read(D/'equivalence.json')
    assert equivalence['passed'] and equivalence['source_seal_sha256']==pin(D/'source_seal.json')['sha256']
    v1=read(D/'v1_reference.json')
    for name,expected in v1.items():assert pin(Path(expected['path']))==expected,name
    v1_partial=read(Path(v1['native2_guard.json']['path']))
    assert not v1_partial['completed'] and v1_partial['whole_group_retired']
    assert v1_partial['error']=="RuntimeError('RSS trigger')"
    assert v1['candidate.py']['sha256']=='40c0ed8d1dd87c77d299d73802c551f0cf47998370eb4400b9714f6255a97f93'
    fixture_guard=guard_ok(D/'shared_fixtures_guard.json')
    assert fixture_guard['command']==[str(NATIVE),'-B',str(D/'shared_fixtures.py')]
    assert fixture_guard['elapsed_s']<=120
    if count==8:
        prior=read(D/'native2_assessment.json')
        assert prior['admitted'] and prior['candidate']==child
    base=[];candidate=[];evidence=[];max_rss=0
    for index in range(blocks):
        size=(2,6)[index];ranks=BLOCKS[index]
        for is_child in (False,True):
            tag=f'hill_climb_hk15_blocked_train{size}_v2' if is_child else f'hill_climb_star_projection32_train{size}_v1'
            folder=H2/tag
            m=read(folder/'manifest.json')
            assert m['ranks']==ranks and m['numpy']=='2.4.6' and m['runtime']==str(NATIVE)
            assert all(v=='1' for v in m['thread_env'].values())
            assert m['sha256']==(child['sha256'] if is_child else S32)
            assert pin(Path(m['candidate']))['sha256']==m['sha256']
            guard_path=D/f'native{size}_guard.json' if is_child else H2/'hill_climb_star_projection32_v1'/f'train{size}_guard.json'
            g=guard_ok(guard_path);max_rss=max(max_rss,g['max_rss_bytes'])
            if is_child:
                assert g['command']==[str(NATIVE),'-B',str(D/'native_probe.py'),child['path'],','.join(map(str,ranks)),tag]
                assert g['elapsed_s']<=300
                setup=m['setup']['hk_setup']
                assert setup['completed'] and setup['descriptors']==84 and setup['HK_sides']==209
                assert setup['old_prebound']==setup['old_fused']==430 and setup['KS_bound']==24
                assert m['setup']['hk_products']==84 and m['setup']['setup_s']<=4.1
            else:
                for p in (folder/'manifest.json',folder/'rows.jsonl',guard_path):
                    assert pin(p)['sha256']==BASE_PINS[str(p.relative_to(H2))]
            rows=[json.loads(line) for line in (folder/'rows.jsonl').read_text().splitlines()]
            assert [r['rank'] for r in rows]==ranks
            prediction_pins=[]
            for row in rows:
                assert all(math.isfinite(row[k]) and row[k]>0 for k in ('mse','adjusted','util','cpu_s','wall_s'))
                assert type(row['flops']) is int and 0<row['flops']<=2**41
                assert row['util']==row['flops']/2**41
                assert math.isclose(row['adjusted'],row['mse']*max(.1,row['util']),rel_tol=1e-14)
                prediction=folder/f'pred_{row["rank"]}.npy'
                p=np.load(prediction,allow_pickle=False)
                assert p.shape==(16,1024) and p.dtype==np.dtype('float64') and np.isfinite(p).all()
                prediction_pins.append(pin(prediction))
                if is_child:
                    if row['rank']==35:
                        reference=np.load(v1['pred_35.npy']['path'],allow_pickle=False)
                        assert reference.shape==p.shape and reference.dtype==p.dtype
                        assert reference.tobytes()==p.tobytes(), 'v1 rank35 all16 bytes changed'
                    assert row['hk']['selectedowners']>0 and row['hk']['HKproducts']>0
                    assert row['hk']['retired']==1 and row['hk_products']==84
                    assert row['ubc']['retired'] and row['ubc']['retained_cache_bytes']==0 and row['ubc']['live_cache_bytes']==0
            (candidate if is_child else base).extend(rows)
            evidence.append(dict(manifest=pin(folder/'manifest.json'),rows=pin(folder/'rows.jsonl'),
                                 guard=pin(guard_path),predictions=prediction_pins))
    b={r['rank']:r for r in base};c={r['rank']:r for r in candidate}
    ranks=sorted(b);assert ranks==sorted(c) and len(ranks)==count
    delta=lambda key:100*(sum(c[r][key] for r in ranks)/sum(b[r][key] for r in ranks)-1)
    adjusted=delta('adjusted');raw=delta('mse');bill=delta('flops')
    wins=sum(c[r]['adjusted']<b[r]['adjusted'] for r in ranks)
    bb=np.array([b[r]['adjusted'] for r in ranks]);cc=np.array([c[r]['adjusted'] for r in ranks])
    rng=np.random.default_rng(20260926);idx=rng.integers(0,count,size=(50000,count))
    ci=(100*np.quantile(cc[idx].mean(1)/bb[idx].mean(1)-1,[.025,.975])).tolist()
    gates=dict(adjusted_gain=adjusted<=-1.,wins=wins>=(2 if count==2 else 6),raw_regression=raw<=.25)
    if count==8:gates['ci_upper_negative']=ci[1]<0
    result=dict(passed_integrity=True,admitted=all(gates.values()),candidate=child,
                protocol=pin(D/'PROTOCOL.md'),source_seal=pin(D/'source_seal.json'),assessor=pin(Path(__file__)),
                n=count,ranks=ranks,adjusted_pct=adjusted,raw_pct=raw,bill_pct=bill,
                adjusted_wins=wins,adjusted_ci95_pct=ci,gates=gates,maximum_guard_rss_bytes=max_rss,
                rows=[dict(rank=r,adjusted_pct=100*(c[r]['adjusted']/b[r]['adjusted']-1),
                           raw_pct=100*(c[r]['mse']/b[r]['mse']-1),bill_pct=100*(c[r]['flops']/b[r]['flops']-1),
                           hk=c[r]['hk']) for r in ranks],evidence=evidence,
                v1_rank35_all16_bytes_exact=True,targets_read_by_assessor=False,serving_qualified=False,submitted=False)
    with (D/f'native{count}_assessment.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ('admitted','n','adjusted_pct','raw_pct','bill_pct','adjusted_wins','adjusted_ci95_pct','gates')}))


if __name__=='__main__':main()
