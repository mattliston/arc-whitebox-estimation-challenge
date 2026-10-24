import argparse,resource
import numpy as np
from common import *
ap=argparse.ArgumentParser();ap.add_argument('--confirmation',action='store_true');args=ap.parse_args()
seal_check();meta=read(HERE/'inputs.json');build=read(HERE/'build.json');verify(build['candidate'])
def checks(row,parent_prediction,parent_flops,bank=False):
    verify(row['prediction']);verify(row['capture']);verify(row['weights'])
    with np.load(row['capture']['path'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    pred=np.load(row['prediction']['path'],allow_pickle=False)
    assert np.array_equal(pred,a['prediction']) and np.array_equal(pred[:15],parent_prediction[:15])
    assert row['flops']==parent_flops
    if bank:assert np.array_equal(pred,parent_prediction)
    first=a['sigma'].astype(np.float32)*np.float32(a['rms'])
    middle=a['hidden']@a['head']+np.float32(a['head_b'])
    last=np.float32(1)+np.float32(.25)*np.tanh(np.asarray(a['gain'],np.float32))
    delta=(first*middle)*last
    fp64=a['sigma'].astype(np.float32).astype(float)*float(a['rms'])*(a['hidden'].astype(float)@a['head'].astype(float)+float(a['head_b']))*(1+.25*np.tanh(float(a['gain'])))
    abs_error=float(np.max(np.abs(delta-a['delta'])));rel=relative(a['delta'],fp64)
    assert abs_error<=2e-7 and rel<=3e-5,(row['rank'],abs_error,rel)
    expected=np.maximum(a['host'].astype(float)+a['delta'].astype(float),0).astype(np.float32)
    assert np.array_equal(expected,pred[-1])
    if not bank:
        with np.load(D/'full_lower_readout_fit_v1/all_train_coefficients.npz',allow_pickle=False) as z:h=z['head']
        assert np.array_equal(np.r_[a['head'],a['head_b']],h)
    return dict(rank=row['rank'],index=row['index'],all16_exact=bank,first15_exact=True,complete_bill_exact=True,
        fp32_delta_max_error=abs_error,fp64_delta_relative=rel,host_plus_delta_exact=True),a
rows=[]
if not args.confirmation:
    for name in ('bank_triple','candidate_triple','cold100'):done(name)
    prior=read(meta['bank_triple']['path'])['records']
    branches={name:read(HERE/name/'result.json') for name in ('bank_triple','candidate_triple','cold100')}
    assert branches['bank_triple']['source']==pin(PARENT)
    assert all(branches[n]['source']==build['candidate'] for n in ('candidate_triple','cold100'))
    bank=branches['bank_triple']['records'];candidate=branches['candidate_triple']['records'];cold=branches['cold100']['records'][0]
    assert [r['rank'] for r in bank]==[r['rank'] for r in candidate]==[35,100,35]
    for i,(old,b,c) in enumerate(zip(prior,bank,candidate)):
        saved=np.load(old['output'],allow_pickle=False)
        br,ba=checks(b,saved,old['flops'],True);cr,ca=checks(c,saved,old['flops'])
        for k in ('host','sigma','hidden','rms','gain'):assert ba[k].tobytes()==ca[k].tobytes(),k
        assert b['includes_initialization']==c['includes_initialization']==(i==0)
        rows += [dict(branch='bank',**br),dict(branch='candidate',**cr)]
    assert candidate[0]['prediction']['sha256']==candidate[2]['prediction']['sha256']
    assert candidate[0]['capture']['sha256']==candidate[2]['capture']['sha256']
    cold_receipt=read(meta['bank_cold100']['path']);verify(cold_receipt['file'])
    with np.load(cold_receipt['file']['path'],allow_pickle=False) as z:pp=z['prediction'].copy()
    cr,ca=checks(cold,pp,cold_receipt['parent_flops']);rows.append(dict(branch='cold100',**cr))
    assert cold['prediction']['sha256']==candidate[1]['prediction']['sha256']
    assert cold['capture']['sha256']==candidate[1]['capture']['sha256']
    write(HERE/'qualification.json',dict(passed=True,rows=rows,candidate=build['candidate'],bank=pin(PARENT),
        first15_and_complete_bills_exact=True,bank_all16_parity=True,repeat_exact=True,warm_cold100_exact=True,
        final_hidden_host_sigma_bytes_unchanged=True,targets_decoded=False,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
else:
    done('qualify');assert read(HERE/'qualification.json')['passed']
    sealed=[]
    for entry in meta['confirmation']:
        rank=entry['rank'];name=f'confirmation_{rank}';guard=done(name);path=HERE/name/'result.json';result=read(path)
        assert result['source']==build['candidate'] and len(result['records'])==1 and not result['truth_accessed']
        row=result['records'][0];assert row['rank']==rank and row['includes_initialization'] and not row['bank']
        assert row['weights']==entry['weights'];verify(entry['parent']['prediction'])
        pp=np.load(entry['parent']['prediction']['path'],allow_pickle=False)
        cr,ca=checks(row,pp,entry['parent']['flops']);rows.append(cr)
        sealed.append(dict(**entry,call=row,result=pin(path),watch=pin(HERE/f'{name}_watch.json')))
    assert [x['rank'] for x in sealed]==CONFIRM
    write(HERE/'predictions_sealed.json',dict(completed=True,rows=sealed,checks=rows,candidate=build['candidate'],
        sources=pin(HERE/'sources_sealed.json'),targets_decoded=False,complete_native_calls=8,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(passed=True,confirmation=args.confirmation,checks=len(rows)),flush=True)
