"""Root-only actual fixed endpoint admission; CHECK8 follows an actual FIT4 pass."""
from pathlib import Path
import json, os, sys
import prepare as b

D=Path(__file__).resolve().parent


def main():
    assert len(sys.argv)==4 and sys.argv[3] in ('fit4','check8')
    stage=sys.argv[3];s,pp,c,r=b.endpoint_configuration(sys.argv[1],sys.argv[2],stage)
    q=b.module(c['engine']['path'],'amplitude_endpoint_metrics');f=b.module(c['helper']['path'],'amplitude_endpoint_panels')
    assert str(Path(sys.executable).absolute())==c['runtime']==str(q.NATIVE) and q.np.__version__=='2.4.6'
    assert all(os.environ.get(k)=='1' for k in q.THREADS)
    fitted=q.pin(D/'fit_result.json')
    truth=f.truths(q,c,stage);ranks=b.FIT if stage=='fit4' else b.CHECK;control=c['controls'][stage]
    base,ba,bp=f.panel(q,c,control['arm'],ranks,truth,control['artifacts'])
    candidate,ca,cp=f.panel(q,c,b.arm(c,r['endpoint'],'endpoint',stage),ranks,truth)
    for rank in ranks:assert ca[rank][:2].tobytes()==ba[rank][:2].tobytes()
    metrics=q.metrics(base,candidate);admitted=metrics['raw_pct']<0 and metrics['adjusted_pct']<0
    result=dict(passed_integrity=True,admitted=admitted,stage=stage,protocol=pp,assessor=q.pin(__file__),
                endpoint=r['endpoint'],parent=c['parent'],fit_result=fitted,metrics=metrics,
                all16_finite=True,exact_parity_layers=[0,1],artifacts=dict(base=bp,candidate=cp),
                bank_qualified=False,full40_own_resources_and_serving_still_required=True)
    q.once(D/('endpoint_'+stage+'_assessment.json'),result)
    if admitted and stage=='fit4':q.once(D/'endpoint_check8_commands.json',b.commands(s,pp,c,r['endpoint'],'check8'))
    print(json.dumps(dict(stage=stage,admitted=admitted,metrics=metrics)))


if __name__=='__main__':main()
