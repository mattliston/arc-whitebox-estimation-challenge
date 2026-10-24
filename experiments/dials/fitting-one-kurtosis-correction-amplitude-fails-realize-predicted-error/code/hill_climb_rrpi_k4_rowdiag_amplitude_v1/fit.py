"""Inactive fixed one-coordinate proposal; only root runs saved native FIT4 fitting."""
from pathlib import Path
import json, math, os, sys
import prepare as b

D=Path(__file__).resolve().parent


def main():
    assert len(sys.argv)==3;s,pp,c=b.configuration(sys.argv[1],sys.argv[2])
    q=b.module(c['engine']['path'],'amplitude_actual_metrics');f=b.module(c['helper']['path'],'amplitude_saved_panels');np=q.np
    assert str(Path(sys.executable).absolute())==c['runtime']==str(q.NATIVE) and np.__version__=='2.4.6'
    assert all(os.environ.get(k)=='1' for k in q.THREADS)
    truth=f.truths(q,c,'fit4');control=c['controls']['fit4'];assert control['arm']['source']==c['parent']
    base,yy,bp=f.panel(q,c,control['arm'],b.FIT,truth,control['artifacts'])
    arrays={};artifacts={'base':bp};metrics={}
    for name in ('minus','plus'):
        rows,arrays[name],artifacts[name]=f.panel(q,c,c['arms'][name],b.FIT,truth)
        for rank in b.FIT:assert arrays[name][rank][:2].tobytes()==yy[rank][:2].tobytes()
        metrics[name]=q.metrics(base,rows)
    odd=[];error=[];even=[]
    for row in base:
        rank=row['rank'];scale=math.sqrt(max(.1,row['util'])/1024)
        a=(arrays['plus'][rank][15]-arrays['minus'][rank][15])/2
        e=(arrays['plus'][rank][15]+arrays['minus'][rank][15])/2-yy[rank][15]
        odd.append(a*scale);error.append((yy[rank][15]-truth[rank])*scale)
        ae=float(np.sum((a*scale)**2));ee=float(np.sum((e*scale)**2))
        even.append(dict(rank=rank,odd_weighted_squared=ae,even_weighted_squared=ee,even_to_odd_energy=ee/ae if ae>0 else None))
    A=np.concatenate(odd);residual=np.concatenate(error);G=float(A@A);g=float(A@residual);Q=1.1*G
    assert all(math.isfinite(x) for x in (G,g,Q)) and G>=0
    x=0. if G==0 else min(2.,max(-2.,-g/Q))
    den=float(residual@residual);pred=float((residual+x*A)@(residual+x*A));assert den>0 and math.isfinite(pred)
    gain=100*(1-pred/den);admitted=gain>0 and G>0
    result=dict(passed_integrity=True,endpoint_admitted=admitted,protocol=pp,fitter=q.pin(__file__),parent=c['parent'],
                fit_ranks=b.FIT,alpha=x,G=G,g=g,Q=Q,predicted_parent_weighted_loss_gain_pct=gain,
                forecast_uses_parent_bill_weights=True,forecast_includes_candidate_compute=False,
                actual_candidate_bills_used_at_endpoint_gate=True,
                even_diagnostics=even,aggregate_even_to_odd_energy=sum(v['even_weighted_squared'] for v in even)/G if G>0 else None,
                arm_metrics=metrics,artifacts=artifacts,endpoint=None,check_targets_read=False,
                exact_gradient_claim=False,one_endpoint_only=True,bank_qualified=False)
    if admitted:
        source,proof=b.edit_source(q.checked(c['rowdiag_source']).decode(),x)
        with (D/'endpoint.py').open('x') as out:out.write(source)
        result.update(endpoint=q.pin(D/'endpoint.py'),source_inverse=proof)
    q.once(D/'fit_result.json',result)
    if admitted:q.once(D/'endpoint_fit4_commands.json',b.commands(s,pp,c,result['endpoint'],'fit4'))
    print(json.dumps({k:result[k] for k in ('endpoint_admitted','alpha','predicted_parent_weighted_loss_gain_pct','aggregate_even_to_odd_energy')}))


if __name__=='__main__':main()
