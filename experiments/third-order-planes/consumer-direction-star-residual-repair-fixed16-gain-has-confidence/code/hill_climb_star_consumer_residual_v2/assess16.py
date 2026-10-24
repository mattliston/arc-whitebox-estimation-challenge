"""Offline fixed-sixteen closeout; no estimator, truth read or automatic neighbor."""
import math
from pathlib import Path
from assess_screen import HERE, H2, pin, read, check, guard, load_run, np
import json


def main():
    output=HERE/'train16_assessment.json'
    check(not output.exists(),'exclusive fixed16 assessment')
    evidence=[]; parity=[]; failures=[]
    result=dict(passed=False,retained=False,evidence=evidence,layer0=parity,failures=failures,
                assessor=pin(Path(__file__).resolve()),no_estimator_execution=True,
                no_truth_reads=True,no_automatic_neighbors=True,release_qualification=False)
    try:
        p=read(HERE/'protocol.json'); admission=read(HERE/'train2_assessment.json'); build=read(HERE/'build.json')
        check(admission['passed'] and admission['advance'],'TRAIN2 admission')
        check(admission['candidate']==p['candidate'] and admission['protocol']==pin(HERE/'protocol.json'),'admission identity')
        for record in admission['evidence']+[admission['assessor'],p['candidate'],p['parent'],p['fragment']]:
            check(pin(Path(record['path']))==record,'changed input pin')
        check(build['candidate']==p['candidate'] and build['protocol']==pin(HERE/'protocol.json') and build['passed'],'build identity')
        blocks=[[35,100],*p['if_advance_fixed_blocks']]
        check(blocks==[[35,100],[36,45,52,58,64,70],[3,12,40,41,43,44,57,68]],'fixed roster/order')
        tags=['hill_climb_star_consumer_residual_train2_v2','hill_climb_star_consumer_residual_train6_v2',
              'hill_climb_star_consumer_residual_extra8_v2']
        evidence.extend([pin(HERE/'protocol.json'),pin(HERE/'build.json'),pin(HERE/'train2_assessment.json'),pin(HERE/'assess_screen.py')])
        manifests=[[],[]]; rows=[[],[]]
        for i,ranks in enumerate(blocks):
            seconds=[p['limits']['train2_seconds'],p['limits']['train6_seconds'],p['limits']['extra8_seconds']][i]
            own_guard=['train2_guard.json','train6_guard.json','extra8_guard.json'][i]
            base_guard=['train2_guard.json','train6_guard.json','extra8_candidate_guard.json'][i]
            evidence.append(guard(HERE/own_guard,p['candidate'],tags[i],ranks,seconds))
            evidence.append(guard(H2/'hill_climb_star_projection32_v1'/base_guard,p['parent'],p['controls'][i],ranks,seconds))
            bm,br,bp=load_run(p['controls'][i],p['parent'],ranks,evidence)
            cm,cr,cp=load_run(tags[i],p['candidate'],ranks,evidence)
            manifests[0].append(bm);manifests[1].append(cm);rows[0].extend(br);rows[1].extend(cr)
            for rank in ranks:
                files=[bp/f'pred_{rank}.npy',cp/f'pred_{rank}.npy']; evidence.extend(pin(path) for path in files)
                b,c=[np.load(path,allow_pickle=False) for path in files]
                check(b.shape==c.shape==(16,1024) and b.dtype==c.dtype==np.dtype('float64'),'prediction layout')
                check(np.isfinite(b).all() and np.isfinite(c).all(),'finite all16')
                exact=b[0].tobytes()==c[0].tobytes();parity.append(dict(rank=rank,exact=exact))
                check(exact,'L0 byte parity')
        comparisons=[]
        for name,offset in [('train16_comparison.json',0),('additional14_comparison.json',1)]:
            c=read(HERE/name);evidence.append(pin(HERE/name));comparisons.append(c)
            check(c['base_sources']==manifests[0][offset:] and c['candidate_sources']==manifests[1][offset:],'comparison source lists')
            base=rows[0][2*offset:];cand=rows[1][2*offset:]
            check(c['ranks']==sorted(row['rank'] for row in base) and c['n']==len(base),'comparison roster')
            for field,key in [('raw_pct','mse'),('adjusted_pct','adjusted'),('mean_util_pct','util'),('mean_cpu_pct','cpu_s')]:
                value=100*(sum(r[key] for r in cand)/sum(r[key] for r in base)-1)
                check(math.isfinite(c[field]) and math.isclose(value,c[field],abs_tol=1e-10),'comparison '+field)
            check(c['adjusted_wins']==sum(y['adjusted']<x['adjusted'] for x,y in zip(base,cand)),'comparison wins')
            ci=c['adjusted_ci95_pct'];check(len(ci)==2 and all(math.isfinite(x) for x in ci) and ci[0]<=ci[1],'finite ordered CI')
        overall,additional=comparisons;gate=p['train16_retention']
        check(gate==dict(adjusted_pct_at_most=-.1,wins_min=10,ci95_upper_below=0,additional14_gain_positive=True),'unchanged fixed16 gates')
        decisions=dict(adjusted=overall['adjusted_pct']<=gate['adjusted_pct_at_most'],
                       wins=overall['adjusted_wins']>=gate['wins_min'],
                       ci=overall['adjusted_ci95_pct'][1]<gate['ci95_upper_below'],
                       additional14=additional['adjusted_pct']<0)
        result.update(passed=True,retained=all(decisions.values()),candidate=p['candidate'],parent=p['parent'],
            protocol=pin(HERE/'protocol.json'),comparison=pin(HERE/'train16_comparison.json'),
            additional14=pin(HERE/'additional14_comparison.json'),metrics=overall,
            additional14_adjusted_pct=additional['adjusted_pct'],gate=gate,decisions=decisions)
    except Exception as exc:
        failures.append(dict(type=type(exc).__name__,message=str(exc)))
    with output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({key:result[key] for key in ['passed','retained','failures']}))
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
