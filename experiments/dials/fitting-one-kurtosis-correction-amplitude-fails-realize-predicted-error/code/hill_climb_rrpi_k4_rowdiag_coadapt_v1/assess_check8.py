"""Root-only replay of exact native CHECK8 bills and fitted-head comparisons."""
from pathlib import Path
import importlib.util,json,os,sys

D=Path(__file__).resolve().parent


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    assert len(sys.argv)==3
    b=module(D/'prepare_check8.py','coadapt_check8_preflight')
    pp,c=b.configuration(Path(sys.argv[1]).resolve(),sys.argv[2])
    f=module(c['helper']['path'],'coadapt_check8_panels');q=module(c['engine']['path'],'coadapt_check8_metrics')
    assert str(Path(sys.executable).absolute())==c['runtime']==str(q.NATIVE)
    assert q.np.__version__=='2.4.6' and all(os.environ.get(k)=='1' for k in q.THREADS)
    truth=f.truths(q,c,'check8');control=c['controls']['check8']
    assert control['arm']['source']==c['parent']
    base,ba,bp=f.panel(q,c,control['arm'],c['ranks'],truth,control['artifacts'])
    rows={};metrics={};artifacts={'original_rrpi':bp};parity={}
    for arm,a in c['arms'].items():
        rows[arm],arrays,artifacts[arm]=f.panel(q,c,a,c['ranks'],truth)
        count=15 if arm=='rrpi' else 2
        for rank in c['ranks']:assert arrays[rank][:count].tobytes()==ba[rank][:count].tobytes()
        parity[arm]=list(range(count));metrics[arm+'_vs_original']=q.metrics(base,rows[arm])
    if 'k4' in rows:metrics['k4_vs_fitted_rrpi']=q.metrics(rows['rrpi'],rows['k4'])
    def positive(m):return m['raw_pct']<0 and m['adjusted_pct']<0
    eligible={arm:bool(c['binding']['exports'][arm]['eligible_standalone'] and positive(metrics[arm+'_vs_original'])) for arm in rows}
    if 'k4' in rows:eligible['k4']=bool(eligible['k4'] and positive(metrics['k4_vs_fitted_rrpi']))
    result=dict(schema='coadapt-native-check8-assessment-v1',passed_integrity=True,plan=pp,assessor=q.pin(__file__),
        parent=c['parent'],candidates={k:v['source'] for k,v in c['arms'].items()},binding=c['binding'],ranks=c['ranks'],
        artifacts=artifacts,metrics=metrics,exact_parity_layers=parity,eligible_arms=eligible,
        actual_candidate_bills=True,all16_finite=True,wins_and_ci_descriptive=True,
        adaptive_reused_train_check_not_independent_validation=True,
        full40_own_resources_and_serving_still_required=True,bank_qualified=False,submitted=False)
    q.once(D/'check8_assessment.json',result)
    print(json.dumps({k:result[k] for k in ('passed_integrity','metrics','eligible_arms')}))


if __name__=='__main__':main()
