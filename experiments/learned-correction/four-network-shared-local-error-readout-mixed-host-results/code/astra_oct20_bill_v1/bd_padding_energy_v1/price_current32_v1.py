"""Bound exact0.12.3 selector debit using current777a32 birth counts.

Only synthetic1024-square base arrays and extracted selector helpers execute;
no estimator import/setup/prediction and no truth or network data are used.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,ast,hashlib,json,resource,subprocess,sys,threading,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;R=HERE.parents[1];V=R/'astra_oct20_vm_frontier_v1';CENSUS=V/'science32_msg8_loaderfix_parallel5_v1';OFFICIAL=R/'astra_crunch_runtime_v1/failure336631_v1/flopscope_0.12.3'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def guard():
    while True:
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>1_000_000_000:os._exit(97)
        time.sleep(.1)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();assert not a.out.exists();threading.Thread(target=guard,daemon=True).start()
    m=json.loads((HERE/'CANDIDATES_v1.json').read_text());source=Path(m['candidates'][0]['science']['parent_path']);assert sha(source)==m['parent_source_sha256']
    sys.path.insert(0,str(OFFICIAL));import numpy as np;import flopscope as fs;import flopscope.numpy as fnp
    assert fs.__version__.startswith('0.12.3') and not os.environ.get('FLOPSCOPE_WEIGHTS_FILE')
    tree=ast.parse(source.read_text());select=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='select'][-1]
    env=dict(fnp=fnp,_E2_LAYER={});exec(compile(ast.Module(body=[select],type_ignores=[]),'<original select>','exec'),env);exec(compile((HERE/'selector_v1.py').read_text(),'<candidate helpers>','exec'),env)
    def measure(call):
        with fs.BudgetContext(flop_budget=2**60,quiet=True) as b:y=call()
        return dict(flops=int(b.flops_used),charged_calls=len(b._op_log),shape=list(y[0].shape))
    with fs.BudgetContext(flop_budget=2**60,quiet=True):
        base=fnp.asarray(np.random.default_rng(202610210245).normal(size=(1024,1024)).astype('f'));aa=fnp.ones(1024);tt=fnp.ones(1024);ss=fnp.zeros(1024)
    def candidate(score):
        ij,mask=env['_o21_birth_prepare'](score,.01);env['_o21_birth_refine'](base,aa,tt,ss,mask);return env['_o21_birth_finish'](ij,mask)
    rows=[];pins=[];counts=set()
    for path in sorted(CENSUS.glob('candidate_part*/rows.jsonl')):
        assert json.loads((path.parent/'manifest.json').read_text())['sha256']==m['parent_source_sha256'];pins.append(dict(path=str(path),sha256=sha(path)));previous=[]
        for line in path.read_text().splitlines():
            row=json.loads(line);all_shapes=row['jun_stats']['shapes'];assert all_shapes[:len(previous)]==previous;births=[int(s[3])for s in all_shapes[len(previous):]];previous=all_shapes;assert len(births)==15
            rows.append(dict(rank=row['rank'],parent_flops=row['flops'],birth_counts=births));counts.update(births)
    prices={}
    for count in sorted(counts):
        cases=[]
        for padding in (0,1,31):
            required=count-padding
            if required<0:continue
            score=np.linspace(0,.009,1024);score[:required]=np.linspace(.02,.49,required)
            with fs.BudgetContext(flop_budget=2**60,quiet=True):sf=fnp.asarray(score)
            old,new=measure(lambda:env['select'](sf,.01)),measure(lambda:candidate(sf))
            assert old['shape']==new['shape']==[count]
            cases.append(dict(required=required,padding=padding,old=old,new=new,additional_flops=new['flops']-old['flops'],additional_calls=new['charged_calls']-old['charged_calls']))
        prices[count]=dict(count=count,cases=cases,max_additional_flops=max(x['additional_flops'] for x in cases),max_additional_calls=max(x['additional_calls'] for x in cases))
    for row in rows:
        extra=sum(prices[n]['max_additional_flops'] for n in row['birth_counts']);calls=sum(prices[n]['max_additional_calls'] for n in row['birth_counts'])
        row.update(max_additional_flops=extra,max_additional_calls=calls,max_bill_delta_pct=100*extra/row['parent_flops'],raw_break_even_upper_pct=100*(row['parent_flops']/(row['parent_flops']+extra)-1),max_padding_units=sum(31 for n in row['birth_counts']if n not in (0,1024)))
    assert len(rows)==32
    receipt=dict(utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),status='OFFICIAL_SELECTOR_DEBIT_UPPER_BOUND_ON_ACTUAL777A32_COUNTS',reference_submission=337422,source_sha256=sha(source),selector_sha256=sha(HERE/'selector_v1.py'),script_sha256=sha(__file__),official_version=fs.__version__,census_pins=pins,shape_prices=list(prices.values()),records=sorted(rows,key=lambda x:x['rank']),mean_max_bill_delta_pct=sum(x['max_bill_delta_pct']for x in rows)/32,max_bill_delta_pct=max(x['max_bill_delta_pct']for x in rows),mean_max_raw_break_even_pct=sum(x['raw_break_even_upper_pct']for x in rows)/32,max_call_delta=max(x['max_additional_calls']for x in rows),direct_compute_saving_ceiling_pct=0.,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,caveat='Existing rows omit pre-round threshold cardinalities, so each nonfull roster is charged for worstpadding31. Exact-multiple rosters incurzeroextra. Large-product counts are unchanged at this selection; changed predictions can alter later rosters. No accuracy gain is claimed.')
    with a.out.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(json.dumps({k:receipt[k]for k in('status','mean_max_bill_delta_pct','max_bill_delta_pct','mean_max_raw_break_even_pct','max_call_delta','direct_compute_saving_ceiling_pct','peak_rss_bytes')},indent=2))
if __name__=='__main__':main()
