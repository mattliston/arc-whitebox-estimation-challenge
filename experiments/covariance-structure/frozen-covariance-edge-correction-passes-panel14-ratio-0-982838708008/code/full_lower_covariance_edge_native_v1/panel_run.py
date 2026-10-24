"""Serial unchanged native endpoint on prospectively fixed larger rosters."""
import argparse,subprocess,sys
from common import HERE,D,RUNTIME,read,write,pin,verify,seal_check,guarded,sha
ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['panel','confirmation']);stage=ap.parse_args().stage
seal_check()
for p in read(HERE/'panel_sources_sealed.json')['pins']:verify(p)
assert read(HERE/('score.json' if stage=='panel' else 'panel_score.json'))['passed']
meta=read(HERE/'panel_inputs.json');rows=[]
for entry in meta[stage]:
    rank=entry['rank'];name=f'{stage}_{rank}';watch=HERE/f'{name}_watch.json';out=HERE/name
    command=[sys.executable,'-B',str(D/'run_guarded.py'),'--record',str(watch),'--seconds','120',
             str(RUNTIME),'-B','-W','ignore',str(D/'weight5_reuse_v1/native.py'),str(HERE/'candidate.py'),str(out),'--ranks',str(rank)]
    print(dict(starting=name),flush=True)
    with (HERE/f'{name}.log').open('x') as log:r=subprocess.run(command,cwd=D,stdout=log,stderr=subprocess.STDOUT)
    assert r.returncode==0,(name,r.returncode)
    g=guarded(name);receipt=read(out/'result.json');assert receipt['candidate_sha256']==sha(HERE/'candidate.py') and not receipt['truth_accessed']
    call=receipt['records'][0];assert call['rank']==rank and call['includes_initialization'] and call['weight5_cache']['numeric_cache_retired']
    rows.append(dict(**entry,call=call,result=pin(out/'result.json'),watch=pin(watch)))
    print(dict(completed=name,flops=call['flops'],wall_s=call['wall_s'],peak_group_rss=g['max_rss_bytes']),flush=True)
write(HERE/f'{stage}_predictions_sealed.json',dict(completed=True,rows=rows,source=pin(HERE/'candidate.py'),
    sources=pin(HERE/'panel_sources_sealed.json'),inputs=pin(HERE/'panel_inputs.json'),truth_arrays_accessed=False))
name=f'{stage}_score';command=[sys.executable,'-B',str(D/'run_guarded.py'),'--record',str(HERE/f'{name}_watch.json'),
    '--seconds','60',str(RUNTIME),'-B',str(HERE/'panel_score.py'),stage]
with (HERE/f'{name}.log').open('x') as log:r=subprocess.run(command,cwd=D,stdout=log,stderr=subprocess.STDOUT)
assert r.returncode==0;(guarded(name))
s=read(HERE/f'{name}.json');print({k:s[k] for k in ['passed','pooled_adjusted_ratio','wins','worst_adjusted_ratio']},flush=True)
