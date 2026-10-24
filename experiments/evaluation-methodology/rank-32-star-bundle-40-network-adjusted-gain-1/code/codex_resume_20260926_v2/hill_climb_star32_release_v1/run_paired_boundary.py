from pathlib import Path
import json,hashlib,subprocess,datetime
S=Path(__file__).resolve().parent;H=S.parent;P=H.parent;R=P.parents[3]
LIGHT=R/'.venv/bin/python';G=P/'codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py'
cfg=json.loads((S/'config.json').read_text());base=H/'candidate_old_both_tail_trace_bounded_v6.py'
pin=lambda p:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
assert pin(base)['sha256']=='5d43930820d144db00aac86ace9281fe0232240e321f53701cfdd8fe1ec98f26'
assert pin(Path(cfg['candidate']['path']))==cfg['candidate']
plan={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'purpose':'One fixed fresh S31 then frozen candidate boundary pair to check historical residual timing drift; preserve original failing +10ms screen. No candidate or threshold change.','control':pin(base),'candidate':cfg['candidate'],'ranks':[35,100],'order':['s31','candidate'],'no_automatic_retry':True,'legacy_assessment':pin(S/'qualification_assessment.json')}
with(S/'paired_boundary_protocol.json').open('x')as f:json.dump(plan,f,indent=2);f.write('\n')
for name,source in [('s31',plan['control']),('candidate',plan['candidate'])]:
 assert pin(Path(source['path']))==source
 tag=S.name+'_residual_'+name;print(json.dumps({'event':'starting','boundary':name}),flush=True)
 with(S/('paired_'+name+'.log')).open('x')as f:
  run=subprocess.run([str(LIGHT),'-B',str(G),'--record',str(S/('paired_'+name+'_guard.json')),'--seconds','240','--rss-limit','7500000000',str(LIGHT),'-B',str(H/'run_boundary_census.py'),source['path'],'35,100',tag],stdout=f,stderr=subprocess.STDOUT)
 g=json.loads((S/('paired_'+name+'_guard.json')).read_text());assert run.returncode==0 and g['completed'] and g['exit_code']==0 and g['error'] is None and g['whole_group_retired'] and not g['remaining_rows'] and g['max_swap_growth_bytes']==0 and g['max_rss_bytes']<7500000000
 print(json.dumps({'event':'completed','boundary':name,'seconds':g['elapsed_s'],'rss':g['max_rss_bytes']}),flush=True)
print('PAIRED_BOUNDARY_COMPLETE',flush=True)
