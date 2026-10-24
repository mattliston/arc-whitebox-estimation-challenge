import subprocess,re,json,sys,hashlib
from pathlib import Path
D=Path(__file__).resolve().parent;O=D.parents[3];R=D.parents[1]
rank=int(sys.argv[1]);side=sys.argv[2];assert rank in(378,390,400,424)and side in('control','candidate')
pre=json.loads((D/'NATIVE_PREREG_v1.json').read_text());source=Path(pre['parent'if side=='control'else'candidate']);assert hashlib.sha256(source.read_bytes()).hexdigest()==pre['parent_sha'if side=='control'else'candidate_sha']
p=subprocess.check_output(['/usr/bin/memory_pressure'],text=True);free=int(re.search(r'System-wide memory free percentage: (\d+)%',p).group(1));assert free>=60,free
stem=f'{rank}_{side}_v1';guard=D/f'guard_{stem}.json';out=D/f'res_{stem}.json';assert not out.exists()and not guard.exists()
with (D/f'log_{stem}.txt').open('x')as f:
 c=subprocess.run([sys.executable,'-B',str(D.parent/'guard_memory50_v1.py'),'--record',str(guard),'--seconds','150','--footprint-limit','7500000000','--rss-backstop','7500000000',str(O.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'),'-B',str(D/'judge_v1.py'),str(source),str(rank),'ctl',str(out)],stdout=f,stderr=subprocess.STDOUT)
g=json.loads(guard.read_text());assert c.returncode==0 and g['completed']and g['whole_group_retired'],g
x=json.loads(out.read_text());print(json.dumps(dict(rank=rank,side=side,U=x['U'],flops=x['flops'],operations=sum(v['calls']for v in x['operations'].values()),entryfree=free,rss=g['max_rss_bytes'],retired=True)),flush=True)
a=D/f'res_{rank}_control_v1.json';b=D/f'res_{rank}_candidate_v1.json'
if a.exists()and b.exists():
 a=json.loads(a.read_text());b=json.loads(b.read_text());print(json.dumps(dict(rank=rank,raw_pct=100*(b['U']/a['U']-1),bill_pct=100*(b['flops']/a['flops']-1),adjusted_pct=100*(b['U']/a['U']*b['flops']/a['flops']-1),calls_delta=sum(v['calls']for v in b['operations'].values())-sum(v['calls']for v in a['operations'].values()),shape_equal=a['ott_shapes']==b['ott_shapes'])),flush=True)
