"""One explicitly frozen production-boundary session; no source mutation."""
from pathlib import Path
import argparse,hashlib,json,os,re,subprocess,sys,time
D=Path(__file__).resolve().parent;R=D.parents[1];O=R.parents[1];P=O.parent
parser=argparse.ArgumentParser();parser.add_argument('source',type=Path);parser.add_argument('sha');parser.add_argument('ranks');parser.add_argument('tag');parser.add_argument('--parity-receipt',type=Path,required=True);parser.add_argument('--clock',choices=('short','actual'),default='short');args=parser.parse_args()
source=args.source.resolve();assert hashlib.sha256(source.read_bytes()).hexdigest()==args.sha
ranks=[int(x)for x in args.ranks.split(',')];assert ranks==[35,100]or(len(ranks)==1 and ranks[0]in(300,310,312,316,320,328,333))
assert args.parity_receipt.is_file();assert Path(args.tag).name==args.tag
out=D/args.tag;out.mkdir();free=int(re.search(r'System-wide memory free percentage: (\d+)%',subprocess.check_output(['/usr/bin/memory_pressure'],text=True,timeout=10)).group(1));assert free>=60,free
threads={k:'1'for k in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','NUMBA_NUM_THREADS','OMP_THREAD_LIMIT')};env=dict(os.environ,**threads)
light=Path('/Users/matt/arc-white-box-estimation/.venv/bin/python');native=P/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python';guard=R/'astra_crunch_physics_v1/guard_memory50_v1.py'
plan=dict(source=str(source),source_sha256=args.sha,ranks=ranks,clock=args.clock,parity_receipt=str(args.parity_receipt.resolve()),parity_receipt_sha256=hashlib.sha256(args.parity_receipt.read_bytes()).hexdigest(),footprint_limit=7500000000,rss_backstop=10000000000,seconds=150,entry_free_pct=free,hard_free_pct=50,source_unchanged=True)
(out/'PLAN_v1.json').write_text(json.dumps(plan,indent=2)+'\n')
command=[str(light),'-B',str(guard),'--record',str(out/'guard.json'),'--seconds','150','--footprint-limit','7500000000','--rss-backstop','10000000000',str(light),'-B',str(D/'boundary_run_v1.py'),str(source),args.ranks,str(out)]
if args.clock=='short':command.append('--short-setup')
print(json.dumps(dict(event='START',**plan)),flush=True)
with(out/'guard.log').open('x')as log:code=subprocess.call(command,env=env,stdout=log,stderr=subprocess.STDOUT)
assert hashlib.sha256(source.read_bytes()).hexdigest()==args.sha
g=json.loads((out/'guard.json').read_text());result=dict(event='COMPLETE',exit_code=code,guard_pass=bool(code==0 and g.get('completed')and g.get('whole_group_retired')),guard=g)
if (out/'rows.jsonl').exists()and(out/'manifest.json').exists():
 grade=subprocess.run([str(native),'-B',str(D/'grade_predictions_v1.py'),str(out)],env=env,capture_output=True,text=True)
 (out/'grade.log').write_text(grade.stdout+grade.stderr);result['grade_exit_code']=grade.returncode
 if grade.returncode==0:result['accuracy']=json.loads((out/'ACCURACY_v1.json').read_text())
(out/'SESSION_RESULT_v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
sys.exit(0 if result['guard_pass']else 1)
