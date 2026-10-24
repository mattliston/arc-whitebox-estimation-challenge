from pathlib import Path
import subprocess,sys,hashlib,json
D=Path(__file__).resolve().parent
source=D/'candidate_95c_re22cross_old0_HELD_v1.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='a509b450c0c70088527cb3f180e6e4ce60bff8772a4a745e8b1acd8f24036cb1'
for rank in (35,200):
 code=subprocess.call([sys.executable,str(D/'harness/run_guarded_rtt_v1.py'),str(source),str(rank),str(D/f'native_{rank}_v1')])
 print('retired_rank',rank,'exit',code,flush=True)
 if code:sys.exit(code)
(D/'ALL_RETIRED_v1.json').open('x').write(json.dumps({'fixed_ranks':[35,200],'exit_codes':[0,0]})+'\n')
