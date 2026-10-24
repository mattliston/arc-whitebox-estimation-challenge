import json,os,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent
p=json.loads((D/'ROOT_GO_gradient268_v1.json').read_text())
e=dict(os.environ,TRAINER_THREADS='1',TRAINER_PL='1',TRAINER_CARD='M',TRAINER_RELIEF='1',TRAINER_STAR_ATTACH='0',TRAINER_BIRTH_ATTACH='0')
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):e[k]='1'
with(D/'LOG_gradient268_v1.txt').open('x')as f:r=subprocess.run(p['command'],env=e,stdout=f,stderr=subprocess.STDOUT)
print('owned_gradient_exit',r.returncode,flush=True)
raise SystemExit(r.returncode)
