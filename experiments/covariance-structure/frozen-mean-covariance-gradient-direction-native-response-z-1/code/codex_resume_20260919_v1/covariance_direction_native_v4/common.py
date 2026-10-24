import ast, hashlib, importlib.util, json, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
D=HERE.parent
P2=D.parent
A=P2/'astra_phase2_reexamination_v1'
SOURCE=D/'lazy_rotation_objective_v1/candidate.py'
MODEL=A/'current_comp_joint_p_covariance_v1/model.py'
FIT=(35,36,38,40)
H=1/32
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def meta(path):
    p=Path(path).resolve();return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);sys.modules[name]=obj;spec.loader.exec_module(obj);return obj
def check():
    seal=read(HERE/'sources_sealed.json')
    for row in seal['sources']+seal['external_pins']:
        assert sha(row['path'])==row['sha256'],row['path']
    return seal
def literals(source):
    wanted={'_GNN_LITERAL','_BDNH_A_LITERAL','_BD_COEFFICIENT_LITERAL'}
    return {n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(source).body
            if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)
            and n.targets[0].id in wanted}
