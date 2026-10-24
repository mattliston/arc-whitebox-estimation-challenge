"""Translate the sealed production carrier; no numerical imports at entry."""
import ast,hashlib
from common import *
from model_bank import Translate
def environment(torch):
    paths=[BI/'carrier_fnp.py',D/'joint_remainder_carrier_implementation_v1/post_planes_fnp.py']
    body=[]
    for path in paths:
        tree=ast.parse(path.read_text())
        assert all(isinstance(n,(ast.Expr,ast.Assign,ast.FunctionDef)) for n in tree.body)
        body.extend(tree.body)
    translated=ast.fix_missing_locations(Translate().visit(ast.Module(body=body,type_ignores=[])))
    env={'_proposal_diagonal_write':lambda x,v:torch.diagonal_scatter(x,torch.as_tensor(v,dtype=x.dtype,device=x.device).expand(x.shape[0]))}
    exec(compile(translated,'sealed_carrier:offline_translation','exec'),env)
    env['translation_ast_sha256']=hashlib.sha256(ast.dump(translated,include_attributes=False).encode()).hexdigest()
    return env
