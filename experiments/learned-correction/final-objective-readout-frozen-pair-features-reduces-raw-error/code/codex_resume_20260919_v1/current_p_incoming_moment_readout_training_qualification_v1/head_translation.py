"""Translate exactly three diagonal writes to functional differentiable writes."""
import ast
import copy
from common import *
from model_bank import Translate
_DIAGONAL_LOOP=ast.parse('for x in (a,b,c): xp.fill_diagonal(x,0.)').body[0]

class HeadTranslate(Translate):
    def __init__(self):
        super().__init__();self.diagonal_loops=0
    def visit_For(self,node):
        if ast.dump(node,include_attributes=False)==ast.dump(_DIAGONAL_LOOP,include_attributes=False):
            self.diagonal_loops+=1
            return [ast.copy_location(ast.Assign(targets=[ast.Name(id=name,ctx=ast.Store())],
                value=ast.Call(func=ast.Name(id='_proposal_diagonal_write',ctx=ast.Load()),
                    args=[ast.Name(id=name,ctx=ast.Load()),ast.Constant(value=0.)],keywords=[])),node)
                for name in ('a','b','c')]
        return self.generic_visit(node)

def environment(torch):
    original=ast.parse((SMALL/'head.py').read_text())
    translator=HeadTranslate();tree=ast.fix_missing_locations(translator.visit(copy.deepcopy(original)))
    assert translator.diagonal_loops==1
    env=dict(_proposal_diagonal_write=lambda x,v:torch.diagonal_scatter(
        x,torch.as_tensor(v,dtype=x.dtype,device=x.device).expand(x.shape[0])))
    exec(compile(tree,str(SMALL/'head.py')+':torch_functional_translation','exec'),env)
    env['translation_ast_sha256']=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()
    return env
