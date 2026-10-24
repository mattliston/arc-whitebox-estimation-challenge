"""Native explicit bound values; avoids local MPS scalar-bound cache reuse."""
import ast
import torch


def tensor_clamp_min(x,bound):
    floor = torch.full_like(x,bound)
    # Equality retains x, matching clamp_min's derivative1 at the boundary.
    return torch.where(x<floor,floor,x)


class _ExplicitClamp(ast.NodeTransformer):
    def __init__(self):
        self.count = 0

    def visit_Call(self,node):
        node = self.generic_visit(node)
        if isinstance(node.func,ast.Attribute) and node.func.attr=='clamp_min':
            assert len(node.args)==1 and not node.keywords
            self.count += 1
            return ast.copy_location(ast.Call(func=ast.Name(id='tensor_clamp_min',ctx=ast.Load()),
                                     args=[node.func.value,node.args[0]],keywords=[]),node)
        return node


def rewrite_clamps(source,filename):
    visitor = _ExplicitClamp()
    tree = visitor.visit(ast.parse(source))
    ast.fix_missing_locations(tree)
    return compile(tree,filename,'exec'),visitor.count
