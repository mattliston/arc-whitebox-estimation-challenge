"""Current numerical bank with MPS-safe explicit lower bounds everywhere."""
import ast
import inspect
import textwrap
import centered_bank_autodiff_v1 as parent
from centered_torch_numerics_v3 import marginals,nonlinear,normal_cdf
from tensor_clamp_min_v1 import tensor_clamp_min,rewrite_clamps


def fixed_function(fn,source_path=None):
    if source_path is None:
        source = textwrap.dedent(inspect.getsource(fn))
    else:
        # shadow_v3 compiles v1 source under its short wrapper filename;
        # inspect cannot recover those longer virtual source line numbers.
        original = source_path.read_text()
        matches = [n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef) and n.name==fn.__name__]
        assert len(matches)==1
        n = matches[0]
        source = '\n'.join(original.splitlines()[n.lineno-1:n.end_lineno])
    code,count = rewrite_clamps(source,__file__+':'+fn.__name__)
    env = dict(fn.__globals__,tensor_clamp_min=tensor_clamp_min)
    exec(code,env)
    return env[fn.__name__],count


node_features,feature_clamps = fixed_function(parent.node_features,parent.HERE/'current_transport_shadow_v1.py')
gnn,gnn_clamps = fixed_function(parent.gnn,parent.HERE/'current_transport_shadow_v1.py')
scalar_features,scalar_feature_clamps = fixed_function(parent.scalar_features)
final_read,final_read_clamps = fixed_function(parent.final_read)
_prefix_env = dict(parent._prefix_env,nonlinear=nonlinear,normal_cdf=normal_cdf,
                   node_features=node_features,gnn=gnn,tensor_clamp_min=tensor_clamp_min)
_final_env = dict(parent._final_env,marginals=marginals,scalar_features=scalar_features,
                  final_read=final_read,tensor_clamp_min=tensor_clamp_min)
pc,prefix_clamps = rewrite_clamps(parent.prefix_source,__file__+':prefix')
fc,final_clamps = rewrite_clamps(parent.final_source,__file__+':final')
exec(pc,_prefix_env)
exec(fc,_final_env)
CLAMP_COUNTS = dict(prefix=prefix_clamps,final=final_clamps,features=feature_clamps,
                    gnn=gnn_clamps,scalar_features=scalar_feature_clamps,final_read=final_read_clamps)
assert prefix_clamps==2 and final_clamps==0 and feature_clamps==5 and gnn_clamps==3 and final_read_clamps==3


class ExplicitBoundBankFullCard(parent.CenteredBankFullCard):
    def step(self,l,state,control,P):
        fn = _prefix_env['step'] if l < self.depth-1 else _final_env['step']
        return fn(self,l,state,control,P)
