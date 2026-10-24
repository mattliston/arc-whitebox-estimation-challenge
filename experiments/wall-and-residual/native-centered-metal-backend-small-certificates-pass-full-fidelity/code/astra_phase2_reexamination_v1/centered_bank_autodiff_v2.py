"""Same offline centered card with only the native CDF substitution."""
import centered_bank_autodiff_v1 as parent
from centered_torch_numerics_v2 import marginals,nonlinear,normal_cdf

_prefix_env = dict(parent._prefix_env,nonlinear=nonlinear,normal_cdf=normal_cdf)
_final_env = dict(parent._final_env,marginals=marginals)
exec(compile(parent.prefix_source,__file__+':prefix','exec'),_prefix_env)
exec(compile(parent.final_source,__file__+':final','exec'),_final_env)


class NativeCenteredBankFullCard(parent.CenteredBankFullCard):
    def step(self,l,state,control,P):
        fn = _prefix_env['step'] if l < self.depth-1 else _final_env['step']
        return fn(self,l,state,control,P)
