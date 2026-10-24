"""Original full-bank gates, native numerical maps and explicit lower bounds."""
from pathlib import Path
from tensor_clamp_min_v1 import tensor_clamp_min

here = Path(__file__).resolve().parent
source = (here/'check_centered_bank_baseline_v1.py').read_text()
source = source.replace('from centered_bank_autodiff_v1 import CenteredBankFullCard',
                        'from centered_bank_autodiff_v3 import ExplicitBoundBankFullCard as CenteredBankFullCard')
source = source.replace('centered_bank_baseline_{args.device}_{args.dtype}_v1.json',
                        'centered_bank_baseline_{args.device}_{args.dtype}_v3.json')
source = source.replace('centered_torch_numerics_{args.device}_small_v1.json',
                        'centered_torch_numerics_{args.device}_small_v3.json')
source = source.replace("HERE/'centered_torch_numerics_v1.py'","HERE/'centered_torch_numerics_v3.py'")
source = source.replace("HERE/'centered_bank_autodiff_v1.py'","HERE/'centered_bank_autodiff_v3.py'")
source = source.replace('state[0].clamp_min(0)','tensor_clamp_min(state[0],0.)')
exec(compile(source,__file__+':certificate','exec'),globals())
