"""Original unchanged full-bank gate, with native-CDF centered arithmetic."""
from pathlib import Path

here = Path(__file__).resolve().parent
source = (here/'check_centered_bank_baseline_v1.py').read_text()
source = source.replace('from centered_bank_autodiff_v1 import CenteredBankFullCard',
                        'from centered_bank_autodiff_v2 import NativeCenteredBankFullCard as CenteredBankFullCard')
source = source.replace('centered_bank_baseline_{args.device}_{args.dtype}_v1.json',
                        'centered_bank_baseline_{args.device}_{args.dtype}_v2.json')
source = source.replace('centered_torch_numerics_{args.device}_small_v1.json',
                        'centered_torch_numerics_{args.device}_small_v2.json')
source = source.replace("HERE/'centered_torch_numerics_v1.py'","HERE/'centered_torch_numerics_v2.py'")
source = source.replace("HERE/'centered_bank_autodiff_v1.py'","HERE/'centered_bank_autodiff_v2.py'")
exec(compile(source,__file__+':certificate','exec'),globals())
