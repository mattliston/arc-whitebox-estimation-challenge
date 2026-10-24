"""Reprice the preserved exact-fringe DP with certified new circuit counts."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'smirnov_linear_circuit_envelope_v1.json'
assert not OUT.exists()
source = (HERE / 'price_mixed_fmm_envelope_v1.py').read_text()
source = source.replace("OUT=HERE/'mixed_fmm_envelope_v1.json'", "OUT=HERE/'smirnov_linear_circuit_envelope_v1.json'")
needle = "programs.append(dict(name='bank_winograd'"
prefix, suffix = source.split(needle, 1)
source = prefix + '''
optimized = json.loads((HERE / 'smirnov_linear_circuit_v1.json').read_text())
assert optimized['passed']
ca, cb, cc = (optimized['optimized_arithmetic'][s] for s in ('A', 'B', 'C'))
dual = [((3,3,6), (ca,cb,cc)),
        ((3,6,3), (cb,cc+18-40,ca+40-9)),
        ((6,3,3), (cc+18-40,ca,cb+40-18))]
for index, (shape, arithmetic) in enumerate(dual):
    programs.append(dict(name='signed_circuit_'+str(index), shape=shape,
                         rank=40, arithmetic=arithmetic))
    programs.append(dict(name='signed_circuit_'+str(index)+'_transpose',
                         shape=tuple(reversed(shape)), rank=40,
                         arithmetic=(arithmetic[1],arithmetic[0],arithmetic[2])))
''' + needle + suffix
# Reuse exactly the prior authored fringe identity, not downloaded code.
peeler = (HERE / 'price_mixed_fmm_peel_v2.py').read_text()
fringe = peeler.split("source=source.replace(needle,needle+'''", 1)[1].split("''')", 1)[0]
needle = "                options.append((p['rank']*cost+linear,p['rank']*memory,p['name']+' / '+child))"
assert source.count(needle) == 1
source = source.replace(needle, needle + fringe)
source = source.replace("qualification='Bounded optimistic arithmetic envelope:",
                        "qualification='Includes exact signed/scaled primary circuit and linear-transposition arithmetic envelopes, plus exact classical fringe peeling. Bounded optimistic arithmetic envelope:")
exec(compile(source, str(HERE / 'price_mixed_fmm_envelope_v1.py'), 'exec'),
     dict(__file__=str(HERE / 'price_mixed_fmm_envelope_v1.py'), __name__='__main__'))
