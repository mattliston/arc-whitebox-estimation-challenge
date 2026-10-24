"""[opus-lowres] single-change pricing vs LRB (warm_judge: first = 7103248, warm = 7103256): exact flops %, billed-op delta,
process instructions % (server-compute proxy), plain MSE delta (warm net). usage: grade_wj_v1.py TAG..."""
import json, math, sys
from pathlib import Path
D = Path(__file__).resolve().parent
b = json.loads((D/'wj/base_7103256.json').read_text()); bf, bw = b['first'], b['rows'][0]
print(f"base first flops {bf['flops']/1e9:.3f}G ops {bf['ops']} instr {bf['instructions']/1e9:.1f}G | warm flops {bw['flops']/1e9:.3f}G ops {bw['ops']} instr {bw['instructions']/1e9:.1f}G mse {bw['mse_final']:.4e}")
for t in sys.argv[1:]:
    p = D/f'wj/{t}_7103256.json'
    if not p.exists(): print(t, 'missing'); continue
    d = json.loads(p.read_text()); f, w = d['first'], d['rows'][0]
    print(f"{t:9s} first flops {100*(f['flops']/bf['flops']-1):+.3f}% ops {f['ops']-bf['ops']:+5d} instr {100*(f['instructions']/bf['instructions']-1):+5.1f}% | "
          f"warm flops {100*(w['flops']/bw['flops']-1):+.3f}% ops {w['ops']-bw['ops']:+5d} instr {100*(w['instructions']/bw['instructions']-1):+5.1f}% | warm mse {100*math.log(w['mse_final']/bw['mse_final']):+.3f}%")
