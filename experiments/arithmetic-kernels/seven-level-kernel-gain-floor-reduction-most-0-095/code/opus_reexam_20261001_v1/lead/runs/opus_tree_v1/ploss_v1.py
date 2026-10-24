"""P(loss) per 100 networks exactly as designation_judge --primary (loss_model_v3, first priced, 6 first / 94 warm)."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.stats import genpareto as g
R=Path(__file__).resolve().parents[3]/'release'
x=np.array(json.loads((R/'runs'/'net_factors_20261011b.json').read_text())['ours']);e=x[x>1.10]-1.10;c,_,s=g.fit(e,floc=0)
S=lambda t:len(e)/len(x)*g.sf(t-1.10,c,0,s)
def p(first,warm):
    mw=warm*7.35e-6;return 1-(1-S(0.4/(mw*first/warm)))**6*(1-S(0.4/mw))**94*(1-2/359)
if __name__=='__main__':
    for a in sys.argv[1:]:
        f,w=map(int,a.split(','));print('first %6d warm %6d -> P(loss) %.2f%%'%(f,w,100*p(f,w)))
