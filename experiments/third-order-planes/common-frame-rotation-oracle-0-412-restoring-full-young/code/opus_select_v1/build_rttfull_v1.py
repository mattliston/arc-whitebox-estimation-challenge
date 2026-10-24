"""ORACLE-ish bound (metered, exact): RTT2 young-read prefix truncation OFF (c15 = c2: every live column kept) via env SEL_RTTFULL=1."""
from pathlib import Path
import hashlib
D=Path(__file__).resolve().parent;R=D.parent
src=(R/'astra_smirnov_current24_virtual_v1/candidate_smirnov_arena_RPE_SPLIT_COV_HELD_v15.py').read_text()
assert hashlib.sha256(src.encode()).hexdigest().startswith('2b10843a')
o="    c15=min(c2,(int(fnp.nonzero(_E2_LAYER['g']>=floor)[0].shape[0])+31)//32*32)\n"
n="    c15=c2 if _SEL_RTTFULL else min(c2,(int(fnp.nonzero(_E2_LAYER['g']>=floor)[0].shape[0])+31)//32*32)\n"
i=src.rindex(o);src=src[:i]+n+src[i+len(o):]
src+="\n# BEGIN OPUS-SELECT rttfull switch (screen only)\nimport os as _rf_os\n_SEL_RTTFULL = _rf_os.environ.get('SEL_RTTFULL') == '1'\n# END OPUS-SELECT rttfull switch\n"
o2=D/'candidate_2b108_rttfull_v1.py';o2.open('x').write(src);compile(src,str(o2),'exec');print(hashlib.sha256(src.encode()).hexdigest(),'patched at char',i)
