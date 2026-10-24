"""2b108 + tree_r1 fragment. Writes once."""
from pathlib import Path
import hashlib
D=Path(__file__).resolve().parent;R=D.parent
s=(R/'astra_smirnov_current24_virtual_v1/candidate_smirnov_arena_RPE_SPLIT_COV_HELD_v15.py').read_bytes();assert hashlib.sha256(s).hexdigest().startswith('2b10843a')
fr=(D/'fragment_tree_r1_v1.py.txt').read_bytes();o=D/'candidate_2b108_treeR1_v1.py'
with o.open('xb') as f:f.write(s+fr)
compile(o.read_text(),str(o),'exec');print(hashlib.sha256(o.read_bytes()).hexdigest(),'fragment',hashlib.sha256(fr).hexdigest()[:8])
