"""Preserved trajectory/targets, unique positive-resummation source test."""
import hashlib
from pathlib import Path

here=Path(__file__).resolve().parent;parent=here/'check_cumulative_backward_source_v1.py'
source=parent.read_text()
changes=(
    ('from cumulative_backward_source_v1 import source_slices,certificate','from cumulative_positive_path_source_v1 import source_slices,certificate'),
    ('cumulative_backward_source_r35_l12_v1','cumulative_positive_path_source_r35_l12_v1'),
    ("cache.update(g=aux['gate']*alive,d=d*alive,e=-aux['alpha']/aux['sigma']*d*alive)",
     """cache.update(g=aux['gate']*alive,d=d*alive,e=-aux['alpha']/aux['sigma']*d*alive)
        v=aux['sigma']**2
        cache.update(K21=K21.copy(),K22=cache['radial']*(np.outer(v,v)+2*cache['C']*cache['C'])/3,
                     eta=-aux['alpha']/aux['sigma'])"""),
    ("'Bpost':v['Bpost'].copy()}","'Bpost':v['Bpost'].copy(),'births':v['births'].copy()}"),
    ("v['signal_error_ratio']<=.65", "v['signal_error_ratio']<=.514161982572145"),
    ("v['A_ratio']<=.7 and v['B_ratio']<=.7", "v['A_ratio']<=.55 and v['B_ratio']<=.55"),
    ('record=dict(rank=35,','record=dict(parent_source_sha256=parent_sha,rank=35,'),
    ('current radial metric defect. Exact full-edge',
     'current radial metric defect and triangle; complete 24 path terms replaced by positive rank-one resummed actual selected births. Exact full-edge'),
)
for old,new in changes:
    assert source.count(old)==1,(old,source.count(old));source=source.replace(old,new)
exec(compile(source,str(parent),'exec'),dict(__file__=__file__,__name__='__main__',
                                          parent_sha=hashlib.sha256(parent.read_bytes()).hexdigest()))
