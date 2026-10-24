"""Positive-resummed births with the bank's unchanged scalar final read."""
import compact_fullpair_final_reference_v1 as parent
from positive_path_birth_reference_v1 import resummed_columns

source=parent.source
edits=(
    ('gate_floor=.01, final_hook=None):','gate_floor=.01, final_hook=None, path_mode=True):'),
    ('        bp = g[:, None]*Cpre','        bp,path_stats = resummed_columns(Cpre,K21,k22,aux,radial,path_mode)'),
    ('states.append(dict(layer=l, active=int(alive.sum()), births=len(selected), columns=p.shape[1]))',
     'states.append(dict(layer=l, active=int(alive.sum()), births=len(selected), columns=p.shape[1],**path_stats))'))
for old,new in edits:
    assert source.count(old)==1;source=source.replace(old,new)
env=dict(parent.env,resummed_columns=resummed_columns)
exec(compile(source,__file__,'exec'),env)
predict=env['predict']
