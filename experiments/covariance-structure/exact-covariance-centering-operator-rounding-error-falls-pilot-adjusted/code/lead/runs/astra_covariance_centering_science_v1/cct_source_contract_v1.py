"""Fixed hidden covariance-centering implementation contract."""
def cleanup(mod):
    return dict(active_clear=mod._CCT_ACTIVE==[False],on=mod._CCT_ON is True)
def gate(row):
    st=row['stats'];r=st['_CCT_STATS']
    assert r['retired'] is True and st['_CCT_ACTIVE']==[False]
    assert r['calls']==14 and r['layers']==list(range(1,15))
    assert len(r['rows'])==14 and all(type(m)is int and 0<m<=1024 and m%64==0 for m in r['rows'])
    assert row['cct_cleanup']==dict(active_clear=True,on=True)

