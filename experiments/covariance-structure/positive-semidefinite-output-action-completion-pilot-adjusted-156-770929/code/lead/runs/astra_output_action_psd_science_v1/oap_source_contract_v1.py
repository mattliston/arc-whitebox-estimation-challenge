def cleanup(mod):
    return dict(state_empty=mod._OAC_STATE=={}, active_clear=mod._OAC_ACTIVE==[False], sides_empty=mod._OAC_SIDES=={})
def gate(row):
    st=row['stats']['_OAC_STATS']
    assert row['ocg_cleanup']==dict(state_empty=True,active_clear=True,sides_empty=True)
    assert row['stats']['_OAC_ACTIVE']==[False] and st['retired'] is True
    assert [x['layer'] for x in st['calls']]==list(range(1,15))
    changed=[]
    for x in st['calls']:
        assert x['threshold']==.05
        assert 64<=x['input']<=1024 and x['input']%64==0
        assert 64<=x['output']<=1024 and x['output']%64==0
        assert x['changed']==(x['output']<1024)
        assert x['preserved_diagonal']==x['changed']
        if x['changed']:changed.append(x['layer'])
    assert changed
    assert st['feature_layers']==changed
    assert st['np_layers']==changed
