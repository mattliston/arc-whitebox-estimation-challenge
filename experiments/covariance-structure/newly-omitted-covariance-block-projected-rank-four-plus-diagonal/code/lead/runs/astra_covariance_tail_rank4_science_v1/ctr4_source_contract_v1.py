"""Frozen CV4 implementation checks, no scientific scoring."""
def cleanup(mod):
    constants = mod._CV4_CONST
    return dict(active_clear=mod._CV4_ACTIVE == [False],
                probe_owner_registered='_CV4_CONST' in mod._CS_DICTS,
                probe_shape_valid=(not constants or (tuple(constants['probes'].shape)==(1024,4) and str(constants['probes'].dtype)=='float32')))

def gate(row):
    st=row['stats'];cv=st['_CV4_STATS'];calls=cv['calls']
    assert cv['retired'] is True and st['_CV4_ACTIVE']==[False]
    assert [x['layer'] for x in calls]==list(range(1,16))
    for x in calls:
        assert type(x['m0']) is int and type(x['m1']) is int
        assert 0<=x['m1']<=x['m0']<=1024 and x['m0']%64==x['m1']%64==0
        assert type(x['changed']) is bool and type(x['fallback']) is bool and type(x['terminal']) is bool
        if x['changed']:
            assert x['rank']==4 and x['m1']>=64 and x['m0']-x['m1']>=4 and not x['fallback']
            assert x['terminal']==(x['layer']==15)
        else:
            assert x['rank']==0 and not x['terminal']
            assert x['fallback']==(x['m1']!=x['m0'])
    assert row['ctr4_cleanup']==dict(active_clear=True,probe_owner_registered=True,probe_shape_valid=True)
    rr=st['_RTT_STATS'];assert rr['retired'] is True
    assert [x['layer'] for x in rr['reads']]==list(range(1,15))
    for x in rr['reads']:
        if x['changed']:
            assert x['old_inner_calls']==int(x['old_rank']>0)
