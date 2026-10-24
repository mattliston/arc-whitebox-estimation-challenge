def cleanup(mod):
    return dict(pending_clear=mod._RE22_PENDING==[None],active_clear=mod._RE22_ACTIVE==[False])
def rpe_gate(row):
    st=row['stats']['_RE22_STATS']
    assert row['re22_cleanup']==dict(pending_clear=True,active_clear=True)
    assert row['stats']['_RE22_ACTIVE']==[False] and st['retired'] is True
    changed=[r for r in row['stats']['_RTT_STATS']['reads'] if r['changed']]
    assert changed
    assert [r['layer']for r in st['reads']]==[r['layer']for r in changed]
    assert st['feature_layers']==[r['layer']for r in changed]
    for r,b in zip(st['reads'],changed):
        assert 1<=r['layer']<=14 and r['m']==b['m']
        assert r['prefix']==b['c15'] and r['tail']==b['c2']-b['c15']>0
        assert r['qrank']==b['refine_rank_out'] and r['qrank']in (0,1,2)

PLAN_KEYS=frozenset(((1024, 4096, 960, 0), (928, 8128, 896, 0), (864, 9984, 832, 0), (768, 7168, 800, 384), (736, 8704, 768, 384), (768, 10176, 800, 384), (672, 6144, 768, 320), (576, 5376, 704, 384), (992, 6144, 928, 0), (864, 9984, 864, 0), (832, 11712, 832, 0), (736, 8832, 768, 384), (736, 10304, 768, 384), (672, 6208, 768, 320), (640, 7552, 736, 320), (576, 8832, 736, 320), (576, 5248, 736, 384), (960, 6144, 928, 0), (928, 8064, 896, 0), (864, 9920, 832, 0), (832, 11648, 832, 0), (736, 8768, 768, 384), (672, 10240, 768, 384), (608, 7360, 768, 320), (512, 4992, 704, 384), (480, 6016, 672, 384), (928, 8128, 864, 0), (768, 7232, 800, 384), (736, 10240, 768, 384), (704, 6144, 768, 320), (896, 9984, 864, 0), (832, 11776, 832, 0), (736, 8896, 768, 384), (704, 10368, 768, 384), (576, 5312, 704, 384), (960, 6144, 896, 0), (896, 8064, 896, 0), (864, 9856, 832, 0), (832, 11584, 832, 0), (768, 8704, 800, 384), (704, 7552, 768, 320), (640, 8960, 768, 320), (576, 5440, 704, 384), (960, 8128, 896, 0), (896, 10048, 864, 0), (832, 11840, 832, 0), (768, 8960, 800, 384), (736, 10496, 768, 384), (672, 6272, 768, 320), (576, 5312, 736, 384), (544, 6464, 672, 384), (768, 8960, 832, 384), (736, 6272, 768, 320), (672, 7744, 768, 320), (640, 9088, 736, 320), (608, 5568, 768, 384), (960, 8128, 928, 0), (768, 10560, 800, 384), (736, 6400, 768, 320), (672, 7872, 736, 320), (576, 6848, 736, 384), (928, 8064, 864, 0), (864, 9920, 864, 0), (672, 6080, 736, 320), (704, 10240, 768, 384), (672, 7488, 736, 320), (672, 8832, 768, 320), (864, 11776, 832, 0), (704, 10496, 768, 384), (576, 5184, 736, 384), (768, 10496, 800, 384), (736, 6336, 768, 320), (704, 9216, 768, 320), (896, 9920, 864, 0), (704, 8768, 768, 384), (672, 10176, 768, 384), (576, 5056, 704, 384), (768, 7104, 800, 384), (704, 8640, 768, 384), (736, 10048, 768, 384), (640, 6016, 768, 320), (672, 8640, 768, 384), (672, 9984, 768, 384), (576, 7040, 672, 320), (512, 5952, 704, 384), (736, 7168, 768, 384), (736, 8640, 768, 384), (672, 10112, 736, 384), (672, 5888, 768, 320), (640, 7232, 768, 320), (576, 5120, 704, 384), (768, 8768, 768, 384), (640, 7488, 768, 320), (672, 6336, 768, 320), (672, 10176, 736, 384), (672, 5952, 736, 320), (576, 5120, 672, 384), (832, 9984, 864, 0), (768, 11648, 800, 0), (768, 7040, 800, 384), (736, 8576, 768, 384), (576, 6464, 736, 384), (672, 10368, 736, 384), (640, 6080, 768, 320), (768, 8960, 768, 384)))
def lifecycle(mod):
    e=mod._ASTRA_SM40_ENV
    return dict(owner_matches=mod._SCRATCH is not None and e._sm40_owner is mod._SCRATCH[0],frames=len(e._sm40_frames),sides=len(e._sm40_cache),outer_owners=len(e._sm40_outer),outer_views=len(e._sm40_outer_cache),io_views=len(e._sm40_io_cache),trace_clear=mod._TR_LAST==[None])
def gate(row):
    rpe_gate(row)
    before=row['sm40_lifecycle_before'];after=row['sm40_lifecycle_after'];st=row['stats']['_ASTRA_SM40_STATS'];bound=row['stats']['_ASTRA_SM40_BIND_STATS']
    assert before==dict(owner_matches=True,frames=4,sides=0,outer_owners=3,outer_views=0,io_views=0,trace_clear=True)
    assert bound==dict(bound=True,families=4,error=None)
    assert st['retired'] is True and st['owner_misses']==0 and 0<st['calls']<=14
    assert len(st['shapes'])==st['calls'] and all(tuple(x)in PLAN_KEYS for x in st['shapes'])
    assert st['max_workspace_bytes']>0
    assert after['owner_matches'] and after['frames']==4 and after['outer_owners']==3 and after['trace_clear']
    assert after['sides']>0 and after['outer_views']>0 and after['io_views']>0
