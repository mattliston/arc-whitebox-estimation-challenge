"""Price final all-push-off and three-read fallback: all source geometry, no numerical kernels."""
from pathlib import Path
import hashlib,json,resource,time
D=Path(__file__).resolve().parent;R=D.parents[1];t0=time.monotonic()
source=R/'astra_crunch_runtime_v1/failure336640_v1/candidate_setup_hold_v19e.py'
pre=D.parent/'bini_apa_v1/price_outer_232_v3.py'
code=pre.read_text().split('def make_bini')[0].replace("p.SOURCE=O/'lead/runs/opus_pubmax_v1/cand_pubmax_B_parity.py'",'p.SOURCE=Path('+repr(str(source))+')')
ns={'__file__':str(pre),'__name__':'family_repair_metadata'};exec(code,ns)
M,fn,fs,arena,measure=(ns[k]for k in('M','fn','fs','arena','measure'))
original=(dict(M._PEEL_PUSH_PLANS),dict(M._PEEL_READ_PLANS),dict(M._SMX_FAMILIES))
four={('push',(1024,992,8064)),('push',(1024,992,8128)),('read',(1024,4096,960,0)),('read',(992,6144,928,0))}
extra={('push',key) for key in original[0]}|{('read',(992,6144,896,0))}
dead1={'peel6_p20_5_11_5'}
live={p[1] for k,p in original[1].items() if ('read',k) not in four|extra}|{p[1] for p in M._SM12_PLANS.values()}
dead3=set(original[2])-live
with fs.BudgetContext(flop_budget=2**60,quiet=True):
    M._ASTRA_SM40_ENV._SCRATCH=M._SCRATCH
    M._astra_sm40_bind_universal(M._ASTRA_SM40_ENV,arena)
rows=[]
original_orientations={name:sorted({tuple(M._SMX_RECORDS[p[0]]['shape']) for plans in original[:2] for p in plans.values() if p[1]==name}) for name in original[2]}
for label,excluded,dead in [('four',four,dead1),('all_push_off_three_reads',four|extra,dead3)]:
    M._PEEL_PUSH_PLANS={k:v for k,v in original[0].items()if('push',k)not in excluded}
    M._PEEL_READ_PLANS={k:v for k,v in original[1].items()if('read',k)not in excluded}
    used={p[1]for plans in(M._PEEL_PUSH_PLANS,M._PEEL_READ_PLANS,M._SM12_PLANS)for p in plans.values()}
    assert not(used&dead)
    M._SMX_FAMILIES={k:v for k,v in original[2].items()if k not in dead}
    for name in M._SMX_FAMILIES:
        surviving=sorted({tuple(M._SMX_RECORDS[p[0]]['shape']) for plans in (M._PEEL_PUSH_PLANS,M._PEEL_READ_PLANS) for p in plans.values() if p[1]==name})
        assert surviving==original_orientations[name],(name,surviving,original_orientations[name])
    M._PEEL_OWNER[0]=None
    phases={}
    for name in('_smx_bind','_peel_bind_read_io','_peel_prebind_topology','_sm12_prebind_topology'):
        phases[name]=measure(getattr(M,name))
    rows.append(dict(label=label,excluded=sorted(excluded),dead_families=sorted(dead),phases=phases,
        calls=sum(x['requests']for x in phases.values()),flops=sum(x['flops']for x in phases.values()),
        frames=len(M._SMX_ENV._sm40_frames),side_cache=len(M._SMX_ENV._sm40_cache),io_cache=len(M._SMX_ENV._sm40_io_cache),
        groups=len(M._SMX_ENV._sm40_group_cache),read_io=len(M._PEEL_READ_IO)))
M._PEEL_PUSH_PLANS,M._PEEL_READ_PLANS,M._SMX_FAMILIES=original
out=dict(status='METADATA_ONLY',models=0,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),rows=rows,
    removed_setup_calls=rows[0]['calls']-rows[1]['calls'],removed_setup_flops=rows[0]['flops']-rows[1]['flops'],
    added_read_parent_has_mainread_plan=(992,6144,896,0)in M._ASTRA_SM40_PLANS,
    seconds=time.monotonic()-t0,maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    surviving_family_dimensions_and_orientation_sets_unchanged=True,limitations=['Only fixed SMX/peel/smallSM setup components differ; runtime must measure full startup and hidden free trips.',
                 'The added read parent is generic if no mainread plan; its cold cache demand is not priced here.'])
with(D/'ALL_PUSH_SETUP_PRICE_v1.json').open('x')as f:json.dump(out,f,indent=2)
print(json.dumps(out))
