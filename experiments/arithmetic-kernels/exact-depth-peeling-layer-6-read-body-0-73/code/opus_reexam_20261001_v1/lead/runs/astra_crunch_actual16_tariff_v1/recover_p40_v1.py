from pathlib import Path
import json,itertools,numpy as np
D=Path(__file__).resolve().parent;R=D.parent
P=R/'astra_crunch_smallsm_residual_v1';T={tuple(x['shape']):x for x in json.loads((P/'TARIFF_v1.json').read_text())['rows']}
actual=[]
for row in map(json.loads,(R/'astra_crunch_runtime_v1/s2tr_peel_v1/boundary_v12/rows.jsonl').read_text().splitlines()):
 roster=[dict(shape=x[2:5],count=x[5],record=x[0],family=x[1])for x in row['smallsm_shapes']]
 gain=sum(T[tuple(x['shape'])]['gain']*x['count']for x in roster);req=sum(T[tuple(x['shape'])]['request_delta']*x['count']for x in roster)
 actual.append(dict(rank=row['rank'],roster=roster,exact_warm_gain_flops=gain,exact_warm_request_delta=req,bill_ref=row['flops'],gain_pct=100*gain/(row['flops']+gain),note='Denominator adds back fixed12 credit to final v12 bill; other final components unchanged. Cold35 denominator only approximate warm.'))
ranks=[35,150,16,200,250,259,277,287];rows=[]
for rank in ranks:
 saved=json.loads((R/f'astra_crunch_s2tr_fullstack_v1/res_{rank}_v1.json').read_text());rd={x['layer']:x for x in saved['rtt_reads']}
 b=P/('boundary_35_150_v1'if rank in (35,150)else f'boundary_{rank}_fixed8_v1');row=next(x for x in map(json.loads,(b/'rows.jsonl').read_text().splitlines())if x['rank']==rank)
 # Layers7..9 have one direct old inner at oldcolumns6144; later UVP decomposes the signed pair.
 known=[(rd[l]['m'],6144,384)for l in (7,8,9)if (rd[l]['m'],6144,384)in T]
 width10=2*sum(rd[l]['m']for l in (3,4,5));width13=2*sum(rd[l]['m']for l in (6,7,8))
 options=[]
 for l,w in ((7,6144),(10,width10),(13,width13)):
  pair=[((384,m,w),(m,w,384))for m in (864,928)if (384,m,w)in T and (m,w,384)in T]
  assert len(pair)<=1
  if pair:options.append((l,pair[0]))
 missing=row['sm12_delta']-len(known);assert missing>=0 and missing%2==0
 possibilities=[]
 for pick in itertools.combinations(options,missing//2):
  sh=known+[s for _,p in pick for s in p]
  possibilities.append(dict(joins=[l for l,p in pick],shapes=[list(x)for x in sh],gain_flops=sum(T[x]['gain']for x in sh),request_delta=sum(T[x]['request_delta']for x in sh)))
 assert possibilities
 lo=min(x['gain_flops']for x in possibilities);hi=max(x['gain_flops']for x in possibilities)
 rows.append(dict(rank=rank,observed_fixed12_count=row['sm12_delta'],saved_read_known_shapes=known,cohort_widths=dict(join7=6144,join10=width10,join13=width13),possible_rosters=possibilities,gain_flops_bounds=[lo,hi],gain_pct_bounds=[100*lo/saved['flops'],100*hi/saved['flops']],exact_given_saved_geometry=lo==hi))
out=dict(actual_v12=actual,fixed8_geometry_inference=rows,mean_gain_pct_bounds=[float(np.mean([x['gain_pct_bounds'][k]for x in rows]))for k in (0,1)],mean_gain_flops_bounds=[float(np.mean([x['gain_flops_bounds'][k]for x in rows]))for k in (0,1)],provenance='Actual v12 accepted roster exact on35/100. Fixed8 uses observed route counts plus saved48c RTT m and cohort widths; no model rerun.',limitations=['Inferred fixed8 bounds are conditional on unchanged saved48c read geometry, which was not logged in P40 boundary files. Do not relabel exact P40 telemetry.','Join Y/Fj give paired complementary calls; independent first-old read at layers7..9 gives one call. Other old-bank calls are either rank384 inner/reconstruction or UVP split widths and cannot match fixed12.','Full actual16 old-inner rosters were not logged; seven later boundary files have no RTT read geometry. No exact16 P40 population bill from current files.'],models=0)
(D/'P40_RECOVERY_v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
