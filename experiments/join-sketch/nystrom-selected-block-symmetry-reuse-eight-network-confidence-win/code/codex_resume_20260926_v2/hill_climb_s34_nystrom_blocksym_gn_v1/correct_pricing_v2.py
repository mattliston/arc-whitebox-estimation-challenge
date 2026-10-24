"""Prospective scalar correction only; original source and all first records preserved."""
from pathlib import Path
import copy,hashlib,json
D=Path(__file__).resolve().parent
def pin(p):
    p=Path(p).resolve();b=p.read_bytes()
    return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def once(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
price=json.loads((D/'price.json').read_text())
assert pin(D/'candidate.py')['sha256']=='6dfbcfa4637a6d3c3e4ffb126baaae9b0e76410edf81340d6230e61571fcafbb'
price['original_price_preserved']=pin(D/'price.json')
price['correction_script']=pin(__file__)
for row in price['records']:
    m,k,r=row['shape'];area=sum(b*n for b,_,n in row['strips'])
    diagonal=sum(b*b for b,_,_ in row['strips']);off=(r*r-diagonal)//2
    original=row['extra_gather_copy_allowance']
    corrected=4*((m-r)*k+r*k+k*sum(r-s for s in (0,128,256)))+5*area+13*off+4*m*r+10*m
    row['original_extra_gather_copy_allowance']=original
    row['extra_gather_copy_allowance']=corrected
    row['new_model_flops']+=corrected-original
    row['saved_flops']=row['old_model_flops']-row['new_model_flops']
price['forecast_saved_flops']=sum(row['saved_flops'] for row in price['records'])
price['forecast_pct']=100*price['forecast_saved_flops']/price['reference_s34_rank35_flops']
price['corrected_allowance']='4*((m-r)*k+r*k+k*sum(r-start)) + 5*computed_upper_area + 13*mirrored_off_area + 4*m*r + 10*m. Charge4 for gathers/slices and conservatively also transposes; upper store4+copy1, mirrored source4+transpose4+destination4+copy1. Original selected gather/weighted RHS are common to both arms and cancel.'
price['local_billing_evidence']=pin(D.parent/'hill_climb_s34_pair_blocksym_v1/price.json')
price['standalone_nystrom_above_point1']=False
price['combined_pilot_basis']='Root selected bundle before execution; GN and scan each passed isolated all16 byte parity. Their observed gains are separate evidence, not an isolated Nyström or measured bundle claim.'
assert 0.0802<price['forecast_pct']<0.0803
once(D/'price_v2.json',price)
p=json.loads((D/'protocol.json').read_text());p['price']=pin(D/'price_v2.json')
p['original_protocol_preserved']=pin(D/'protocol.json')
p['pricing_correction']='Fourfold getitem allowance replaces undercount before candidate execution. Source and all score/resource gates unchanged.'
once(D/'protocol_v2.json',p)
b=json.loads((D/'build.json').read_text());b['price']=pin(D/'price_v2.json');b['protocol']=pin(D/'protocol_v2.json')
b['original_build_preserved']=pin(D/'build.json');b['pricing_correction']=pin(__file__)
once(D/'build_v2.json',b)
c=json.loads((D/'native_screen.json').read_text());original=copy.deepcopy(c)
replacements={str((D/name).resolve()):pin(D/new) for name,new in
    (('price.json','price_v2.json'),('protocol.json','protocol_v2.json'),('build.json','build_v2.json'))}
c['provenance']=[replacements.get(x['path'],x) for x in c['provenance']]
c['provenance'] += [pin(__file__),pin(D/'native_screen.json'),pin(D/'price.json'),pin(D/'protocol.json'),pin(D/'build.json')]
c['pricing_revision']=dict(build=pin(D/'build_v2.json'),protocol=pin(D/'protocol_v2.json'),price=pin(D/'price_v2.json'),original_config_preserved=pin(D/'native_screen.json'),source_unchanged=True)
c['interpretation']+=' Corrected conservative Nyström-alone forecast is .080245%; observed isolated GN/scan gains remain separately reported. No score/resource gate changes.'
assert all(c[k]==original[k] for k in ('engine','arms','stages','parity_layers','future_only','raw_error_veto','prospective_scope','boundary_before8'))
once(D/'native_screen_v2.json',c)
print(json.dumps(dict(candidate=pin(D/'candidate.py'),config=pin(D/'native_screen_v2.json'),build=pin(D/'build_v2.json'),price=pin(D/'price_v2.json'),forecast_pct=price['forecast_pct'])))
