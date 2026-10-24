"""Freeze exact runtime companions for current337466. No setup/native/fire."""
import argparse,ast,hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
RUNS=HERE.parent.parent
K=RUNS/'fable_oct19_kc_v1'; A=K/'agent_astra_joint_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
PARENTS={
 'science':(K/'candidate_astra_o21_edge8_local90_per_layer960_v1_sci_v1.py','434b3a4730febb062b59a8a29c042d27e1f4bdbf8301cef3a72c90b2e798fcb4'),
 'release':(K/'candidate_rel_astra_o21_edge8_local90_per_layer960_v1.py','660bc2d6daa5950c8614ca505775d305d3e19031a7a57eb0ac815b90ab5b8200')}
TAILS={
 'features':(A/'tail_features_grouped_v1.py','067e1cf1ce32636015071afcbd9135d39405e80607cf50d4387ea69796f83054'),
 'pair':(A/'tail_pair_grouped_current_v2.py','bdd6e100b8f5a762019faf048b6f5fda9b3310ade779f4e4f700b5f4ae384318')}
for p,h in [*PARENTS.values(),*TAILS.values()]:assert sha(p)==h,(p,sha(p))
nogov=A/'tail_disable_fast_rescue_astra_v1.py'
off='''\n# Fixed exact grouped Smirnov input circuit. Chosen at source construction.\n# No runtime trigger, output fallback, coefficient change, or guard change.\nassert _O20_SMDIRECT_ON is True\nassert _O20_SMDIRECT_PARENT.__name__ == '_astra_sm40_linear_grouped'\n_O20_SMDIRECT_ON = False\n'''
static=[]; outputs={}
for key,group in [('smoff',False),('smoff_grouped',True)]:
 outputs[key]={}
 for role,(parent,digest) in PARENTS.items():
  text=parent.read_text(); tree=ast.parse(text)
  assert ast.unparse(tree.body[-1])=='_FAST_RESCUE_FRAC = {}'
  assert not any(isinstance(n,ast.FunctionDef) and n.name=='_vecb_features' for n in tree.body)
  assert any(isinstance(n,ast.Assign) and ast.unparse(n)=='_O20_SMDIRECT_ON = True' for n in tree.body)
  tail=off
  if group:tail+='\n'+TAILS['features'][0].read_text()+'\n'+TAILS['pair'][0].read_text()
  tail+='\n'+nogov.read_text()
  target=HERE/f'candidate_{key}_{role}_v1.py'
  assert not target.exists(),target
  target.write_text(text+'\n'+tail)
  parsed=ast.parse(target.read_text())
  assert ast.dump(ast.Module(body=parsed.body[:len(tree.body)],type_ignores=[]),include_attributes=False)==ast.dump(tree,include_attributes=False)
  assert ast.unparse(parsed.body[-1])=='_FAST_RESCUE_FRAC = {}'
  outputs[key][role]=dict(path=str(target),sha256=sha(target),parent_path=str(parent),parent_sha256=digest)
  static.append(dict(key=key,role=role,parent_prefix_byte_exact=target.read_bytes().startswith(parent.read_bytes()),last_statement_permanent_nogov=True))
r=dict(utc=utc(),status='SOURCE_PREPARED_NOT_NATIVE_QUALIFIED',outputs=outputs,static=static,
 tails={k:dict(path=str(p),sha256=h) for k,(p,h) in TAILS.items()},nogov_sha256=sha(nogov),
 smoff_tail=off,no_governor=True,no_fire=True,
 unmeasured_historical_projection=dict(smoff_flops_added=337715200,smoff_requests=-534,smoff_approx_bill_pct=.143,
 grouped_extra_flops_warm=516096,grouped_extra_flops_cold=516104,grouped_extra_calls_warm=-112,grouped_extra_calls_cold=-111),
 limitations=['Historical pricing is not a current measurement.','Request reduction is not measured runtime reduction.','Never a standalone runtime twin submission.'])
with (HERE/'BUILD_v1.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(dict(utc=r['utc'],outputs=outputs,status=r['status'])))
