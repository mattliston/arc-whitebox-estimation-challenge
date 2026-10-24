from pathlib import Path
import ast,hashlib,json
D=Path(__file__).resolve().parent
P=D.parent/'astra_yr7_setup_headroom_v1/candidate_yr7_HELD_v1.py'
SHA='d401532639bdc3d1b294518828f4af03a873727e510e0d4ec91399b1fc169d5a'
def transform(data):
 assert hashlib.sha256(data).hexdigest()==SHA
 source=data.decode();tree=ast.parse(source)
 inner=[x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='_uvp_inner'][-1]
 ks=[x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='_ubx_ks_product'and 15000<x.lineno<16000][0]
 a=ast.get_source_segment(source,inner).replace('def _uvp_inner(', 'def _prd224_inner(')
 old="inner = _ubx_product(Dv, Fv2.T, uv, 'vt')";assert a.count(old)==1
 a=a.replace(old,'inner = _prd_difference(Dv, Fv2, uv, lam_cols)')
 b=ast.get_source_segment(source,ks).replace('def _ubx_ks_product(', 'def _prd224_ks_product(')
 old='if ctx is not None and not hzc:';assert b.count(old)==1
 b=b.replace(old,"if ctx is not None and (not hzc or (n == 224 and ctx['tag'][0] == 'prd224')):")
 fragment=(D/'fragment_prd224_HELD_v1.py').read_bytes()
 return data+b'\n\n'+a.encode()+b'\n\n'+b.encode()+b'\n\n'+fragment
if __name__=='__main__':
 data=transform(P.read_bytes());compile(data,'candidate_prd224_HELD_v1.py','exec')
 out=D/'candidate_prd224_HELD_v1.py'
 with out.open('xb')as f:f.write(data)
 print(hashlib.sha256(data).hexdigest())
