from pathlib import Path
import hashlib,json
D=Path(__file__).resolve().parent;O=D.parents[2]
P=O/'lead/runs/astra_yr7_setup_headroom_v1/candidate_yr7_HELD_v1.py'
P_SHA='d401532639bdc3d1b294518828f4af03a873727e510e0d4ec91399b1fc169d5a'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def function(s,name):
 ls=s.splitlines(True);a=[i for i,l in enumerate(ls)if l.startswith('def '+name+'(')][-1];b=a+1
 while b<len(ls)and(not ls[b].strip()or ls[b][0].isspace()):b+=1
 return ''.join(ls[a:b]).rstrip()+'\n'
def fragment(s):
 head='''# Fixed recent-tier difference prefix320. Approximation, not an identity claim.
_RV320_PARENT_STATE = _uvp_state
_RV320_PARENT_RECON = _uvp_recon
_RV320_PARENT_BCC = _astra_bcc_uvrecon
_RV320_PARENT_INNER = _uvp_inner
_RV320_PARENT_PREDICT = Estimator.predict
_RV320_RECENT = [None]
_RV320_STATS = dict(reconstruction=[], read=[], frames=0, retired=True)

def _uvp_state(state):
    uv = _RV320_PARENT_STATE(state)
    group = state.get('lazy')
    if (_UVP_CUR[0] in (13,14,15) and uv[0]=='l' and group is not None
            and group['layer']==13 and group['base_rank']==320 and int(state['rank'])==384):
        recent=uv[2]
        if (recent[3]==256 and int(recent[1].shape[0])==384
                and 0<int(recent[1].shape[1])<=3072 and int(recent[1].shape[1])%32==0):
            if _RV320_RECENT[0] is not recent:_RV320_STATS['frames']+=1
            _RV320_RECENT[0]=recent
        else:_RV320_RECENT[0]=None
    else:_RV320_RECENT[0]=None
    return uv

def _rv320_active(uv):
    return uv is _RV320_RECENT[0] and _UVP_CUR[0] in (13,14,15)

'''
 bcc=function(s,'_astra_bcc_uvrecon').replace('    Fu, Fv2, blocks, ru = uv','    if not _rv320_active(uv):return _RV320_PARENT_BCC(U,uv,out)\n    Fu, Fv2, blocks, ru = uv',1)
 bcc=bcc.replace("Pv = _ubc_product(U, Fv2, uv, 'v')","Pv = _ubc_product(U[:, :320], Fv2[:320], uv, 'v')",1)
 bcc=bcc.replace('    return out',"    _RV320_STATS['reconstruction'].append([int(_UVP_CUR[0]),rows,int(Fv2.shape[1])])\n    return out",1)
 recon='''
def _uvp_recon(U,uv):
    if not _rv320_active(uv):return _RV320_PARENT_RECON(U,uv)
    out=fnp.empty((int(U.shape[0]),2*int(uv[0].shape[1])),dtype=fnp.float32)
    return _astra_bcc_uvrecon(U,uv,out)

'''
 inner=function(s,'_uvp_inner').replace('    Fu, Fv2, blocks, ru = uv','    if not _rv320_active(uv):return _RV320_PARENT_INNER(left_sq,lam_cols,uv)\n    Fu, Fv2, blocks, ru = uv',1)
 inner=inner.replace("    inner = _ubx_product(Dv, Fv2.T, uv, 'vt')","    reduced = _ubx_product(Dv, Fv2[:320].T, uv, 'vt')\n    inner = fnp.zeros((rows,384),dtype=fnp.float32)\n    _BH_HOLD.append(fnp.copyto(inner[:,:320],reduced))\n    del reduced",1)
 inner=inner.replace('    return inner',"    _RV320_STATS['read'].append([int(_UVP_CUR[0]),rows,P])\n    return inner",1)
 end='''
def _rv320_predict(self,mlp,flop_budget):
    if _RV320_RECENT[0] is not None:raise RuntimeError('RV320 stale frame')
    _RV320_STATS.update(reconstruction=[],read=[],frames=0,retired=False)
    try:return _RV320_PARENT_PREDICT(self,mlp,flop_budget)
    finally:
        _RV320_RECENT[0]=None
        _RV320_STATS['retired']=True
Estimator.predict=_rv320_predict
'''
 return head+bcc+recon+inner+end

def transform(raw):
 assert hashlib.sha256(raw).hexdigest()==P_SHA
 return raw+b'\n\n'+fragment(raw.decode()).encode()
if __name__=='__main__':
 raw=P.read_bytes();assert sha(P)==P_SHA
 f=D/'fragment_rv320_HELD_v1.py';c=D/'candidate_rv320_HELD_v1.py'
 f.write_text(fragment(raw.decode()));c.write_bytes(transform(raw));compile(c.read_text(),str(c),'exec')
 print(json.dumps(dict(parent_sha256=P_SHA,fragment_sha256=sha(f),candidate_sha256=sha(c))))
