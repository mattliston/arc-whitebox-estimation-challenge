"""Fold the frozen width8 fit into the exact msg8 teacher literals, without new inference ops."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[k]='1'
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

TEACHER='777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510'
RELEASE='e41095e464be4f5973cae13252a716bc1d85a56bd3ba9d0837e28509c3d08f05'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')

def main():
    ap=argparse.ArgumentParser()
    for k in ['fit','science','release','out']:ap.add_argument('--'+k,type=Path,required=True)
    a=ap.parse_args();f=json.loads(a.fit.read_text())
    assert f['teacher_source_sha256']==TEACHER and sha(a.science)==TEACHER and sha(a.release)==RELEASE
    assert f['fit_source_sha256']=='ea5c8f131bfbe2c6cdd1bef4b7036fb0d1d3dd686ade8fa16c59b12a0e4d470d'
    assert f['width']==8 and not f['uses_network_truth'] and not f['network_score_gain_claimed']
    assert f['recipe']['train_ranks']==[35,150,200,250] and f['recipe']['validation_ranks']==[259,277,287]
    mp=Path(f['captures'][0]['path']).parent/'METADATA_v1.json'
    assert sha(mp)==f['captures'][0]['metadata_sha256']
    meta=json.loads(mp.read_text());edge,context=f['edge_literal'],f['message_edge_literal']
    Q,c,o=(np.asarray(f[k],dtype='f4') for k in ['Q','offset','output'])
    assert Q.shape==(14,8) and c.shape==o.shape==(8,)
    for key,value in edge.items():
        old=np.asarray(meta['edge_literal'][key],dtype='f4')
        if key in ('left','right','edge'):expected=old@Q
        elif key=='bias':expected=old@Q+c
        elif key=='output':expected=o
        else:expected=old
        np.testing.assert_array_equal(np.asarray(value,dtype='f4'),expected)
    for key,value in context.items():
        assert np.asarray(value).shape==(24,8)
        np.testing.assert_array_equal(np.asarray(value,dtype='f4'),np.asarray(meta['message_edge_literal'][key],dtype='f4')@Q)
    assert np.asarray(edge['left']).shape==np.asarray(edge['right']).shape==(14,8)
    assert np.asarray(edge['edge']).shape==(3,8)
    tail='\n# Frozen train4 hidden mixing, folded into existing projections.\n'
    tail+='assert _EDGE_PARAMETERS is None and _MESSAGE_EDGE_PARAMETERS is None\n'
    tail+='assert not _E2_EDGE_STATE\nassert _gn_message_edge_source_base is _g5_message_edge_base\n'
    tail+='assert _VMFR_CM == 8 and _O20_EDGE_HIDDEN == 14\n'
    tail+='assert _O20_FLATEXTRA_ON is False\n'
    tail+='_EDGE_LITERAL = '+repr(edge)+'\n_MESSAGE_EDGE_LITERAL = '+repr(context)+'\n'
    tail+='_O20_EDGE_HIDDEN = 8\n_FAST_RESCUE_FRAC = {}\n'
    a.out.mkdir(parents=True,exist_ok=False)
    tp=a.out/'tail_msg8_edge_hidden8_nonlinear_v1.py'
    with tp.open('x') as out:out.write(tail)
    outputs={}
    for role,parent in [('science',a.science),('release',a.release)]:
        target=a.out/('candidate_msg8_edge_hidden8_nonlinear_'+role+'_v1.py')
        source=parent.read_text()+tail;compile(source,str(target),'exec')
        assert ast.unparse(ast.parse(source).body[-1])=='_FAST_RESCUE_FRAC = {}'
        with target.open('x') as out:out.write(source)
        outputs[role]=dict(path=str(target),sha256=sha(target),parent_sha256=sha(parent),parent_path=str(parent))
    write(a.out/'BUILD_v1.json',dict(utc=utc(),fit_path=str(a.fit),fit_sha256=sha(a.fit),outputs=outputs,
        tail_sha256=sha(tp),message_width=8,edge_hidden_width=8,new_deployed_ops=0,
        folded_parameters_exact=True,training_ranks=[35,150,200,250],validation_ranks=[259,277,287],
        truth_used=False,source_sha256=sha(__file__),native_score_unknown=True))
    print(json.dumps(outputs,indent=2))

if __name__=='__main__':main()
