"""One fixed TRAIN16 observer-only oracle; no truths, model calls or search."""
from pathlib import Path
import argparse,hashlib,json,math
import numpy as np

def basis(x,y):
    a=np.column_stack((x,y));q,r=np.linalg.qr(a,mode='reduced')
    eps=32*np.finfo(np.float64).eps*max(len(x),2)
    if abs(r[0,0]) and abs(r[1,1])>eps*(abs(r[0,1])+abs(r[1,1])):return q
    out=[]
    for z in (x,y):
        z=z.copy();original=np.linalg.norm(z)
        for _ in range(2):
            for b in out:z-=b*np.dot(b,z)
        norm=np.linalg.norm(z)
        if norm>np.finfo(float).eps*max(len(z),2)*original:out.append(z/norm)
    return np.column_stack(out)if out else np.zeros((len(x),0))

def project(Q,Y):return Y-Q@(Q.T@Y)
def zero_diag(E,rows,cols):
    _,ri,ci=np.intersect1d(rows,cols,return_indices=True)
    E[ri,ci]=0

def energies(E,d,g,rows,cols,n):
    W=.5*(d[:,None]*E)*g[None,:]
    C=np.zeros((n,n));C[np.ix_(rows,cols)]=W;C=C+C.T
    return dict(young_error=float(np.sum(E*E)),weighted_directed=float(np.sum(W*W)),leading_symmetric_covariance=float(np.sum(C*C)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--capture-dir',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    directory=Path(a.capture_dir);out=Path(a.output);assert not out.exists()
    files=sorted(directory.glob('L*.npz'));assert len(files)>=1
    result=[];evidence={};pooled={}
    for path in files:
        evidence[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path,allow_pickle=False)as z:
            A=z['left_young'].astype(float);B=z['right_young'].astype(float)*z['lambda_young'].astype(float)
            rows=z['rows'].astype(int);cols=z['cols'].astype(int)
            mu=z['mu'].astype(float);v=z['vc'].astype(float);gate=z['gate'].astype(float)
            Q=z['Q'].astype(float);c15=int(z['c15']);c2=int(z['c2'])
        m,k=A.shape;assert B.shape==(c2,k)and Q.shape[0]==m and c15<c2
        Y=A@B.T;del A,B
        alpha=mu[rows]/np.sqrt(v[rows]);d=np.exp(-.5*alpha*alpha)/(np.sqrt(2*np.pi)*np.sqrt(v[rows]));g=gate[cols]
        parent=project(Q,Y);parent[:,:c15]=0;zero_diag(parent,rows,cols)
        core=c15-96;stop=min(core+160,c2);assert core>0 and stop-core>=32
        Q0=basis(v[rows],mu[rows]);U=Y[:,:core]@(Y[:,:core].T@Q0)
        Qc=basis(U[:,0],U[:,1]if U.shape[1]==2 else np.zeros(m))
        if Qc.shape[1]<Q0.shape[1]:Qc=Q0
        residual=project(Qc,Y);zero_diag(residual,rows,cols)
        sample=np.floor((np.arange(8)+.5)*m/8).astype(int);assert len(np.unique(sample))==8
        band=residual[:,core:stop]
        exact_energy=np.sum((d[:,None]*band)**2,axis=0)*g[core:stop]**2
        sampled_energy=np.sum((d[sample,None]*band[sample])**2,axis=0)*g[core:stop]**2
        sets=dict(gate_budget=np.arange(core,core+32),full_row_oracle=core+np.argsort(-exact_energy,kind='stable')[:32],sample8=core+np.argsort(-sampled_energy,kind='stable')[:32])
        cases={'incumbent_QR':energies(parent,d,g,rows,cols,len(mu))}
        for label,chosen in sets.items():
            E=residual.copy();E[:,:core]=0;E[:,chosen]=0
            cases[label]=energies(E,d,g,rows,cols,len(mu))
        for label,vals in cases.items():
            for key,value in vals.items():pooled.setdefault(label,{}).setdefault(key,0.);pooled[label][key]+=value
        ratios={label:{key:val/max(cases['incumbent_QR'][key],1e-300)for key,val in vals.items()}for label,vals in cases.items()if label!='incumbent_QR'}
        result.append(dict(layer=int(path.stem[1:]),m=m,k=k,c15=c15,c2=c2,core=core,band=stop-core,exact_count=core+32,cases=cases,ratio_to_incumbent=ratios,sampled_oracle_overlap=int(len(np.intersect1d(sets['sample8'],sets['full_row_oracle'])))))
        print(path.stem,json.dumps(ratios),flush=True)
    ratios={label:{key:val/max(pooled['incumbent_QR'][key],1e-300)for key,val in vals.items()}for label,vals in pooled.items()if label!='incumbent_QR'}
    doc=dict(status='FIXED_OFFLINE_COLUMN_BUDGET_ORACLE',formula='exactcore=c15-96; newQ from one signed young-core power step; pick32 among next min160 columns; fixed8 midpoint response rows; row pdf/sigma and column gate sensitivity; exact overwritten global diagonal excluded',per_layer=result,pooled=pooled,pooled_ratio_to_incumbent=ratios,evidence_sha256=evidence,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),limits=['TRAIN16 only. No truth or terminal MSE, no network call, no parameter sweep.','Full-row oracle is a ceiling; sample8 is the one deployable fixed rule. Gate-budget report isolates selection at the same newcore Q.','QR refined basis reconstructed in float64 real arithmetic from exact young output; production float32 reductions may differ.','Sensitivity is leading covariance K21 response, not the complete edge/head/birth or future-network sensitivity.','No candidate or numerical followup authorized by this result.'])
    out.write_text(json.dumps(doc,indent=2)+'\n');print(json.dumps(ratios));print(hashlib.sha256(out.read_bytes()).hexdigest())
if __name__=='__main__':main()
