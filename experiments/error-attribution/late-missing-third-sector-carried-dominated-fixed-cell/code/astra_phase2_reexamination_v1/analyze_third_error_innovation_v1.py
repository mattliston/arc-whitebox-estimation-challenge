"""Fixed carried-versus-fresh third-error attribution; no source arm fitting."""
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from threadpoolctl import threadpool_limits
from residual_one_high_oracle_v1 import distinct_plane

HERE=Path(__file__).resolve().parent;OUT=HERE/'third_error_innovation_r35_l12_v1.json';ART=OUT.with_suffix('.npz')
assert not OUT.exists() and not ART.exists()
f=np.load(HERE/'residual_one_high_inputs_r35_l12_v1.npz');U=f['U'];W=f['next_weight'];indices=f['indices']
ip=HERE/'third_innovation_inputs_r35_l12_v1.npz';p=np.load(ip);g=p['g'];ihash=hashlib.sha256(ip.read_bytes()).hexdigest()
old=np.load(HERE/'residual_one_high_oracle_r35_l12_v1.npz')
F={};E={};M={};provenance={};linearity={};start=time.time()
with threadpool_limits(limits=4):
    for name in ('A','B'):
        prepath=HERE/f'third_innovation_pre_r35_l12_{name}_v1.npz';meta=json.loads(prepath.with_suffix('.json').read_text())
        assert meta['frozen_input_sha256']==ihash and meta['output_sha256']==hashlib.sha256(prepath.read_bytes()).hexdigest()
        tp=HERE/f'residual_one_high_truth_r35_l12_{name}_v1.npz';assert meta['previous_stream_sha256']==hashlib.sha256(tp.read_bytes()).hexdigest()
        pre=np.load(prepath);post=np.load(tp)
        total=post['slices']-f['model_slices']
        carried=(pre['pre_slices']-p['model_pre_slices'])*g[None,:,None]*g[None,None,:]
        fresh=total-carried;M[name]=dict(total=total,carried=carried,fresh=fresh)
        F[name]={k:distinct_plane(v,U,W,indices) for k,v in M[name].items()}
        difference=float(np.max(np.abs(F[name]['total']-F[name]['carried']-F[name]['fresh'])))
        parent_difference=float(np.max(np.abs(F[name]['total']-old[f'{name}_distinct'])))
        assert difference<1e-12 and parent_difference<1e-12,(difference,parent_difference)
        E[name]=post['next_K21']-f['model_next_K21'];provenance[name]=meta
        linearity[name]=dict(decomposition_max_error=difference,original_oracle_query_max_error=parent_difference)

    def cross(a,b):return float(.5*np.mean(a['A']*b['B']+a['B']*b['A']))
    def report(container,weight):
        vectors={k:{s:container[s][k]*weight for s in ('A','B')} for k in ('total','carried','fresh')}
        energies={k:cross(v,v) for k,v in vectors.items()}
        correlations={}
        for a,b in (('carried','total'),('fresh','total'),('carried','fresh')):
            z=cross(vectors[a],vectors[b]);den=energies[a]*energies[b]
            correlations[a+'_'+b]=dict(cross_energy=z,cosine=z/np.sqrt(den) if den>0 else None)
        return dict(energies=energies,fresh_fraction_of_total=energies['fresh']/energies['total'],
                    carried_fraction_of_total=energies['carried']/energies['total'],cross=correlations,
                    energy_identity_error=energies['total']-energies['carried']-energies['fresh']-2*correlations['carried_fresh']['cross_energy'])
    geometry=dict(source_slices=report(M,1.),next_plane=report(F,1.),weighted_next_plane=report(F,f['consumer_weight']))
    def correction_metrics(arm):
        w=f['consumer_weight'];EA=E['A']*w;EB=E['B']*w;FA=F['A'][arm]*w;FB=F['B'][arm]*w
        baseline=float(np.mean(EA*EB));energy=float(np.mean(FA*FB));alignment=float(.5*np.mean(EA*FB+EB*FA))
        return dict(signal_error_ratio=(baseline-2*alignment+energy)/baseline,
                    signal_cosine=alignment/np.sqrt(baseline*energy) if energy>0 else None,
                    A_target_B_correction_ratio=float(np.sum((EA-FB)**2)/np.sum(EA*EA)),
                    B_target_A_correction_ratio=float(np.sum((EB-FA)**2)/np.sum(EB*EB)))
    effects={k:correction_metrics(k) for k in ('total','carried','fresh')}
    v=geometry['weighted_next_plane'];co=v['cross']['carried_total']['cosine']
    dominated=v['energies']['fresh']>=0 and v['fresh_fraction_of_total']<=.25 and co is not None and co>=.8
np.savez(ART,**{s+'_'+k:v for s,rows in F.items() for k,v in rows.items()},indices=indices)
record=dict(rank=35,layer=12,geometry=geometry,weighted_natural_effects=effects,
            carried_dominated_by_predeclared_gate=bool(dominated),linearity=linearity,acquisition=provenance,
            wall_s=time.time()-start,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            qualification='Carried G^3 pre-third error versus fresh remainder, same fixed q4/32-row consumer and two independent8M laws. Linear decomposition only; includes parent live mask. Sampling-noise-separated point estimates, not confidence bounds. No failed source gate re-score, fitted arm, deployable correction, raw or adjusted score.')
OUT.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k!='acquisition'},indent=2),flush=True)
