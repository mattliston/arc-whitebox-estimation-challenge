"""Inert target-blind quadratic RQMC information acquisition; never a score."""
import argparse
import ast
from datetime import datetime, timezone
from decimal import Decimal, localcontext
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import sys
import time
import traceback
import types

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
AP = '/Users/matt/anaconda3/bin/python'
SCHEMA = 'current_first_layer_quadratic_rqmc_information_v1'
WATCH_SCHEMA = 'current_first_layer_quadratic_rqmc_watch_v1'
PROTOCOL_SHA = '93dfd3461af6c9c9c5e4c852800560d9d91909f00f6447a09b943f1770920afa'
SAMPLER = P2/'rqmc_fullbudget_rank1024_cv/estimator.py'
SAMPLER_SHA = '6801eb3edecfd12b1e7f74e51b5987111594a6857885492f2934f68f40f37e92'
PROVIDER = NEW/'positive_gmm4_split_merge_prototype_v1.py'
PROVIDER_SHA = '16cc38b25bcb6ff0796b5630eb538bb60ccaba166c97be944994ce905d317a58'
WEIGHT_CONTRACT = P2/'learned_residual_observer_design/v4_feature_store/contract.py'
WEIGHT_CONTRACT_SHA = '9744e49d3e17d92a57debe939f67a3f677f03d93250eca8308eb9a219303083c'
R2 = NEW/'current_p1_native_objective_p2_monitor_v2'
P1 = NEW/'current_d50_native_objective_p1_v1'
MONITOR_SHA = 'd4b3af53b716593670d8e0c7f0fc6f2133b57f1088d9966c06635d3a23f39873'
CLASSIFIER_SHA = '4e8177d6ff943cdee37e48ccca750bf9d552b32cec344aa2b37e6304cbd55fb3'
MOCK_SHA = 'c5a3f5b8a3dc99ca7e3550fb9ccb474b76968a49b96e00ed7e5606597476bf0e'
MOCK_END_SHA = 'bfb69013c603f0578a56c5e268e3654221b1ab50e00a8bd10159ef074b9e846f'
RANKS = (35,36,38,100)
ROSTER = [dict(rule_index=16*i+j,rank=r,replicate=j,seed=9309100000+100*r+j,
    tail_seed=(9309100000+100*r+j)^0xA17C9E52D643B80F) for i,r in enumerate(RANKS) for j in range(16)]
WIDTH, DIRECTIONS, PATHS = 1024,4096,8192
HOST, DRIVER, LIMIT, PROGRESS, SWAP = 8_000_000_000,8_000_000_000,600.,120.,0
PLANNED_F = 428_726_925_824
BLAS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
VERSIONS = dict(numpy='1.26.4',scipy='1.11.1',torch='2.7.0')


def require(value,message):
    if not value: raise RuntimeError(message)


def sha(path,tick=None):
    path = Path(path).resolve()
    require(path.is_relative_to(P2) and not (path.suffix in ('.npy','.npz') and 'truth' in str(path).lower()),
        'Only fixed Phase2 sources/weights/own outputs, never target arrays')
    before = path.stat(); h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''):
            h.update(block)
            if tick: tick()
    after = path.stat()
    require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Immutable file while hashing')
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def weight_payload_sha(array):
    """Exact STORE array_payload_sha256: dtype/shape envelope, NOT raw SHA."""
    value = np.ascontiguousarray(array); digest = hashlib.sha256()
    digest.update(value.dtype.str.encode('ascii'))
    digest.update(json.dumps(list(value.shape),separators=(',',':')).encode('ascii'))
    digest.update(memoryview(value).cast('B'))
    return digest.hexdigest()


def write(path,value):
    path = Path(path); temporary = path.with_name(path.name+'.publishing')
    with temporary.open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False); stream.write('\n')
    os.link(temporary,path); temporary.unlink()


def extract(path,names):
    source = Path(path).read_text(); lines = source.splitlines(keepends=True)
    nodes = [n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in names]
    require(len(nodes)==len(names) and {n.name for n in nodes}==set(names),'Exact unique function roster')
    pieces,changed,edits = [],source,[]
    for n in nodes:
        piece = ''.join(lines[n.lineno-1:n.end_lineno]); token = '# extracted '+n.name+'\n'
        require(changed.count(piece)==1,'Unique entire function source')
        changed = changed.replace(piece,token); pieces.append(piece); edits.append((piece,token))
    for piece,token in reversed(edits): changed = changed.replace(token,piece)
    require(changed==source,'Full source-owner inverse')
    return '\n'.join(pieces),dict(parent_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        functions=list(names),inverse_source_byte_exact=True,arithmetic_substitutions=0)


def environment():
    expected = dict.fromkeys(BLAS,'1') | dict(PYTHONDONTWRITEBYTECODE='1',
        PYTORCH_ENABLE_MPS_FALLBACK='0',PYTORCH_MPS_LOW_WATERMARK_RATIO='0.1')
    require(sys.executable==AP and sys.dont_write_bytecode and sys.flags.optimize==0
        and os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)==v for k,v in expected.items()),
        'Reviewed AP/nice10/BLAS1/bytecode/MPS environment')
    return dict(executable=sys.executable,nice=os.getpriority(os.PRIO_PROCESS,0),launch=expected)


def decimal_radius():
    """Fixed exact formula, NOT the earlier approximate radius literal."""
    records,roundings = [],[]
    for precision in (80,120):
        with localcontext() as context:
            context.prec = precision+30
            def arctan_inverse(integer):
                x = Decimal(1)/integer; power = x; total = Decimal(0); k = 0
                while True:
                    term = power/(2*k+1); total += term if k%2==0 else -term
                    power *= x*x; k += 1
                    bound = power/(2*k+1)
                    if bound < Decimal(10)**(-precision-20): return total,bound,k
            a,ba,ka = arctan_inverse(5); b,bb,kb = arctan_inverse(239)
            pi = 16*a-4*b
            factorial = (2*pi).sqrt()*Decimal(math.factorial(1024))/(
                Decimal(2)**1024*Decimal(math.factorial(512))*Decimal(math.factorial(511)))
            recurrence = (pi/2).sqrt()
            for k in range(1,512): recurrence *= Decimal(2*k+1)/Decimal(2*k)
            envelope = Decimal(10)**(-precision+5)
            require(abs(factorial-recurrence)<envelope,'Independent chi-radius identities')
            # Generous envelope above truncation and512 operations at p+30 digits.
            values = [float(v+sign*envelope) for v in (factorial,recurrence) for sign in (-1,0,1)]
            require(len(set(values))==1,'Unique FP64 rounding over fixed high-precision envelope')
            roundings.extend(values)
            records.append(dict(precision=precision,pi=str(pi),factorial=str(factorial),recurrence=str(recurrence),
                pi_alternating_truncation_bound=str(16*ba+4*bb),decimal_roundoff_envelope=str(envelope),
                arctan_terms=[ka,kb],rounded_float64=values[0]))
    require(len(set(roundings))==1,'80/120-digit same fixed radius')
    return roundings[0],records


class Stage:
    def __init__(self,args):
        self.start,self.torch,self.args = time.monotonic(),None,args
        self.folder = HERE/'information_v1'; self.folder.mkdir(exist_ok=False)
        self.record = dict(schema=SCHEMA,stage='information',passed=False,completed=False,
            hashes_unchanged=False,sources_unchanged=False,source_bindings={},input_bindings={},weight_bindings={},
            artifacts={},rule_records=[],rank_summaries={},truth_accessed=False,targets_accessed=False,
            training_targets_accessed=False,validation_targets_accessed=False,parameter_updates=0,
            gradient_calls=0,model_fitting=False,native_score_measured=False,native_bill_measured=False,
            scientific_advancement=False,one_rule_paths=PATHS,rules_per_rank=16,ranks=list(RANKS),
            planning_cost=dict(leading_flops=403_726_925_824,auxiliary_allowance=25_000_000_000,
                sampler_allowance_within_auxiliary=2_000_000_000,total=PLANNED_F,multiplier=PLANNED_F/2**41,
                metered=False,verified_whole_source_upper_bound=False,coefficient_setup_paid_per_production_rule=True),
            maximum_sampled_mps_driver_bytes=0,peak_rss_bytes=0,
            biases_not_certified=['midpoint-grid law','PPF and radius rounding','FP64 analytic-center evaluation',
                'FP32 paths/first-layer/control products','nonnegative clamp and FP32 return'],
            started_utc=datetime.now(timezone.utc).isoformat())
        self.weights = {}

    def guard(self):
        elapsed = time.monotonic()-self.start
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        driver = int(self.torch.mps.driver_allocated_memory()) if self.torch is not None else 0
        self.record['peak_rss_bytes'] = max(self.record['peak_rss_bytes'],rss)
        self.record['maximum_sampled_mps_driver_bytes'] = max(self.record['maximum_sampled_mps_driver_bytes'],driver)
        require(elapsed<LIMIT and rss<HOST and driver<DRIVER,'Cooperative600s/8GB host/8GB driver')
        return dict(elapsed_s=elapsed,peak_rss_bytes=rss,mps_driver_bytes=driver)

    def pin(self,path,expected=None,group='input_bindings'):
        path = Path(path).resolve(); digest = sha(path,self.guard)
        require(expected is None or digest==expected,'Exact source/input binding: '+str(path))
        current = self.record[group]
        require(str(path) not in current or current[str(path)]==digest,'Consistent immutable binding')
        current[str(path)] = digest; return digest

    def json(self,name,value):
        path = self.folder/name; write(path,value)
        self.record['artifacts'][str(path)] = sha(path,self.guard)
        return dict(path=str(path),sha256=self.record['artifacts'][str(path)])

    def bundle(self,name,values):
        arrays = {k:np.ascontiguousarray(v) for k,v in values.items()}
        metadata = {k:dict(dtype=v.dtype.str,shape=list(v.shape),raw_sha256=hashlib.sha256(v.tobytes()).hexdigest())
            for k,v in arrays.items()}
        require(all(np.isfinite(v).all() for v in arrays.values()),'Finite saved numerical evidence')
        path = self.folder/name
        with path.open('xb') as stream: np.savez(stream,**arrays)
        digest = sha(path,self.guard); self.record['artifacts'][str(path)] = digest
        return dict(path=str(path),sha256=digest,arrays=metadata)

    def admission(self):
        args = self.args
        require(all(re.fullmatch('[0-9a-f]{64}',getattr(args,k+'_sha256') or '')
            for k in ('source','watcher','protocol')),'Three explicit reviewed source hashes')
        require(args.protocol_sha256==PROTOCOL_SHA,'Fixed prospective protocol')
        for path,digest in ((Path(__file__),args.source_sha256),(HERE/'watch.py',args.watcher_sha256),
            (HERE/'PROTOCOL.md',PROTOCOL_SHA),(SAMPLER,SAMPLER_SHA),(PROVIDER,PROVIDER_SHA),
            (WEIGHT_CONTRACT,WEIGHT_CONTRACT_SHA),
            (R2/'monitor.py',MONITOR_SHA),(P1/'watch.py',CLASSIFIER_SHA)):
            self.pin(path,digest,'source_bindings')
        self.record['environment'] = environment()
        require(not {'numpy','torch','scipy','flopscope','threadpoolctl'} & set(sys.modules),
            'All source/weight admission before numerical imports')
        manifest = STORE/'manifest.json'
        self.pin(manifest,'9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50')
        entries = read(manifest)['roles']['observer_fit']['row_manifests']
        for rank in RANKS:
            matches = [v for v in entries if v['role_rank']==rank]
            require(len(matches)==1,'Unique canonical TRAIN weight row')
            path = (STORE/matches[0]['path']).resolve(); require(path.is_relative_to(STORE),'Canonical row scope')
            self.pin(path,matches[0]['sha256'],'weight_bindings'); row = read(path)
            require(row['role']=='observer_fit' and row['role_rank']==rank,'Canonical role/rank')
            info = row['files']['weights.npy']; wp = path.parent/'weights.npy'
            require(info['dtype']=='<f4' and info['shape']==[16,1024,1024],'Stored weight frame')
            self.pin(wp,info['sha256'],'weight_bindings')
            self.weights[rank] = dict(path=str(wp),**info,mlp_seed=row['mlp_seed'])
        self.record['rule_roster'] = self.json('rule_roster.json',dict(schema=SCHEMA,entries=ROSTER))
        self.json('admission.json',dict(source_bindings=self.record['source_bindings'],
            input_bindings=self.record['input_bindings'],weight_bindings=self.record['weight_bindings'],
            weights=self.weights,roster=ROSTER,truth_accessed=False))

    def finish(self):
        for group in ('source_bindings','input_bindings','weight_bindings','artifacts'):
            for path,digest in self.record[group].items(): require(sha(path,self.guard)==digest,'Final unchanged '+group)
        require(len(self.record['rule_records'])==64 and set(self.record['rank_summaries'])==set(map(str,RANKS)),
            'Entire fixed information experiment, never partial selection')
        self.record.update(passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True,
            numerical_integrity_passed=True,all64_rules_sealed=True)

    def close(self):
        self.record.update(elapsed_s=time.monotonic()-self.start,finished_utc=datetime.now(timezone.utc).isoformat())
        write(self.folder/'result.json',self.record)
        write(self.folder/'completion.json',dict(passed=self.record['passed'],completed=self.record['completed'],
            result_sha256=sha(self.folder/'result.json')))


def first_moments(W0):
    w = W0.astype(np.float64); gram = w.T@w; v = np.diag(gram).copy()
    require(np.isfinite(gram).all() and np.all(v>=0),'Exact first-layer Gram domain')
    sigma = np.sqrt(v); active = sigma>0; scale = sigma[:,None]*sigma[None,:]
    rho = np.divide(gram,scale,out=np.zeros_like(gram),where=scale>0)
    tolerance = 64*np.finfo(np.float64).eps
    require(np.max(np.abs(rho))<=1+tolerance,'First-layer correlation beyond fixed roundoff tolerance')
    clipped = np.clip(rho,-1.,1.); defect = np.abs(clipped-rho)
    np.fill_diagonal(clipped,active.astype(np.float64))
    M = scale*(np.sqrt(np.maximum(1-clipped*clipped,0.))+(math.pi-np.arccos(clipped))*clipped)/(2*math.pi)
    np.fill_diagonal(M,v/2)
    m = sigma/math.sqrt(2*math.pi); C = M-m[:,None]*m[None,:]
    require(np.isfinite(C).all() and np.all(np.diag(C)>=0),'First-layer centered covariance domain')
    return m,M,C,dict(correlation_endpoint_lifts=int(np.count_nonzero(defect)),
        maximum_correlation_endpoint_lift=float(np.max(defect)),zero_columns=int(np.count_nonzero(~active)))


def numerical_helpers(stage):
    code,proof = extract(PROVIDER,('marginal','gaussian_map','factor_state'))
    env = dict(np=np,ndtr=ndtr,math=math,ORDER=12,require=require)
    exec(compile(code,str(PROVIDER)+':unchanged_helpers','exec'),env)
    stage.record['provider_source_inverse'] = proof
    names = ('_sobol_direction_numbers','_sobol_direction_bits','_scrambled_sobol_normals')
    code,proof = extract(SAMPLER,names)
    old = ('    unit = fnp.asarray(residues, dtype=fnp.float64) * _UINT32_SCALE\n'
        '    unit = fnp.maximum(fnp.minimum(unit, 1.0 - _PPF_EPS), _PPF_EPS)\n'
        '    return fnp.asarray(flops.stats.norm.ppf(unit), dtype=fnp.float32)\n')
    new = ('    tail = tail_rng.integers(0, 1 << 20, size=residues.shape, dtype=fnp.uint64)\n'
        '    integers = residues.astype(fnp.uint64)*(1 << 20)+tail\n'
        '    unit = (integers.astype(fnp.float64)+0.5)*(2.0**-52)\n'
        '    require(fnp.all(unit > 0) and fnp.all(unit < 1), "Open52-bit midpoint grid")\n'
        '    return ndtri(unit), unit\n')
    require(code.count(old)==1,'Only original sampler final conversion replaced')
    changed = code.replace(old,new); require(changed.replace(new,old)==code,'Sampler-tail full source inverse')
    node = [v for v in ast.parse(SAMPLER.read_text()).body if isinstance(v,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='_SOBOL_DIRECTION_HEX' for t in v.targets)]
    require(len(node)==1,'Unique source-owned Joe--Kuo table')
    table = ast.literal_eval(node[0].value)
    sampling = dict(fnp=np,ndtri=ndtri,require=require,_SOBOL_BITS=32,_SOBOL_DIRECTION_HEX=table)
    exec(compile(changed,str(SAMPLER)+':midpoint_tail_only','exec'),sampling)
    proof.update(arithmetic_substitutions=1,only_change='clipped32-bit PPF tail to independent20-bit open52 midpoint',
        numpy_alias_is_offline_transport=True,table_bytes_sha256=hashlib.sha256(bytes.fromhex(table)).hexdigest())
    stage.record['sampler_source_inverse'] = proof
    return types.SimpleNamespace(**env),sampling


def tiny_checks(stage):
    radius,certificate = decimal_radius()
    integers = np.array([0,1,(1 << 52)-2,(1 << 52)-1],dtype=np.uint64)
    midpoints = (integers.astype(np.float64)+.5)*2.**-52
    normals = ndtri(midpoints)
    require(midpoints[0]==2.**-53 and midpoints[-1]==1.-2.**-53 and np.isfinite(normals).all()
        and np.max(np.abs(ndtr(normals)-midpoints))<=4*np.finfo(np.float64).eps,
        'Exact open52-bit endpoints and finite PPF roundtrip')
    W = np.array([[1.,1.,-1.,0.],[0.,0.,0.,0.]],dtype=np.float32)
    m,M,C,domain = first_moments(W)
    expected = np.array([[.5,.5,0,0],[.5,.5,0,0],[0,0,.5,0],[0,0,0,0]],dtype=np.float64)
    require(np.max(np.abs(M-expected))<=1e-15 and np.max(np.abs(m-np.array([1,1,1,0])/math.sqrt(2*math.pi)))<=1e-15,
        'Identical/opposite/zero columns exact half-normal moments')
    W = np.eye(2,dtype=np.float32); m,M,_,_ = first_moments(W)
    require(abs(M[0,1]-1/(2*math.pi))<=1e-15,'Independent first-layer raw moment')
    checks = []
    for n,rb,Epositive,Epositive2 in ((1,math.sqrt(2/math.pi),.5,.5),
                                    (2,math.sqrt(math.pi/2),1/math.pi,.25)):
        ms,M2 = 1/math.sqrt(2*math.pi),.5
        mean_s,second_s = rb*Epositive,rb*rb*Epositive2
        defect = .3*(mean_s-ms)-.7*(n/(rb*rb)*second_s-M2)
        require(abs(mean_s-ms)<=1e-15 and abs(defect)<=1e-15,'Exact radial conditional-center identity')
        checks.append(dict(dimension=n,mean_defect=mean_s-ms,quadratic_center_defect=defect))
    stage.record['tiny_certificate'] = stage.json('tiny_certificate.json',dict(passed=True,completed=True,
        radius=radius,radius_records=certificate,first_layer_domain=domain,radial_checks=checks,
        midpoint_endpoint_values=midpoints.tolist(),endpoint_normals=normals.tolist(),
        target_accessed=False,common_bias_certified=False))
    return radius


def coefficients(W,h,stage):
    began = time.monotonic()
    stats = dict(gaussian_pair_maps=0,cholesky_dimensions=[],mean_roundoff_lift_max=0.,variance_roundoff_lift_max=0.,
        tail_roundoff_lift_max=0.,factor_rounding_relative_max=0.,factor_zeroed_variance_coordinates=0)
    m0,M0,C0,initial_domain = first_moments(W[0]); del M0
    state = h.factor_state(dict(id=0,p=1.,m=m0,C=C0),stats); gates = []
    for layer in range(1,16):
        mu = W[layer].astype(np.float64).T@state['m']
        B = state['L'].T@W[layer]
        if layer==15:
            B64 = B.astype(np.float64); variance = np.sum(B64*B64,axis=0)
            provider_final = h.marginal(mu,variance,stats)[0]; break
        raw = (B.T@B).astype(np.float64); raw = .5*(raw+raw.T)
        require(np.isfinite(raw).all() and np.all(np.diag(raw)>=0),'Original rounded affine covariance domain')
        gates.append(h.marginal(mu,np.diag(raw).copy(),stats)[-1])
        mean,cov = h.gaussian_map(mu,raw,stats)
        state = h.factor_state(dict(id=0,p=1.,m=mean,C=cov),stats); stage.guard()
    require(len(gates)==14 and stats['gaussian_pair_maps']==14 and len(stats['cholesky_dimensions'])==15,
        'One fixed full-rank coefficient trajectory')
    A = W[1].copy()
    for layer in range(1,15): A = (A*gates[layer-1].astype(np.float32)[None,:])@W[layer+1]
    A = np.ascontiguousarray(A,dtype=np.float32); A64 = A.astype(np.float64)
    ms = m0@A64; pushed = C0@A64
    vs = np.sum(A64*pushed,axis=0); M2 = vs+ms*ms
    require(np.isfinite(A).all() and np.isfinite(vs).all() and np.all(vs>=0),'Stored-A exact-center variance domain')
    active = vs>0
    require(np.all(ms[~active]==0) and np.all(M2[~active]==0),'Deterministic homogeneous control must vanish')
    sd = np.sqrt(np.where(active,vs,1.)); t = mu/sd
    require(np.isfinite(t).all(),'Finite final coefficient standardized mean')
    beta = np.where(active,np.exp(-.5*t*t)/(2*math.sqrt(2*math.pi)*sd),0.)
    ell = np.where(active,ndtr(t)-2*beta*ms,0.)
    require(np.isfinite(beta).all() and np.isfinite(ell).all(),'Fixed control coefficient domain')
    stats.update(first_layer=initial_domain,provider_fp32_products=29,projection_fp32_products=14,
        center_fp64_products=1,exact_gram_fp64_products=1,pristine_C0_used_for_center=True,
        no_provider_covariance_used_for_center=True,stored_A_dtype=A.dtype.str,elapsed_s=time.monotonic()-began)
    stage.guard()
    return dict(A=A,ms=ms,vs=vs,M2=M2,ell=ell,beta=beta,m0=m0,mu_G15=mu,provider_final=provider_final),stats


def control(s,coeff,radius):
    return coeff['ell']*(s-coeff['ms'])+coeff['beta']*((WIDTH/(radius*radius))*s*s-coeff['M2'])


def sample(entry,sampling,bits,radius):
    rng = np.random.Generator(np.random.PCG64DXSM(entry['seed']))
    sampling['tail_rng'] = np.random.Generator(np.random.PCG64DXSM(entry['tail_seed']))
    z,u = sampling['_scrambled_sobol_normals'](DIRECTIONS,WIDTH,rng,bits)
    require(np.isfinite(z).all(),'Finite un-clipped PPF')
    norm = np.sqrt(np.sum(z*z,axis=1)); require(np.all(norm>0),'Nonzero sphere normalization')
    positive = (radius/norm[:,None])*z
    positive32 = positive.astype(np.float32); X = np.concatenate((positive32,-positive32))
    require(X[4096:].tobytes()==(-X[:4096]).tobytes() and np.isfinite(X).all(),'Exact FP32 antipodes')
    diagnostic = dict(minimum_midpoint=float(u.min()),maximum_midpoint=float(u.max()),
        maximum_abs_normal=float(np.max(np.abs(z))),maximum_radius_defect_fp64=float(np.max(np.abs(
            np.sqrt(np.sum(positive*positive,axis=1))-radius))),
        maximum_radius_defect_fp32=float(np.max(np.abs(np.sqrt(np.sum(
            positive32.astype(np.float64)*positive32.astype(np.float64),axis=1))-radius))),
        exact_antipodes=True,finite_grid_not_continuous=True,tail_bits_per_coordinate=20,ppf_clipped=False)
    small64 = np.concatenate((positive[:8],-positive[:8])) if entry['replicate']==0 else None
    return X,small64,diagnostic


def reference_paths(X,W,A):
    hidden = X.copy(); outputs = []
    for layer in range(16):
        hidden = np.maximum(hidden@W[layer].astype(np.float64),0.)
        if layer==0: s = hidden@A.astype(np.float64)
        outputs.append(hidden.copy())
    return np.stack(outputs),s


def errors(actual,expected):
    d = np.asarray(actual,dtype=np.float64)-np.asarray(expected,dtype=np.float64)
    return dict(maximum_absolute=float(np.max(np.abs(d))),rms=float(np.sqrt(np.mean(d*d))),
        squared_error=float(np.sum(d*d)))


def rule(entry,Wt,At,W,coeff,sampling,bits,radius,stage):
    began = time.monotonic(); X,small64,sampling_diagnostic = sample(entry,sampling,bits,radius)
    selected = np.r_[0:8,4096:4104]
    actual_small = []; means = []
    with torch.inference_mode():
        H = torch.from_numpy(X).to('mps')
        for layer in range(16):
            H = torch.relu(H@Wt[layer]); torch.mps.synchronize()
            host = H.cpu().numpy()
            require(np.isfinite(host).all() and np.all(host>=0),'All actual path-layer domains')
            means.append(np.mean(host,axis=0,dtype=np.float64))
            if small64 is not None: actual_small.append(host[selected].copy())
            if layer==0:
                S = H@At; torch.mps.synchronize(); s32 = S.cpu().numpy().copy(); del S
            if layer==15: final32 = host.copy()
            stage.guard()
        del H,host
    s64 = s32.astype(np.float64); correction = control(s64,coeff,radius)
    residual = final32.astype(np.float64)-correction
    baseline = np.stack(means); baseline[0] = coeff['m0']
    raw = baseline.copy(); raw[-1] = np.mean(residual,axis=0,dtype=np.float64)
    clipped = np.maximum(raw,0.); served = clipped.astype(np.float32)
    require(np.isfinite(raw).all() and np.isfinite(served).all(),'Finite full rule outputs')
    arrays = stage.bundle(f"rule_{entry['rule_index']:02d}_means.npz",dict(raw=raw,served=served,
        uncorrected=baseline,control_mean=np.mean(correction,axis=0,dtype=np.float64)))
    diagnostic = dict(**sampling_diagnostic,clamped_coordinates=int(np.count_nonzero(raw<0)),
        clamp_defect=errors(clipped,raw),return_cast_defect=errors(served,clipped),
        elapsed_s=time.monotonic()-began,coefficient_fit=False,empirical_recentering=False,
        all16_domains_qualified=True,paths=PATHS,arrays=arrays)
    if small64 is not None:
        cpu64,s_cpu64 = reference_paths(small64,W,coeff['A'])
        cpu32,s_cpu32 = reference_paths(X[selected].astype(np.float64),W,coeff['A'])
        actual = np.stack(actual_small); sc = s64[selected]
        q64,q32,qmps = control(s_cpu64,coeff,radius),control(s_cpu32,coeff,radius),correction[selected]
        evidence = stage.bundle(f"reference_rank{entry['rank']:04d}.npz",dict(input_fp64=small64,input_fp32=X[selected],
            cpu64_paths=cpu64,cpu64_on_fp32_input=cpu32,mps_paths=actual,
            cpu64_s=s_cpu64,cpu64_on_fp32_s=s_cpu32,mps_s=sc,
            cpu64_control=q64,cpu64_on_fp32_control=q32,mps_control=qmps,
            cpu64_residual=cpu64[-1]-q64,mps_residual=actual[-1].astype(np.float64)-qmps))
        detail = dict(arrays=evidence,per_layer_input_cast=[errors(cpu32[k],cpu64[k]) for k in range(16)],
            per_layer_mps_products=[errors(actual[k],cpu32[k]) for k in range(16)],
            per_layer_end_to_end=[errors(actual[k],cpu64[k]) for k in range(16)],
            input_cast_s=errors(s_cpu32,s_cpu64),mps_product_s=errors(sc,s_cpu32),end_to_end_s=errors(sc,s_cpu64),
            input_cast_control=errors(q32,q64),mps_product_control=errors(qmps,q32),end_to_end_control=errors(qmps,q64),
            end_to_end_residual=errors(actual[-1].astype(np.float64)-qmps,cpu64[-1]-q64),
            population_common_bias_certified=False,absolute_limit=1e-5)
        diagnostic['reference'] = stage.json(f"reference_rank{entry['rank']:04d}.json",detail)
        stage.json(f"rule_{entry['rule_index']:02d}_attempt.json",diagnostic)
        checks = detail['per_layer_end_to_end']+[detail[k] for k in ('end_to_end_s','end_to_end_control','end_to_end_residual')]
        require(all(v['maximum_absolute']<=1e-5 for v in checks),'Fixed16-path all-layer/control absolute fidelity')
    else: stage.json(f"rule_{entry['rule_index']:02d}_attempt.json",diagnostic)
    # Only detached small outputs escape; all full path tensors/arrays retire here.
    return raw,served,baseline,diagnostic


def variance_summary(values):
    values = np.asarray(values,dtype=np.float64)
    require(values.shape==(16,1024) and np.isfinite(values).all(),'Sixteen complete rule-mean vectors')
    centered = values-np.mean(values,axis=0)
    value = float(np.sum(centered*centered)/(15*1024))
    differences = values[::2]-values[1::2]
    pairs = np.mean(differences*differences,axis=1)/2
    return dict(one_rule_variance=value,replicate_denominator=15,not_divided_by_rules_or_paths=True,
        pair_difference_energies=pairs.tolist(),pair_energy_mean=float(np.mean(pairs)),
        pair_energy_standard_error=float(np.std(pairs,ddof=1)/math.sqrt(8)),
        confidence_interval_claimed=False,neurons_are_not_independent_replicates=True)


def main():
    global np,torch,ndtr,ndtri
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run',action='store_true')
    for key in ('source','watcher','protocol'): parser.add_argument('--'+key+'-sha256')
    args = parser.parse_args()
    if not args.run: parser.print_help(); return 0
    stage = Stage(args); old_alarm = None
    try:
        stage.admission()
        import numpy as np
        import scipy
        from scipy.special import ndtr,ndtri
        import torch
        require(dict(numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__)==VERSIONS,
            'Exact numerical library frame')
        require('threadpoolctl' not in sys.modules,'Never import threadpoolctl in AP')
        require(torch.backends.mps.is_available(),'One genuine MPS device, no fallback')
        torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.set_grad_enabled(False)
        stage.torch = torch
        recommended = int(torch.mps.recommended_max_memory())
        require(recommended>0,'Positive device memory recommendation')
        fraction = min(1.,DRIVER/recommended)
        torch.mps.set_per_process_memory_fraction(fraction)
        stage.record['mps_allocator'] = dict(configured=True,recommended_max_memory_bytes=recommended,
            configured_fraction=fraction,api='torch.mps.set_per_process_memory_fraction',low_watermark_ratio='0.1')
        stage.record['environment']['versions'] = VERSIONS
        def timeout(signum,frame): raise TimeoutError('Fixed information stage600s')
        old_alarm = signal.signal(signal.SIGALRM,timeout)
        signal.setitimer(signal.ITIMER_REAL,max(.001,LIMIT-(time.monotonic()-stage.start)))
        with torch.inference_mode():
            radius = tiny_checks(stage); h,sampling = numerical_helpers(stage)
            stage.json('helper_inverse.json',dict(provider=stage.record['provider_source_inverse'],
                sampler=stage.record['sampler_source_inverse']))
            bits = sampling['_sobol_direction_bits'](sampling['_sobol_direction_numbers'](WIDTH))
            for rank in RANKS:
                rank_began = time.monotonic()
                info = stage.weights[rank]; require(sha(info['path'],stage.guard)==info['sha256'],'Weights rechecked before decode')
                W = np.load(info['path'],allow_pickle=False)
                require(W.shape==(16,1024,1024) and W.dtype==np.float32 and np.isfinite(W).all()
                    and weight_payload_sha(W)==info['array_sha256'],'Exact canonical weight dtype/shape/payload identity')
                coeff,stats = coefficients(W,h,stage)
                coefficient_meta = stage.bundle(f'coefficients_rank{rank:04d}.npz',coeff)
                stage.json(f'coefficients_rank{rank:04d}.json',dict(stats=stats,arrays=coefficient_meta,weights=info,
                    radius=radius,no_truth_or_fitting=True,center_uses_actual_stored_A=True))
                Wt = torch.from_numpy(W).to('mps'); At = torch.from_numpy(coeff['A']).to('mps')
                raws,served_values,baselines = [],[],[]
                for entry in (v for v in ROSTER if v['rank']==rank):
                    raw,served,baseline,diagnostic = rule(entry,Wt,At,W,coeff,sampling,bits,radius,stage)
                    raws.append(raw[-1].copy()); served_values.append(served[-1].copy()); baselines.append(baseline[-1].copy())
                    gc.collect(); torch.mps.synchronize(); torch.mps.empty_cache()
                    marker = dict(**entry,passed=True,completed=True,rule_retired=True,path_state_released=True,
                        all16_domains_qualified=True,exact_antipodes=True,truth_accessed=False,resources=stage.guard(),
                        arrays=diagnostic['arrays'],coefficient_sha256=coefficient_meta['sha256'],
                        elapsed_s=time.monotonic()-stage.start)
                    item = stage.json(f"rule_{entry['rule_index']:02d}_complete.json",marker)
                    stage.record['rule_records'].append(dict(**entry,**item))
                summary = dict(served=variance_summary(served_values),unclamped=variance_summary(raws),
                    uncorrected=variance_summary(baselines),coefficient_sha256=coefficient_meta['sha256'],
                    native_bill_measured=False,common_bias_certified=False,elapsed_s=time.monotonic()-rank_began)
                require(all(hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()==
                    coefficient_meta['arrays'][k]['raw_sha256'] for k,v in coeff.items())
                    and weight_payload_sha(W)==info['array_sha256'],
                    'All coefficient and weight numerical bytes unchanged across all16 rules')
                summary['coefficient_and_weight_bytes_unchanged'] = True
                summary['planning_adjusted_variance'] = summary['served']['one_rule_variance']*PLANNED_F/2**41
                stage.record['rank_summaries'][str(rank)] = summary
                stage.json(f'summary_rank{rank:04d}.json',summary)
                del Wt,At,W,coeff; gc.collect(); torch.mps.synchronize(); torch.mps.empty_cache(); stage.guard()
            projected = math.fsum(v['planning_adjusted_variance'] for v in stage.record['rank_summaries'].values())/4
            stage.record.update(mean_planning_adjusted_variance=projected,
                competitive_information_signal=projected<3e-9,information_threshold=3e-9,
                information_signal_is_not_raw_score=True,common_bias_certified=False,
                current_configuration_changes=0,all_four_rank_summaries_sealed=True)
            stage.record['information_seal'] = stage.json('information_seal.json',dict(
                rule_records=stage.record['rule_records'],rank_summaries=stage.record['rank_summaries'],
                mean_planning_adjusted_variance=projected,truth_accessed=False,scientific_advancement=False))
            stage.finish()
    except BaseException:
        stage.record.update(passed=False,completed=False,error=traceback.format_exc())
    finally:
        if old_alarm is not None:
            signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,old_alarm)
        stage.close()
    return 0 if stage.record['passed'] else 1


if __name__=='__main__': raise SystemExit(main())
