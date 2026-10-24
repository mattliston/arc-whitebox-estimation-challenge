"""Original production-five timing gates and actual saved all16 comparisons."""
import hashlib
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parent
TIMING=HERE.parent/'current_comp_partial_d6_v1/production_watch_v2.py'
TIMING_SHA='5e375679330dfdc0623140aede5a5bb3851f71588387a8e65d54e6d889f2361f'
OUTPUT=HERE/'pair_checks.py'
OUTPUT_SHA='341ab3e95afa891d902d1a68372103f039e8a2fb8aa5f563add3c84b7f07040b'
PINS={TIMING:TIMING_SHA,OUTPUT:OUTPUT_SHA}
RANKS=(35,100,500,3016500,35)
MAPPING=(0,1,7,8,14)


def require(ok,message):
    if not ok:
        raise RuntimeError(message)


def load(path,digest,name):
    require(not path.is_symlink() and hashlib.sha256(path.read_bytes()).hexdigest()==digest,'Frozen original actual-output/timing helper')
    spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module); return module


_timing=load(TIMING,TIMING_SHA,'gain_production_timing_metadata')
_output=load(OUTPUT,OUTPUT_SHA,'gain_production_output_metadata')
check_bill=_timing.check_bill
raw_f32=_output.raw_f32


def compare(native,rows,paths,tick):
    require(native['full_model_calls']==native['retired_call_count']==15 and len(rows)==len(paths)==5,
            'Actual native15 and complete fresh production-five contexts')
    checks=[]; native_raw=[]; remote_raw=[]
    for call,(rank,index,row,path) in enumerate(zip(RANKS,MAPPING,rows,paths,strict=True)):
        check_bill(row,call); trial=native['trials'][index]; prep=native['input_preparation'][index]
        require(trial['rank']==prep['rank']==rank and prep['weight_file_sha256']==
                native['fixture_bindings']['weights_'+str(rank)]['sha256'],'Exact canonical native/production physical input')
        a,av=raw_f32(trial['output'],npy=True); b,bv=raw_f32(path)
        require(hashlib.sha256(a).hexdigest()==trial['output_bytes_sha256']
            and hashlib.sha256(b).hexdigest()==row['output_sha256'],'Actual same-source all16 output payloads')
        delta=[x-y for x,y in zip(av,bv,strict=True)]
        layers=[max(abs(x) for x in delta[i*1024:(i+1)*1024]) for i in range(16)]
        checks.append(dict(call=call,rank=rank,native_call=index,all16_max_abs=max(layers),
            layer_max_abs=layers,output_byte_exact=a==b,original_all16_vector_gate_passed=max(layers)<=1e-5,
            native_flops=trial['flops'],production_flops=row['flops'],complete_flops_delta=trial['flops']-row['flops'],
            final_prediction_difference_mse=sum(x*x for x in delta[-1024:])/1024,
            weight_file_sha256=prep['weight_file_sha256']))
        native_raw.append(a); remote_raw.append(b); tick()
    require(native_raw[0]==native_raw[4] and remote_raw[0]==remote_raw[4], 'Actual complete own repeats byte-exact')
    require(all(x['complete_flops_delta']==0 for x in checks),'Actual same-backend own complete tariffs exact')
    return dict(checks=checks,all16_vector_gate_passed=all(x['original_all16_vector_gate_passed'] for x in checks),
        native_repeat_byte_exact=True,production_repeat_byte_exact=True,all_complete_flops_exact=True,
        native_initialization_flops=native['trials'][0]['flops']-native['trials'][14]['flops'],
        production_initialization_flops=rows[0]['flops']-rows[4]['flops'],
        raw_error_measured=False,adjusted_score_measured=False,truth_accessed=False)
