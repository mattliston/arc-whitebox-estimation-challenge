"""Stdlib adapter for admitted terminal-head exports and existing native CHECK8."""
from pathlib import Path
import importlib.util,json,sys

D=Path(__file__).resolve().parent
H=D.parent
RANKS=[52,58,64,70,76,80,84,88]
PROTOCOL_SHA='93eea5cc3a3ff6d7e889798ba57e78b73f466500b23e8b3801b3897537e06ef3'
COMMON_SHA='626cff0af1a0250e0ce58b5833c09d49b10e3ae444cd5780bb033f81b82c4150'


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def binding():
    s=module(D/'common.py','coadapt_check8_study')
    pp=s.pin(D/'protocol.json');assert pp['sha256']==PROTOCOL_SHA
    _,p=s.plan(pp['path'],pp['sha256']);s.prerequisites(p)
    base=s.pin(H/'hill_climb_rrpi_smooth_kink_v1/prepare.py');assert base['sha256']==COMMON_SHA
    shared=module(base['path'],'coadapt_check8_saved_controls').common()
    assert p['runtime']==shared['runtime'] and p['guard_source']==shared['guard_source']
    cvp=s.pin(D/'cv_result.json');cv=s.read(s.checked(cvp))
    ep=s.pin(D/'fit_export.json');ex=s.read(s.checked(ep))
    assert cv['completed'] and ex['completed'] and cv['protocol']==ex['protocol']==pp
    assert cv['fit_source']==ex['fit_source'] and ex['cv_result']==cvp
    s.checked(ex['fit_source'])
    assert ex['ridge']==p['ridge']==.001 and ex['train']==p['train']
    assert ex['all_other_parameter_bytes_exact']
    wanted=['rrpi','k4'] if cv['physical_admitted'] else ['rrpi']
    assert cv['physical_admitted'] or cv['control_admitted']
    assert set(ex['exports'])==set(wanted)
    seal=s.pin(D/'fit_seal.json');sealed=s.read(s.checked(seal))
    assert sealed['protocol']==pp and sealed['fit_source']==ex['fit_source'] and sealed['cv_gates']==p['cv_gates']
    commands=s.pin(D/'fit_commands.json');jobs=s.read(s.checked(commands))
    assert jobs['protocol']==pp and jobs['seal']==seal
    guards={}
    for mode in ('cv','export'):
        command=[p['runtime'],'-B',ex['fit_source']['path'],'--'+mode,pp['path'],pp['sha256']]
        assert jobs['jobs'][mode]['command']==command
        guards[mode]=s.guard(D/(mode+'_guard.json'),command,120)
    editor=module(p['sources']['source_edit']['path'],'coadapt_check8_head_editor')
    exports={}
    for arm in wanted:
        value=ex['exports'][arm];assert value['source']==p['arms'][arm]
        assert value['eligible_standalone']==bool(cv['control_admitted'] if arm=='rrpi' else cv['physical_admitted'])
        original=s.checked(value['source']).read_text();candidate=s.checked(value['candidate']).read_bytes()
        replay,inverse,_,_=editor.edit(original,editor.terminal_head(candidate))
        assert replay==candidate and inverse==original.encode()
        exports[arm]=value
    return s,shared,dict(protocol=pp,cv_result=cvp,fit_export=ep,fit_guards=guards,fit_seal=seal,fit_commands=commands,
                        fit_source=ex['fit_source'],exports=exports,study_common=s.pin(D/'common.py'),
                        saved_common=base)


def configuration(path,digest):
    s,shared,bound=binding();pp=s.pin(path);assert pp['sha256']==digest
    c=s.read(s.checked(pp))
    assert c['schema']=='coadapt-native-check8-v1' and c['ranks']==RANKS
    assert c['binding']==bound and c['builder']==s.pin(__file__) and c['assessor']==s.pin(D/'assess_check8.py')
    for k,v in shared.items():assert c[k]==v
    assert c['parent']==s.read(bound['protocol']['path'])['arms']['rrpi']
    assert c['gates']==dict(raw_pct_strictly_below=0.,adjusted_pct_strictly_below=0.,physical_must_beat_fitted_control=True)
    assert set(c['arms'])==set(bound['exports']) and c['bank_gain_pct_min']==1.
    for arm,a in c['arms'].items():
        assert a['source']==bound['exports'][arm]['candidate'] and a['ranks']==RANKS
        assert a['command']==[c['runtime'],'-B',c['driver']['path'],a['source']['path'],','.join(map(str,RANKS)),a['tag']]
        assert a['limits']==dict(seconds=300,rss_bytes=7500000000,added_swap_bytes=0)
    return pp,c


def main():
    assert len(sys.argv)==1
    s,shared,bound=binding();arms={}
    for name,export in bound['exports'].items():
        tag=D.name+'_check8_'+name+'_v1';guard=str(D/('check8_'+name+'_guard.json'))
        command=[shared['runtime'],'-B',shared['driver']['path'],export['candidate']['path'],','.join(map(str,RANKS)),tag]
        arms[name]=dict(source=export['candidate'],ranks=RANKS,tag=tag,output=str(H/tag),driver=shared['driver'],
            command=command,guard=guard,limits=dict(seconds=300,rss_bytes=7500000000,added_swap_bytes=0),
            guard_command=[shared['light'],'-B',shared['guard_source']['path'],'--record',guard,
                           '--seconds','300','--rss-limit','7500000000',*command],log=str(D/('check8_'+name+'.log')))
    c=dict(schema='coadapt-native-check8-v1',ranks=RANKS,arms=arms,binding=bound,
           parent=s.read(bound['protocol']['path'])['arms']['rrpi'],builder=s.pin(__file__),assessor=s.pin(D/'assess_check8.py'),
           gates=dict(raw_pct_strictly_below=0.,adjusted_pct_strictly_below=0.,physical_must_beat_fitted_control=True),
           bank_gain_pct_min=1.,adaptive_reused_train_check_not_independent_validation=True,
           full40_own_resources_and_serving_still_required=True,**shared)
    path=D/'native_check8_plan.json';s.once(path,c);pp=s.pin(path)
    command=[c['runtime'],'-B',c['assessor']['path'],pp['path'],pp['sha256']]
    s.once(D/'check8_commands.json',dict(plan=pp,model_jobs=[dict(name=k,command=v['guard_command'],log=v['log']) for k,v in arms.items()],
        assessment=[c['light'],'-B',c['guard_source']['path'],'--record',str(D/'check8_assess_guard.json'),
                    '--seconds','120','--rss-limit','7500000000',*command],assessment_log=str(D/'check8_assess.log'),
        thread_env=dict.fromkeys(s.THREADS,'1'),root_only_serial=True,
        launch_prerequisite='Call prepare_check8.configuration(plan_path,sha) immediately before each launch.',
        no_automatic_full40_resources_or_submission=True))
    print(json.dumps(dict(plan=pp,commands=s.pin(D/'check8_commands.json'))))


if __name__=='__main__':main()
