"""Inert admission for one fixed current-shared anchored TRAIN20 proposal.

No numerical import, array decoding, target opening, model call, optimizer or
stage at import or admission. The returned model is the exact qualification
model. Only the future independently watched training worker may decode the
declared TRAIN labels after complete admission. No CHECK labels are admitted.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import types

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
H15 = NEW/'current_c2_terminal_quadratic_information_v1'
QUALIFIER = HERE/'qualification.py'
QUALIFIER_SHA = '4ed15b49ff9e6d0a58c98e8eb64fee2c64f1b19d320b3cadad2870f620a9772d'
QUALIFIER_PROTOCOL_SHA = 'd29c711c05ba7e377aa646dd3f4da8a687cb01f2f72d456f72c2cf0a44994028'
C2_SHA = 'c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea'
MANIFEST = H15/'train_capture_v2/manifest.json'
MANIFEST_SHA = '7e0c4274d3f22dd1924d98c611360eaefd88c2491141e700c2fc0ae0502b84ae'
LABEL_INDEX = NEW/'joint_prefix_final_fit_v1/manifest.json'
LABEL_INDEX_SHA = 'd61757d28236e3d35d87415962d1a57d18ff38ae189651df52df5e3412b9a971'
TRAIN = (35,36,38,40,41,43,44,45,46,47,49,50,51,52,53,54,55,56,57,58)
SCHEMA = 'current_shared384_anchored_p_train_admission_v1'
WHOLE_SECONDS, RSS_BYTES, RESEARCH_LIMIT = 1500., 8_000_000_000, 160_000_000_000_000
QUALIFICATION = (
 '548dff420376c72ebd4d5bc30938ebf989768ad2841f16b41beac17b1eb821fa',
 '478e4290478adff2db6e594fd355c8382562e8234f0d7f4e5f7e7b6ade9a930d',
 '3ab36790a371b204b374b96b1eb3614492b4433d9c70980bc4160ccd8634bcfe')
QUALIFICATION_WATCH = (
 '716d84bfc4498743c43a9046ac95992a103cec47c310b4dbde2677b8a9ad4768',
 '77f130db990c6691ebad7884d4558e2130b8e73fcb1ab46de4a7bbb464029f5b',
 '18ed9a39bb4f5aabb0109db3a74f3154d6d3f1599a08179a9ebbec5f8f6b75c3')
CAPTURE_WATCH = (
 'e2d79482eef9aa1d60275a84551f3c618303769805fa9e8359c9bc6b7e4b190c',
 '5374a6a94d09a297a7c3cf2d6b92c089d4d9770a869bae5fcd5f7b741d72b1b7',
 'ffc2f04467dabaccda5aad44340eb69bde6ec1fdcae2b3e147db4e73dfccf3e3')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size < 32*1024**2,
            'Bounded regular metadata only')
    return json.loads(path.read_text())


def _not_truth(path):
    path = Path(path)
    return 'truthpost' not in path.parts and not (
        path.suffix in ('.npy','.npz') and 'truth' in path.name.lower())


class _Inventory:
    """One-pass byte hashing; source/fixture admission cannot open truth files."""
    def __init__(self, tick):
        self.tick, self.cache = tick, {}
        self.sources, self.inputs = {}, {}

    def bind(self, path, expected, source=False):
        path = Path(path)
        require(path.is_absolute() and path.is_file() and not path.is_symlink()
                and _not_truth(path) and re.fullmatch('[0-9a-f]{64}',expected or ''),
                'Regular explicit source/input SHA, never labels')
        key = str(path)
        if key not in self.cache:
            before = path.stat(); h = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(1<<20),b''):
                    h.update(block)
                    if self.tick is not None: self.tick()
            after = path.stat()
            require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),
                    'Unchanged file across admission read')
            self.cache[key] = h.hexdigest()
        require(self.cache[key]==expected,'Exact admitted bytes: '+key)
        for group in (self.sources,self.inputs):
            require(key not in group or group[key]==expected,'No conflicting lineage digest')
        (self.sources if source else self.inputs)[key] = expected

    def merge(self, record, source_base=None):
        for name in ('source_bindings','inherited_source_bindings','input_bindings','artifacts'):
            for p,h in record.get(name,{}).items():
                path = Path(p)
                if not path.is_absolute():
                    require(source_base is not None and path.name==p and 'source' in name,
                            'Only actual direct source basenames may be resolved')
                    path = source_base/path
                self.bind(path,h,source='source' in name)

    def seals(self, folder, expected):
        paths = [folder/n for n in ('result.json','completion.json','finalization.json')]
        require(len(expected)==3,'All three actual receipt hashes')
        for p,h in zip(paths,expected): self.bind(p,h)
        r,c,f = map(read,paths)
        require(all(x.get('passed') is True and x.get('completed') is True for x in (r,c,f))
                and c['result_sha256']==f['result_sha256']==expected[0]
                and f['completion_sha256']==expected[1],'Actual positive complete three-seal chain')
        return r


def work_certificate(qualifier):
    """Worst original geometry, not the measured35 selector census.

    Reuse ONLY the authenticated scalar pricing function with an independent
    globals dictionary and larger *pricing* limit. Neither its original globals
    nor its numerical model is changed. Every arithmetic line is identical.
    The anchor stage has four forward-equivalents and one reverse per network:
    baseline + original/recomputed training + terminal, never five/two.
    """
    n,r = 1024,384
    cells = {f'{l}:{slot}':dict(count=n) for l in range(16) for slot in (0,1)}
    shared = {}
    for l in range(16):
        stops = [2*n*k for k in range(1,l+1)]
        old = 0 if l<5 else 2*n*(min(l,14)-4)
        shared[str(l)] = dict(columns=2*n*l,old_columns=old,stops=stops)
    # Cold-complement work is bounded separately below; the all-live matrix
    # dimensions maximize all remaining positive-polynomial product tariffs.
    scalar_globals = dict(qualifier.work_certificate.__globals__)
    scalar_globals['WORK_LIMIT'] = 10**18
    scalar = types.FunctionType(qualifier.work_certificate.__code__,scalar_globals,
                               name='unchanged_scalar_price_at_worst_geometry')
    full = scalar(dict(selection_cells=cells,shared_cells=shared))
    layers = []
    for row in full['layers']:
        # Every potentially missing cold block is conservatively allowed at
        # its full1024 inner width in addition to the full-live upper above.
        # Three resident covariance blocks and one selected star product;
        # each has one reverse (fixed physical W is not trainable).
        cold_forward = (3*2*512*n*512+2*n*n*n) if row['layer'] else 0
        forward = (row['matrix_forward']+row['native_forward_extra']+
                   row['QR_forward']+row['pointwise_forward']+cold_forward)
        reverse = (row['matrix_reverse']+row['QR_reverse']+
                   8*row['pointwise_forward']+cold_forward)
        layers.append(dict(layer=row['layer'],forward_upper=forward,reverse_upper=reverse,
                           one_network_baseline_train_terminal_upper=4*forward+reverse))
    # The reviewed8BF setup/export allowance per three-card network is retained
    # twenty times (including a workspace per net, although the worker uses one)
    # plus explicit100MF owned-ingress allowance per net. A further1BF covers
    # all five Adam/clip steps, loss weights/anchors, metadata and final payload.
    per_net = sum(x['one_network_baseline_train_terminal_upper'] for x in layers)+8_100_000_000
    complete = 20*per_net+1_000_000_000
    require(complete<RESEARCH_LIMIT,'Complete max-geometry60-forward/20-backward source cap')
    return dict(layers=layers,per_network_upper=per_net,complete_source_upper=complete,
        limit=RESEARCH_LIMIT,baseline_forwards=20,training_forwards=20,
        maximum_checkpoint_recomputed_forwards=20,backwards=20,terminal_forwards=20,
        updates=5,setup_input_export_upper=20*8_100_000_000,optimizer_loss_upper=1_000_000_000,
        maximum_columns=2*n*15,maximum_old_columns=2*n*10,rank=r,age=4,
        cold_complement_separately_charged=True,changed_selector_counts_allowed=True,
        scalar_price_owner_sha256=QUALIFIER_SHA,actual_flopscope_bill_claimed=False)


def admission(source_sha256, protocol_sha256, tick=None):
    """Return inert model and exact20 native/weight/cost/declared-target rows.

    The future worker must additionally pin its own worker/watch source owners,
    verify/hash each TRAIN truth immediately before decoding and again after all
    owned numerical contexts, and never pass truth paths to this source hasher.
    """
    forbidden = {'numpy','torch','flopscope','scipy','threadpoolctl'}
    require(not forbidden&set(sys.modules),'Fresh metadata-only admission')
    inv = _Inventory(tick)
    for p,h in ((Path(__file__).resolve(),source_sha256),(HERE/'TRAIN_PROTOCOL.md',protocol_sha256),
                (QUALIFIER,QUALIFIER_SHA),(HERE/'QUALIFICATION_PROTOCOL.md',QUALIFIER_PROTOCOL_SHA)):
        inv.bind(p,h,source=True)
    q = inv.seals(HERE/'qualification_v2',QUALIFICATION)
    qw = inv.seals(HERE/'qualification_watch_v2',QUALIFICATION_WATCH)
    require(q['execution_integrity_qualified'] and q['all_owners_retired']
            and q['full_model_calls']==3 and q['backwards']==2 and q['parameter_updates']==0
            and q['truth_accessed'] is False and q['native_gradient_certified'] is False
            and q['original_native_fidelity_passed'] is False and q['NADJ_requalified'] is False
            and q['qualification_scope']=='same_surrogate_execution_integrity_only',
            'Same-surrogate execution qualification only; old failures stay failed')
    require(qw['whole_owned_tree_retired'] and qw['maximum_swap_growth_bytes']==0
            and qw['maximum_combined_rss_bytes']<RSS_BYTES and qw['maximum_interval_s']<=1
            and qw['elapsed_s']<120 and tuple(qw['child_'+k+'_sha256'] for k in
            ('result','completion','finalization'))==QUALIFICATION,'Exact bounded qualifier/watch join')
    inv.merge(q); inv.merge(qw,HERE)
    require(q['source_bindings'][str(QUALIFIER)]==QUALIFIER_SHA,'Actual qualified helper identity')
    spec = importlib.util.spec_from_file_location('anchored_train_qualified_helper',QUALIFIER)
    qualifier = importlib.util.module_from_spec(spec); spec.loader.exec_module(qualifier)
    prepared = qualifier.prepare(QUALIFIER_SHA,QUALIFIER_PROTOCOL_SHA,tick)
    inv.merge(prepared)
    inv.bind(MANIFEST,MANIFEST_SHA); m = read(MANIFEST)
    w = inv.seals(H15/'train_capture_watch_v2',CAPTURE_WATCH)
    require(m['schema']=='current_c2_terminal_quadratic_train_capture_manifest_v2'
            and m['train']==list(TRAIN) and len(m['rows'])==20 and m['candidate_sha256']==C2_SHA
            and not m['truth_accessed'] and not m['fit_performed'] and not m['score_measured'],
            'Fixed20 target-blind current-c2 primary examples, no failed/repeat row added')
    require(w['manifest_path']==str(MANIFEST) and w['manifest_sha256']==MANIFEST_SHA
            and w['all20_primary_captures_sealed'] and w['cleanup_passed']
            and w['whole_owned_tree_retired'] and w['maximum_swap_growth_bytes']==0
            and w['maximum_combined_rss_bytes']<RSS_BYTES and w['maximum_interval_s']<=1,
            'Actual complete all20 acquisition resource boundary')
    inv.merge(m); inv.merge(w)
    inv.bind(LABEL_INDEX,LABEL_INDEX_SHA)
    targets = [x for x in read(LABEL_INDEX)['target_sources'] if x['rank'] in TRAIN]
    require(len(targets)==20 and {x['rank'] for x in targets}==set(TRAIN),'Unique original20 TRAIN target index')
    target_map = {x['rank']:x for x in targets}; rows=[]; truths={}
    for ordinal,row in enumerate(m['rows']):
        rank=TRAIN[ordinal]
        require(row['rank']==rank and row['train_ordinal']==ordinal and row['source_call']==0
                and row['fresh_cold'] and row['candidate_sha256']==C2_SHA
                and 0<row['own_cold_flops']<=2**41 and all(row['head_replay'].values()),
                'One original cold physical example per fixed TRAIN ordinal')
        folder=Path(row['result_path']).parent
        require(folder.is_relative_to(H15),'Actual H15 producer descendant')
        child=inv.seals(folder,tuple(row['worker_seals'])); inv.merge(child)
        require(child['candidate_sha256']==C2_SHA and child['own_complete_bills']
                and child['all16_domains_qualified'] and child['all47_selectors_qualified']
                and child['all_parameter_and_estimator_owners_retired'] and not child['truth_accessed'],
                'Every native anchor has its own complete positive production record')
        trial=child['trials'][0]; bill=read(row['model_bill']); ibill=read(row['input_bill'])
        require(trial['call']==0 and trial['rank']==rank and trial['includes_initialization'] is True
                and trial['wall_s']<120 and trial['flops']==bill['flops']==row['own_cold_flops']
                and row['model_bill']==str(folder/'call_00_bill.json')
                and child['artifacts'][row['model_bill']]==row['model_bill_sha256']
                and trial['input_bill']==row['input_bill'] and trial['input_bill_sha256']==row['input_bill_sha256']
                and ibill['weight_file_sha256']==row['weights_sha256']
                and child['initialization_flops']==row['initialization_flops'],
                'Native output, actual cold bill, variable lazy startup and canonical weights agree')
        output=Path(trial['output'])
        served=row['arrays']['served']
        require(output.parent==folder and child['artifacts'][str(output)]==trial['output_sha256']
                and served['shape']==[16,1024] and served['dtype']=='float32'
                and served['bytes']==65536 and served['raw_sha256']==trial['output_bytes_sha256'],
                'All16 native anchor is the sealed actual model output, not terminal H/host')
        inv.bind(output,trial['output_sha256']); inv.bind(row['weights'],row['weights_sha256'])
        target=target_map[rank]
        target_path=P2/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{rank:04d}.npz'
        require(target['weight_file']==row['weights'] and target['target_file']==str(target_path)
                and re.fullmatch('[0-9a-f]{64}',target['target_sha256']),
                'Only same-weight original FP32 mc_post1B TRAIN truth is declared')
        truths[str(target_path)]=target['target_sha256']  # Deliberately no file access.
        rows.append(dict(rank=rank,train_ordinal=ordinal,weights=row['weights'],
            weights_sha256=row['weights_sha256'],native_output=str(output),
            native_output_sha256=trial['output_sha256'],native_output_raw_sha256=trial['output_bytes_sha256'],
            native_output_shape=[16,1024],native_output_dtype='float32',
            own_cold_flops=row['own_cold_flops'],initialization_flops=row['initialization_flops'],
            model_bill=row['model_bill'],model_bill_sha256=row['model_bill_sha256'],
            input_bill=row['input_bill'],input_bill_sha256=row['input_bill_sha256'],
            producer_result=row['result_path'],producer_seals=list(row['worker_seals']),
            producer_watch_folder=row['watch_folder'],producer_watch_scope=row['watch_scope'],
            target=str(target_path),target_sha256=target['target_sha256'],target_key='mc_post',
            target_shape=[16,1024],target_dtype='float32',target_paths_declared_not_opened=True))
    cost=work_certificate(qualifier)
    require(not forbidden&set(sys.modules),'Complete source/admission stayed nonnumerical')
    return dict(schema=SCHEMA,model=prepared['model'],prepared_qualification=prepared,
        source_bindings=inv.sources,input_bindings=inv.inputs,truth_bindings=truths,rows=rows,
        train=list(TRAIN),candidate_sha256=C2_SHA,qualification_seals=list(QUALIFICATION),
        qualification_watch_seals=list(QUALIFICATION_WATCH),capture_manifest=str(MANIFEST),
        capture_manifest_sha256=MANIFEST_SHA,capture_watch_seals=list(CAPTURE_WATCH),
        cost_certificate=cost,whole_seconds=WHOLE_SECONDS,rss_bytes=RSS_BYTES,
        no_completed_call_seconds=120.,maximum_completed_observation_seconds=1.,
        maximum_added_swap_bytes=0,truth_accessed=False,training_authorized_by_admission=False,
        baseline_forwards=20,training_forwards=20,backwards=20,updates=5,terminal_forwards=20,
        native_gradient_certified=False,old_failed_gates_preserved=True,
        native_score_measured=False,automatic_next_stage=False)
