import ast
from common import *
cp=D/'continuation_checkpoint_v85.json'
assert sha(cp)=='9aa796cf36f4df89e0aac17c17d165d1d196b26969b5289a858bf2e60ced6b9b'
assert sha(SOURCE)=='facd0dd0d3a425237a7285bd597e4f8ddb162f821f55acbba3bbaf136d80b6a0'
prior=read(cp)
for k in ('study','official_bank','workers','candidate','archive','rejection'):verify(prior[k])
bank=D/'official_best_331684.json';assert sha(bank)=='b6ef6e0284d1ac82eced28377b8ccf9846190797b23ad631de5aeed14f760d49'
external=[cp,SOURCE,bank,D/'run_guarded.py',OLD/'census.py',IDENTITY/'closeout.json',IDENTITY/'RESULT.md',
    D/'all_positive_leaf_catalog_v1/RESULT.md',D/'read6_runtime_v1/PROTOCOL.md']
# The Read6 protocol location is resolved explicitly, without importing it.
if not external[-1].exists():external.pop()
rows=[]
for rank in (35,100):
    result=OLD/f'rank{rank:04d}/result.json';r=read(result)
    attr=OLD/f'rank{rank:04d}/attribution.json';assert sha(attr)==r['attribution_sha256']
    ir=IDENTITY/f'rank{rank:04d}/result.json';i=read(ir);verify(i['trace'])
    assert r['completed'] and r['parent_bill_exact'] and r['all16_prediction_bytes_exact']
    assert r['parent_sha256']==sha(SOURCE) and i['all16_bytes_and_complete_bill_exact']
    rows.append(dict(rank=rank,attribution=pin(attr),old_result=pin(result),identity_result=pin(ir),identity_trace=i['trace']))
    external += [attr,result,ir,Path(i['trace']['path'])]
for rel in ('_pointwise.py','_ndarray.py'):
    external.append(A/'openblas_runtime_env_v1/lib/python3.11/site-packages/flopscope'/rel)
review_paths=[A/'first_layer_antithetic_cv_v1/RESULT_2026_09_08.md',
    A/'current_first_layer_quadratic_rqmc_v1/RESULT_CLOSED.md',
    A/'gaussian_terminal_skew_v2/RESULT_2026_09_08.md',
    A/'CUMULATIVE_BACKWARD_SOURCE_V1_CLOSED.md',A/'CP_SUFFIX_RANGE_PRODUCT_SOURCE_AUDIT_V1.md']
external+=review_paths
external += [Path(prior[k]['path']) for k in ('candidate','archive','rejection')]
write(HERE/'inputs.json',dict(rows=rows,official_adjusted_decimal='3.5844144212297072e-9',
    bank=pin(bank),parent=pin(SOURCE),pending_archive=prior['archive'],new_forward_calls=0,targets_read=False))
for p in HERE.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
write(HERE/'sources_sealed.json',dict(pins=[pin(p) for p in sorted(HERE.iterdir()) if p.is_file()]+
    [pin(p) for p in dict.fromkeys(external)],before_aggregate_result=True,no_targets=True))
print(dict(sealed=True,new_native_calls=0),flush=True)
