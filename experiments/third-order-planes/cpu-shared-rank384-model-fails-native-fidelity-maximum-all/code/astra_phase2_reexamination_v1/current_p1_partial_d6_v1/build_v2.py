"""Inert, reversible UTF-8 span-cache repair of the frozen source builder."""
import ast
from functools import lru_cache
import hashlib
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
OUT = HERE/'build_v2'
ORIGINAL = HERE/'build.py'
ORIGINAL_SHA = 'c7afaa3a4fe347b0aeac2c0181ebb7ea76512d2910b26cc813c0375744be3138'
PROTOCOL_SHA = 'b4fc1edd8cabec7b59d11ede87de42ce3f3c142aa0b142582bcccac8bdcdaff9'
AMENDMENT = HERE/'SOURCE_BUILD_V2.md'
AMENDMENT_SHA = 'a8bb785fdedc3678a2ec44478be8d338cbee057795646dbd1a99ce6d70137cc8'
FAILED_RESULT = HERE/'build_v1/result.json'
FAILED_RESULT_SHA = '4da862eb19744eb0de1d895d492a297832171d84132a6cde6631ea5ddc4f728f'
FAILED_COMPLETION = HERE/'build_v1/completion.json'
FAILED_COMPLETION_SHA = 'e67c1d1154f1c19d33ac8ba3531a86d187c5e29c36b8259a4b707dba9d099b40'


def _need(value, message):
    if not value:
        raise RuntimeError(message)


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=8)
def _source_index(source):
    """One immutable UTF-8 copy; only Python-parser line boundaries count."""
    data = source.encode('utf-8')
    starts = (0,)+tuple(match.end() for match in re.finditer(rb'\r\n|\r|\n',data))
    return data,starts


def _source_span(source, node):
    data,starts = _source_index(source)
    first,last = node.lineno,node.end_lineno
    left,right = node.col_offset,node.end_col_offset
    _need(all(type(x) is int for x in (first,last,left,right)) and
          1 <= first <= last <= len(starts) and left >= 0 and right >= 0,
          'Complete valid AST byte locations')
    start,end = starts[first-1]+left,starts[last-1]+right
    _need(0 <= start <= end <= len(data), 'AST byte span lies inside its immutable source')
    return data,start,end


def _source_segment(source, node):
    """Equivalent to unpadded ast.get_source_segment for complete AST nodes."""
    try:
        if node.end_lineno is None or node.end_col_offset is None:
            return None
        node.lineno; node.col_offset
    except AttributeError:
        return None
    data,start,end = _source_span(source,node)
    return data[start:end].decode('utf-8')


def _check_spans():
    # Fixed tiny text only: no execution of these parsed statements.
    base = 'é = (1 +\n    2)\n\fα = "😀\u2028"\ndef f():\n    return α'
    for ending in ('\n','\r','\r\n'):
        source = base.replace('\n',ending)
        for node in ast.walk(ast.parse(source)):
            _need(_source_segment(source,node) == ast.get_source_segment(source,node),
                  'Cached span agrees with stdlib on the fixed Unicode/newline fixture')
        boundary = ast.Pass(lineno=1,col_offset=0,end_lineno=2,end_col_offset=0)
        _need(_source_segment(source,boundary) == ast.get_source_segment(source,boundary),
              'Exact next-line column-zero endpoint agrees with stdlib')
        source_with_final_newline = source+ending
        last = ast.parse(source_with_final_newline).body[-1]
        _need(_source_segment(source_with_final_newline,last) ==
              ast.get_source_segment(source_with_final_newline,last),
              'Exact final newline boundary agrees with stdlib')


def derived_source():
    """Authenticate and invert the adapter without importing any estimator."""
    for path,digest in ((ORIGINAL,ORIGINAL_SHA),(HERE/'PROTOCOL.md',PROTOCOL_SHA),
                        (AMENDMENT,AMENDMENT_SHA)):
        _need(_digest(path) == digest, 'Frozen builder/protocol/amendment: '+str(path))
    original = ORIGINAL.read_text()
    _need(all(re.search(r'\b'+re.escape(name)+r'\b',original) is None for name in
              ('_source_segment','_source_span','_source_build_lineage')),
          'Adapter helper names do not replace any original builder owner')
    old_span = '''    lines = source.splitlines(keepends=True)
    start = sum(map(len,lines[:node.lineno-1]))+node.col_offset
    end = sum(map(len,lines[:node.end_lineno-1]))+node.end_col_offset
    before = source[start:end]
    require(before == _source_segment(source,node), 'Exact AST-selected source span')
    changed = source[:start]+replacement+source[end:]'''
    new_span = '''    data,start,end = _source_span(source,node)
    before = data[start:end].decode('utf-8')
    require(before == _source_segment(source,node), 'Exact AST-selected source span')
    changed = (data[:start]+replacement.encode('utf-8')+data[end:]).decode('utf-8')'''
    edits = (
        ("OUT = HERE/'build_v1'","OUT = HERE/'build_v2'",1),
        ('current_p1_partial_d6_build_v1','current_p1_partial_d6_build_v2',2),
        ('ast.get_source_segment(','_source_segment(',9),
        (old_span,new_span,1),
        ('result.update(source_bindings=sources,input_bindings=inputs)',
         'result.update(source_bindings=sources,input_bindings=inputs,source_build_lineage=_source_build_lineage())',1),
    )
    changed = original
    for before,after,count in edits:
        _need(changed.count(before) == count, 'Exact frozen adapter span/count')
        changed = changed.replace(before,after)
    inverse = changed
    for before,after,count in reversed(edits):
        _need(inverse.count(after) == count, 'Exact inverse adapter span/count')
        inverse = inverse.replace(after,before)
    _need(inverse == original and hashlib.sha256(inverse.encode()).hexdigest() == ORIGINAL_SHA,
          'Whole builder source inverse, with every original proof and guard retained')
    compile(changed,str(Path(__file__).resolve())+':derived','exec')
    return changed,dict(original_builder_sha256=ORIGINAL_SHA,
        derived_builder_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        whole_builder_source_inverse_verified=True)


@lru_cache(maxsize=1)
def _owner():
    source,adapter_proof = derived_source()
    lineage = dict(adapter_proof,protocol_sha256=PROTOCOL_SHA,
        amendment_path=str(AMENDMENT),amendment_sha256=AMENDMENT_SHA,
        failed_v1_result_path=str(FAILED_RESULT),failed_v1_result_sha256=FAILED_RESULT_SHA,
        failed_v1_completion_path=str(FAILED_COMPLETION),
        failed_v1_completion_sha256=FAILED_COMPLETION_SHA,
        preserved_v1_status='failed',span_cache_utf8_byte_offsets=True,span_cache_max_sources=8)
    namespace = dict(__file__=str(Path(__file__).resolve()),__name__='partial_d6_source_builder_v2_owner',
        _source_segment=_source_segment,_source_span=_source_span,
        _source_build_lineage=lambda:dict(lineage))
    exec(compile(source,str(Path(__file__).resolve())+':derived','exec'),namespace)
    namespace['SOURCE_PINS'].update({ORIGINAL:ORIGINAL_SHA,HERE/'PROTOCOL.md':PROTOCOL_SHA,
                                     AMENDMENT:AMENDMENT_SHA})
    namespace['INPUT_PINS'].update({FAILED_RESULT:FAILED_RESULT_SHA,
                                    FAILED_COMPLETION:FAILED_COMPLETION_SHA})
    original_admission = namespace['admission']

    def admission(source_sha, protocol_sha):
        _need(protocol_sha == PROTOCOL_SHA, 'Unchanged scientific protocol; additive span-cache amendment only')
        sources,inputs = original_admission(source_sha,protocol_sha)
        failed,end = namespace['read'](FAILED_RESULT),namespace['read'](FAILED_COMPLETION)
        _need(all(failed.get(k) is False for k in ('passed','completed','hashes_unchanged',
              'sources_unchanged','estimator_imported','numerical_execution','truth_accessed')) and
              end.get('passed') is False and end.get('completed') is False and
              end['result_sha256'] == FAILED_RESULT_SHA and
              'Source-only build60s/2GB guard' in failed['error'] and
              'ast.get_source_segment' in failed['error'] and
              not any((HERE/'build_v1'/name).exists() for name in ('estimator.py','null_estimator.py')),
              'Preserved source-only timeout: no emitted v1 candidate and no numerical execution')
        _check_spans()
        return sources,inputs

    namespace['admission'] = admission
    namespace['_lineage'] = lineage
    return namespace


def construct():
    return _owner()['construct']()


def verify_build(build_sha, candidate_sha, builder_sha, protocol_sha):
    owner = _owner()
    context = owner['verify_build'](build_sha,candidate_sha,builder_sha,protocol_sha)
    _need(context['build'].get('source_build_lineage') == owner['_lineage'],
          'Exact additive v2 lineage in completed build')
    context['source_build_lineage'] = dict(owner['_lineage'])
    return context


def main():
    return _owner()['main']()


if __name__ == '__main__':
    raise SystemExit(main())
