"""Explicit post-acquisition, target-free pricing; never starts a candidate."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
PINS = {'capture.py':'9a39a004f0fcf74e4afbcf2d3f616c010a009374f18c7202b07831d738299a3a',
        'watch.py':'7f868ba0250f142822b2aa36503d1908b21849c47e6e7406320f979e5d93bbcf',
        'price.py':'ad295d1ffeed6c1ac31abb6d89e57ba7d788d0027160dcf9aaddb60c2546fec3',
        'PROTOCOL.md':'57dd4ccedd93c9626a52f0864693a298756bc5e2a06ee2537de99a0ebc4c2dc8'}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    path = Path(path)
    require('truthpost' not in path.parts and 'truth' not in path.name.lower(), 'No targets')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def bind(mapping):
    for path, digest in mapping.items():
        require(sha(path) == digest, 'Preserved artifact: '+str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    for name in ('source','native','watch'):
        parser.add_argument('--'+name+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    began = time.monotonic()
    require(os.getpriority(os.PRIO_PROCESS,0) >= 10 and sys.flags.optimize == 0,
            'Nice10, assertions enabled')
    require(not any(x in sys.modules for x in ('numpy','torch','flopscope')), 'Stdlib only')
    sources = {str(HERE/name):digest for name,digest in PINS.items()}
    sources[str(Path(__file__).resolve())] = args.source_sha256
    bind(sources)
    native,watch = HERE/'native',HERE/'native_watch'
    inputs = {str(native/'result.json'):args.native_sha256,
              str(watch/'result.json'):args.watch_sha256}
    bind(inputs)
    nr,wr = read(native/'result.json'),read(watch/'result.json')
    for folder,record,digest in ((native,nr,args.native_sha256),(watch,wr,args.watch_sha256)):
        end = read(folder/'completion.json')
        inputs[str(folder/'completion.json')] = sha(folder/'completion.json')
        require(all(record.get(k) is True for k in ('passed','completed','sources_unchanged','hashes_unchanged'))
                and end['passed'] and end['completed'] and end['result_sha256'] == digest,
                'Sealed successful acquisition and independent resource watch')
        for key in ('source_bindings','input_bindings','artifacts'):
            bind(record.get(key,{}))
    require(nr['schema'] == 'current_d50_response10_census_native_v1'
            and wr['schema'] == 'current_d50_response10_census_watch_v1'
            and wr['child_result_sha256'] == args.native_sha256
            and wr['whole_owned_tree_retired'] and wr['completed_bill_count'] == 2
            and nr['truth_accessed'] is False and nr['alternate_estimator_executed'] is False,
            'Only completed target-free cold/repeat bank census')
    records = []
    require(len(nr['census_records']) == 2, 'Exactly two measured calls')
    for index,item in enumerate(nr['census_records']):
        path = native/f'call_{index:02d}_census.json'
        require(item['path'] == str(path) and sha(path) == item['sha256'], 'Fixed record ownership')
        record = read(path)
        require(record['schema'] == 'current_d50_response10_census_record_v1'
                and record['added_participant_array_operations'] == 0
                and record['truth_accessed'] is False and record['alternate_estimator_executed'] is False
                and record['actual_flops'] == nr['trials'][index]['flops'], 'Measured unchanged call')
        require(sha(record['score_arrays']) == record['score_arrays_sha256'], 'Exact observed score bytes')
        records.append(record)
    out = HERE/'price_v1'
    require(not out.exists() and not out.is_symlink(), 'Exclusive price stage, no retry')
    out.mkdir()
    result = dict(schema='current_d50_response10_price_stage_v1', passed=False,completed=False,
        source_bindings=sources,input_bindings=inputs,started_utc=datetime.now(timezone.utc).isoformat(),
        truth_accessed=False,candidate_executed=False,official_score=False,automatic_next_stage=False)
    write(out/'started.json',result)
    def expired(signum,frame):
        raise RuntimeError('60 second stdlib price-stage deadline')
    previous = signal.signal(signal.SIGALRM,expired)
    signal.alarm(60)
    try:
        spec = importlib.util.spec_from_file_location('fixed_response_price',HERE/'price.py')
        price = importlib.util.module_from_spec(spec); spec.loader.exec_module(price)
        result['price'] = price.price_pair(records)
        require(not any(x in sys.modules for x in ('numpy','torch','flopscope')), 'No numerical imports')
        bind(sources); bind(inputs)
        for record in (nr,wr):
            for key in ('source_bindings','input_bindings','artifacts'):
                bind(record.get(key,{}))
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        require(peak < 1_000_000_000 and time.monotonic()-began < 60, 'Small bounded stdlib phase')
        result.update(passed=True,completed=True,hashes_unchanged=True,peak_rss_bytes=peak)
    except BaseException:
        result.update(passed=False,completed=False,error=traceback.format_exc())
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM,previous)
        result.update(elapsed_s=time.monotonic()-began,finished_utc=datetime.now(timezone.utc).isoformat())
        write(out/'result.json',result)
        write(out/'completion.json',dict(passed=result['passed'],completed=result['completed'],
            result_sha256=sha(out/'result.json')))
    print(json.dumps(result,allow_nan=False),flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
