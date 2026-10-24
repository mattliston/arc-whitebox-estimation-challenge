"""Inert compatibility adapter for the frozen fixed-40 terminal judge.

--judge retains the four original result-SHA arguments and requires
--development-sha256. Only completed validation_bank_v2, validation_zero_v2,
validation_terminal_v2 are scored into exclusive terminal_score_v2. Original
training, checkpoint, all40 target hashes, actual per-net bill arithmetic,
strict mean-adjusted gain and >=24/40 wins against BOTH comparators are
unchanged. Adds official-input/reset receipt verification and byte/+179306496F
checks of ALL53 recorded qualified v1 overlaps; never scores those partials.
No estimator execution, fitting, or truth access occurs on import/help.
"""
import hashlib
from pathlib import Path

import qualify_native_v2 as v2

HERE = Path(__file__).resolve().parent
PARENT = HERE / 'judge_terminal.py'
PARENT_SHA = '4f4ccb45d75f52c36530cc1266847334c84014ea60eb25d933d03aa933420c0b'

# Exact one-occurrence substitutions only. The scored formulas and terminal
# checkpoint checks are not copied, rewritten, patched, or reimplemented.
REPLACEMENTS = (
    ("    parser.add_argument('--judge', action='store_true')\n",
     "    parser.add_argument('--judge', action='store_true')\n"
     "    parser.add_argument('--development-sha256', help='Passing native_v2_qualification result SHA256')\n"),
    ("    out = HERE / 'terminal_score_v1'\n", "    out = HERE / 'terminal_score_v2'\n"),
    ("schema='birth_direction_terminal_native40_v1'", "schema='birth_direction_terminal_native40_v2'"),
    ("        path = str(Path(path).resolve())\n", "        path = str(native_path(path))\n"),
    ("        pin(Path(__file__))\n",
     "        pin(Path(__file__))\n"
     "        development = verify_development(pin, args.development_sha256)\n"
     "        result['development_qualification_sha256'] = args.development_sha256\n"
     "        result['development_qualification'] = dict(records=development['records'],\n"
     "            initialization_flops=development['initialization_flops'], truth_accessed=False)\n"),
    ("            folder = HERE / f'native_validation_{role}_v1'\n",
     "            folder = HERE / f'validation_{role}_v2'\n"),
    ("            for p in (source, HERE / 'native.py', HERE / 'PROTOCOL.md',\n",
     "            for p in (source, HERE / 'native_v2.py', HERE / 'PROTOCOL.md',\n"),
    ("Path(trial['output']).resolve() == output", "native_path(trial['output']) == output"),
    ("            require({str(Path(p).resolve()): h for p, h in acq['artifacts'].items()} == expected_artifacts,\n"
     "                    'Exact native artifact set')\n",
     "            verify_v2_artifacts(acq, folder, expected_artifacts, pin)\n"),
    ("        def output(path, bytes_sha=None):\n",
     "        result['v1_qualified_overlap_records'] = verify_v1_overlaps(acquisitions, pin)\n"
     "        result['v1_partial_scores_used'] = False\n"
     "        result['official_input_complete_bills_only'] = True\n"
     "\n"
     "        def output(path, bytes_sha=None):\n"),
    ("            array = np.load(path, allow_pickle=False)\n",
     "            array = np.load(native_path(path), allow_pickle=False)\n"),
)


def transformed_source():
    original = PARENT.read_text()
    v2.require(hashlib.sha256(PARENT.read_bytes()).hexdigest() == PARENT_SHA, 'Frozen old judge source')
    changed = original
    for old, new in REPLACEMENTS:
        v2.require(changed.count(old) == 1, 'One exact old judge splice: ' + old[:90])
        changed = changed.replace(old, new, 1)
    restored = changed
    for old, new in reversed(REPLACEMENTS):
        v2.require(restored.count(new) == 1, 'One exact inverse judge splice')
        restored = restored.replace(new, old, 1)
    v2.require(restored == original, 'Literal inverse: every other source byte preserved')
    return changed


def verify_v2_artifacts(acq, folder, expected_outputs, pin):
    outputs = v2.verify_native(acq, folder, v2.VALID, pin)
    v2.require(len(outputs) == 41, 'Complete validation plus repeat')
    actual = {str(v2.path(t['output'])): t['output_sha256'] for t in acq['trials']}
    v2.require(actual == expected_outputs, 'Original output artifact ownership retained')


def main():
    namespace = dict(__name__='birth_direction_frozen_judge_v2', __file__=str(Path(__file__).resolve()),
                     native_path=v2.path, verify_development=v2.verify_development,
                     verify_v2_artifacts=verify_v2_artifacts, verify_v1_overlaps=v2.verify_v1_overlaps)
    source = transformed_source()
    exec(compile(source, str(PARENT), 'exec'), namespace)
    namespace['FIXED'].update(v2.FIXED)
    namespace['FIXED'][PARENT] = PARENT_SHA
    namespace['FIXED'][HERE / 'qualify_native_v2.py'] = v2.sha(HERE / 'qualify_native_v2.py')
    return namespace['main']()


if __name__ == '__main__':
    raise SystemExit(main())
