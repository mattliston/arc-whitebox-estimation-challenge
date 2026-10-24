"""Versioned panel supervisor; original limits, immutable v1 failure."""
from pathlib import Path
import panel_common_v2 as c

OWNER = c.HERE/'panel_watch.py'
OWNER_SHA = '56793bf236fb43ef363abbff7cf8d323a828715947a625ec01b4ecd1c469335f'


def resolved_source():
    c.require(c.sha(OWNER) == OWNER_SHA, 'Exact preserved v1 supervisor source')
    return c.splice(OWNER.read_text(), [
        ('import panel_common as c\n', 'c = REVIEWED_PANEL_V2_METADATA\n'),
        ('EXTRA_MAIN_EDITS = (\n',
         'EXTRA_MAIN_EDITS = (\n'
         '    ("        e.write(e.WATCH_OUT/\'result.json\',result)\\n",\n'
         '     "        if result[\'passed\']:\\n"\n'
         '     "            try:\\n"\n'
         '     "                result.update(e.final_bind((sources,inputs,result[\'artifacts\']),deadline))\\n"\n'
         '     "                result[\'elapsed_s\'] = time.monotonic()-began\\n"\n'
         '     "            except BaseException:\\n"\n'
         '     "                result.update(passed=False,completed=False,final_union_error=traceback.format_exc())\\n"\n'
         '     "        e.write(e.WATCH_OUT/\'result.json\',result)\\n"),\n'),
    ])


def main():
    source = resolved_source()
    namespace = dict(__file__=str(Path(__file__).resolve()),
        __name__='cd_panel_v2_original_supervisor', REVIEWED_PANEL_V2_METADATA=c)
    exec(compile(source, str(OWNER)+':v2-metadata-only', 'exec'), namespace)
    return namespace['main']()


if __name__ == '__main__':
    raise SystemExit(main())
