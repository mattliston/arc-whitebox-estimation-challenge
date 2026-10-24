from common import HERE, RUNTIME, read, write, pin, guarded, seal_check

seal_check()
guarded('unit', 90, [RUNTIME, '-B', '-W', 'ignore', HERE/'unit.py'])
for label in ['base', 'plus', 'minus', 'double']:
    guarded('capture_'+label, 120, [RUNTIME, '-B', '-W', 'ignore', HERE/'capture.py', label, 'capture_'+label])
guarded('capture_base_repeat', 120, [RUNTIME, '-B', '-W', 'ignore', HERE/'capture.py', 'base', 'capture_base_repeat'])
write(HERE/'captures_sealed.json', dict(rows=[pin(HERE/f'capture_{label}/result.json')
      for label in ['base','plus','minus','double','base_repeat']], targets_opened=False))
guarded('analyze', 300, [RUNTIME, '-B', '-W', 'ignore', HERE/'analyze.py'])
write(HERE/'completed.json', dict(completed=True, analysis=pin(HERE/'analysis.json'), targets_opened=False))
