def mark():
    before = marks.stat().st_size if marks.exists() else 0
    deadline = time.monotonic() + 4000 * .0005
    os.kill(server_pid, signal.SIGUSR1)
    for _ in range(4000):
        if marks.exists() and marks.stat().st_size > before:
            with marks.open('rb') as stream:
                stream.seek(before)
                appended = stream.read()
            if appended.endswith(b'\n'):
                try:
                    return json.loads(appended.splitlines()[-1])
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
        if time.monotonic() >= deadline:
            break
        time.sleep(.0005)
    raise RuntimeError('server did not answer mark signal')
