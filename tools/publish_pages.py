#!/usr/bin/env python3
"""Publish the self-contained graph and paper over the repository's Git connection.

Usage: python3 -I tools/publish_pages.py [--dry-run]
Only index.html, WRITEUP.pdf and .nojekyll are staged on gh-pages. The main
checkout is unchanged, existing branch history is retained, and no force push
is used. GitHub Pages must publish gh-pages from /(root).
"""
import argparse
import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SITE = 'https://mattliston.github.io/arc-whitebox-estimation-challenge/'


def git(directory, *args, check=True):
    return subprocess.run(['git', '-C', str(directory), *args], check=check,
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='check publication inputs without pushing')
    args = parser.parse_args()
    inputs = {'index.html': ROOT / 'graph/index.html', 'WRITEUP.pdf': ROOT / 'WRITEUP.pdf'}
    for destination, source in inputs.items():
        data = source.read_bytes()
        if not data or (destination.endswith('.pdf') and not data.startswith(b'%PDF-')):
            raise ValueError(f'Invalid publication input: {source}')
        print(f'{destination}: {len(data):,} bytes, SHA-256 {hashlib.sha256(data).hexdigest()}')
    if args.dry_run:
        print('Inputs valid. No Git changes or network requests made.')
        return

    remote = git(ROOT, 'remote', 'get-url', '--push', 'origin').stdout.strip()
    revision = git(ROOT, 'rev-parse', '--short', 'HEAD').stdout.strip()
    email = git(ROOT, 'config', 'user.email').stdout.strip()
    branch = git(ROOT, 'ls-remote', '--exit-code', '--heads', 'origin', 'gh-pages', check=False)
    if branch.returncode not in (0, 2):
        raise RuntimeError(branch.stderr.strip())

    with tempfile.TemporaryDirectory(prefix='arc-pages-') as temporary:
        site = Path(temporary)
        git(site, 'init', '--initial-branch=gh-pages')
        git(site, 'config', 'user.name', 'Matt Liston')
        git(site, 'config', 'user.email', email)
        git(site, 'remote', 'add', 'origin', remote)
        if branch.returncode == 0:
            git(site, 'fetch', '--depth=1', 'origin', 'gh-pages')
            git(site, 'checkout', '-B', 'gh-pages', 'FETCH_HEAD')
        for destination, source in inputs.items():
            shutil.copyfile(source, site / destination)
        (site / '.nojekyll').touch()
        git(site, 'add', '--', 'index.html', 'WRITEUP.pdf', '.nojekyll')
        unchanged = git(site, 'diff', '--cached', '--quiet', check=False)
        if unchanged.returncode == 0:
            print('The published branch already contains these files.')
            return
        if unchanged.returncode != 1:
            raise RuntimeError(unchanged.stderr.strip())
        git(site, 'commit', '-m', f'Publish research graph and Phase 2 paper from {revision}')
        result = git(site, 'push', 'origin', 'gh-pages:gh-pages')
        print(result.stdout + result.stderr)
    print(f'Publication branch pushed. Check deployment at {SITE}')
    print('If Pages is not enabled, select gh-pages / (root) in repository Settings > Pages.')


if __name__ == '__main__':
    main()
