#!/usr/bin/env bash
# Draft only; never invoked by this audit. Does not provision a host or download packages.
# Requires pre-provisioned CPython 3.10.20 and a vetted Linux x86_64 offline wheelhouse.
set -euo pipefail
if [[ "${1:-}" != "--install-offline-environments" ]]; then
  echo 'DRAFT: review PORTABILITY_PLAN_v1.md. To install later, pass --install-offline-environments with CLOUD_PYTHON, CLOUD_ENV_ROOT and CLOUD_WHEELHOUSE set.'
  exit 0
fi
: "${CLOUD_PYTHON:?Set to Linux CPython 3.10.20 executable}"
: "${CLOUD_ENV_ROOT:?Set to a new empty environment directory}"
: "${CLOUD_WHEELHOUSE:?Set to vetted Linux x86_64 wheels; no network is used}"
cloud_draft_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
"$CLOUD_PYTHON" -c 'import platform,sys; assert sys.version_info[:3]==(3,10,20); assert platform.system()=="Linux"; assert platform.machine()=="x86_64"'
test ! -e "$CLOUD_ENV_ROOT"
mkdir -p "$CLOUD_ENV_ROOT"
for cloud_role in client server judge; do
  "$CLOUD_PYTHON" -m venv "$CLOUD_ENV_ROOT/$cloud_role"
  "$CLOUD_ENV_ROOT/$cloud_role/bin/python" -m pip install --no-index --find-links "$CLOUD_WHEELHOUSE" -r "$cloud_draft_dir/requirements_${cloud_role}_v1.txt"
done
"$CLOUD_ENV_ROOT/client/bin/python" -c 'import importlib.util; assert importlib.util.find_spec("numpy") is None'
echo 'Environment installation only. Linux worker/guard port and paired calibration are still required. No estimator or benchmark has been launched.'
