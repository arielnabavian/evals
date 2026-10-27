#!/usr/bin/env bash
# Set up AppWorld for appworld_harness.py. Builds, under APPWORLD_HOME (default ~/appworld):
#
#   venv_server/        AppWorld 0.1.3 (pydantic v1): each trial's world runs in a server started from here
#   client_src/         AppWorld's repository at the client's commit (installed into the study's environment)
#   root/               APPWORLD_ROOT: AppWorld's data (about 200 MB)
#   cache/              APPWORLD_CACHE: what `appworld install` unpacks besides the package (by default it writes to
#                       ~/.appworld or ~/.cache/appworld, outside APPWORLD_HOME)
#   instructions.yaml   AppWorld's canonical code-agent instructions (harness-optimizer's AppWorld runtime prompt)
#
# and installs the AppWorld client (0.2.0, on pydantic 2.12 or 2.13, which Strands also accepts) into the Python
# environment the study runs in (PYTHON, default: python). Run it from that environment:
#
#   python -m pip install -e ".[optimization]" strands-harness
#   bash examples/optimization/appworld_setup.sh
#   export APPWORLD_HOME=~/appworld
#
# Each world's server listens on 127.0.0.1 only (appworld_workload.py forces it, and stops a server that says
# otherwise). Never start it with AppWorld's --docker, which publishes the port on every interface.
set -euo pipefail

APPWORLD_HOME=${APPWORLD_HOME:-$HOME/appworld}
PYTHON=${PYTHON:-python}
SERVER_PYTHON=${SERVER_PYTHON:-3.12}  # the server venv's Python version: 3.11 or 3.12 (AppWorld 0.1.3 needs 3.11+)
# The client: AppWorld 0.2.0 is not on PyPI. This is the commit the experiments ran, with the one change they made to
# it: it reads data version 0.1.0, the data AppWorld 0.1.3 downloads and its server serves (the commit expects 0.2.0).
# Its app bundles are Git LFS files, so the setup needs git-lfs (apt install git-lfs, brew install git-lfs).
CLIENT_REPO=https://github.com/StonyBrookNLP/appworld
CLIENT_COMMIT=afc941710b91ab9260b95b7bdebb1179deba90cf
INSTRUCTIONS="https://raw.githubusercontent.com/strands-labs/harness-optimizer/780d629f8ba70c46cde864780c5b939382c72b2f/examples/appworld/appworld_runtime/prompts/appworld_code_instructions.yaml"

command -v git-lfs >/dev/null || { echo "git-lfs is needed (the AppWorld client's app bundles are Git LFS files)"; exit 1; }
mkdir -p "$APPWORLD_HOME/root" "$APPWORLD_HOME/cache"
cd "$APPWORLD_HOME"
export APPWORLD_CACHE="$APPWORLD_HOME/cache"

echo "== the server's venv: AppWorld 0.1.3 ($APPWORLD_HOME/venv_server)"
if [ ! -x venv_server/bin/python ]; then  # with pip in it: by uv when it is installed, else by venv
  if command -v uv >/dev/null; then uv venv -q --seed --python "$SERVER_PYTHON" venv_server; else "python$SERVER_PYTHON" -m venv venv_server; fi
fi
venv_server/bin/python -m pip install -q --upgrade pip
venv_server/bin/python -m pip install -q "appworld==0.1.3" "typer==0.12.5" "click==8.0.0"
(cd root && APPWORLD_ROOT="$PWD" ../venv_server/bin/appworld install)
if [ ! -d root/data/tasks ]; then
  (cd root && APPWORLD_ROOT="$PWD" ../venv_server/bin/appworld download data --root .)
fi

echo "== the client, in the study's environment ($("$PYTHON" -c 'import sys; print(sys.prefix)'))"
[ -d client_src ] || GIT_LFS_SKIP_SMUDGE=1 git clone -q "$CLIENT_REPO" client_src
git -C client_src checkout -q -f "$CLIENT_COMMIT"
git -C client_src lfs install --local >/dev/null
git -C client_src lfs pull --include "src/appworld/.source/*.bundle"
CONSTANTS=client_src/src/appworld/common/constants.py
sed -i.orig 's/^DATA_VERSION = "0.2.0"$/DATA_VERSION = "0.1.0"/' "$CONSTANTS" && rm -f "$CONSTANTS.orig"
grep -q '^DATA_VERSION = "0.1.0"$' "$CONSTANTS" || { echo "could not set the client's data version in $CONSTANTS"; exit 1; }
"$PYTHON" -m pip install -q ./client_src
"$PYTHON" -m pip install -q "pydantic>=2.12,<2.14"  # the client needs 2.12+; appworld install fails on 2.14
CLIENT_BIN="$("$PYTHON" -c 'import sys, os; print(os.path.join(sys.prefix, "bin"))')"
(cd root && APPWORLD_ROOT="$PWD" "$CLIENT_BIN/appworld" install)

echo "== AppWorld's code-agent instructions"
curl -fsSL "$INSTRUCTIONS" -o instructions.yaml

echo "done: export APPWORLD_HOME=$APPWORLD_HOME"
