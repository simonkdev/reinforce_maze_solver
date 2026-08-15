#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BUILD_DIR="$ROOT/packaging/build"
DIST_DIR="$ROOT/packaging/dist"
ARTIFACT_DIR="$ROOT/packaging/artifacts"
VENV_DIR="$BUILD_DIR/venv"

mkdir -p "$BUILD_DIR/tmp" "$ARTIFACT_DIR"
export TMPDIR="$BUILD_DIR/tmp"

python -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r packaging/requirements-packaging.txt

if python -c "import tkinter; tkinter.Tk().destroy()" >/dev/null 2>&1; then
  :
else
  export TCL_LIBRARY="${TCL_LIBRARY:-$(python -c "import tkinter; print(tkinter.Tcl().eval('info library'))")}"
  export TK_LIBRARY="${TK_LIBRARY:-$(python - <<'PY'
import pathlib
import re
import subprocess
import _tkinter

linked = subprocess.check_output(["ldd", pathlib.Path(_tkinter.__file__).resolve()], text=True)
match = re.search(r"libtk\.so => (/.+?)/lib/libtk\.so", linked)
if not match:
    raise SystemExit("Could not infer TK_LIBRARY from _tkinter linkage.")
print(pathlib.Path(match.group(1)) / "lib" / "tk8.6")
PY
)}"
fi

python -m PyInstaller \
  --clean \
  --noconfirm \
  --workpath "$BUILD_DIR/pyinstaller" \
  --distpath "$DIST_DIR" \
  packaging/reinforce_maze_lab.spec

python -m pip install -r packaging/requirements-linux.txt
staticx "$DIST_DIR/reinforce-maze-lab" "$ARTIFACT_DIR/reinforce-maze-lab-linux-x86_64"
chmod +x "$ARTIFACT_DIR/reinforce-maze-lab-linux-x86_64"

if ldd "$ARTIFACT_DIR/reinforce-maze-lab-linux-x86_64" 2>&1 | grep -q "not a dynamic executable"; then
  file "$ARTIFACT_DIR/reinforce-maze-lab-linux-x86_64"
  exit 0
fi

ldd "$ARTIFACT_DIR/reinforce-maze-lab-linux-x86_64"
echo "Linux artifact is still dynamically linked; refusing to publish it." >&2
exit 1
