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

export TCL_LIBRARY="${TCL_LIBRARY:-$(python - <<'PY'
import pathlib
import tkinter

candidate = pathlib.Path(tkinter.Tcl().eval("info library"))
if candidate.is_dir() and (candidate / "init.tcl").exists():
    print(candidate)
    raise SystemExit(0)

for root in (pathlib.Path("/usr"), pathlib.Path("/opt/hostedtoolcache")):
    for item in root.glob("**/tcl8.*/init.tcl"):
        print(item.parent)
        raise SystemExit(0)

raise SystemExit("Could not infer TCL_LIBRARY.")
PY
)}"

export TK_LIBRARY="${TK_LIBRARY:-$(python - <<'PY'
import pathlib
import re
import subprocess
import _tkinter

linked = subprocess.check_output(["ldd", pathlib.Path(_tkinter.__file__).resolve()], text=True)
match = re.search(r"libtk(?:\d+\.\d+)?\.so(?:\.\d+)? => (/\S+)", linked)
if match:
    lib = pathlib.Path(match.group(1)).resolve()
    for parent in (lib.parent, *lib.parents):
        for item in parent.glob("**/tk8.*/tk.tcl"):
            print(item.parent)
            raise SystemExit(0)

for root in (pathlib.Path("/usr"), pathlib.Path("/opt/hostedtoolcache")):
    for item in root.glob("**/tk8.*/tk.tcl"):
        print(item.parent)
        raise SystemExit(0)

raise SystemExit("Could not infer TK_LIBRARY.")
PY
)}"

if [[ ! -f "$TCL_LIBRARY/init.tcl" ]]; then
  echo "Invalid TCL_LIBRARY: $TCL_LIBRARY" >&2
  exit 1
fi

if [[ ! -f "$TK_LIBRARY/tk.tcl" ]]; then
  echo "Invalid TK_LIBRARY: $TK_LIBRARY" >&2
  exit 1
fi

echo "Using TCL_LIBRARY=$TCL_LIBRARY"
echo "Using TK_LIBRARY=$TK_LIBRARY"

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
